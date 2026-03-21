import sys
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
import laspy
from scipy.spatial import cKDTree
import time
from collections import deque
import os
from newton import point_to_curve_distance_newton
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QPushButton, QFileDialog,
    QLabel, QHBoxLayout, QColorDialog,
    QDoubleSpinBox, QToolButton, QFrame, QMessageBox
)
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt + PyVista PointCloud Platform")
        self.resize(1400, 900)
        self.seed_id = set()
        central = QWidget()
        self.setCentralWidget(central)
        self.radus = 0.15
        layout = QHBoxLayout(central)
        self.extract_mode = "point"   # "point" 或 "curve"
        # 左侧工具栏
        left_panel = QVBoxLayout()
        self.load_btn = QPushButton("加载点云")
        self.info_label = QLabel("没有选中任何种子点")
        self.color_btn = QPushButton("Palette Color")
        self.extract_btn = QPushButton("Extract Wire")
        self.delete_btn = QPushButton("Delete Wire")
        self.undo_btn = QPushButton("Undo Delete")
        self.save_btn = QPushButton("Save Point Cloud")
        # 半径输入框
        self.radius_box = QDoubleSpinBox()
        self.radius_box.setRange(0.01, 5.0)
        self.radius_box.setSingleStep(0.05)
        self.radius_box.setValue(self.radus)
        self.radius_box.valueChanged.connect(self.update_radius)




        self.delete_btn.clicked.connect(self.delete_wire)
        self.extract_btn.clicked.connect(self.extract_wire)
        self.color_btn.clicked.connect(self.change_point_color)
        self.undo_btn.clicked.connect(self.undo_last)
        self.save_btn.clicked.connect(self.save_point_cloud)
        self.load_btn.clicked.connect(self.load_point_cloud)

        left_panel.addWidget(self.delete_btn)
        left_panel.addWidget(self.extract_btn)
        left_panel.addWidget(self.load_btn)
        left_panel.addWidget(self.color_btn)
        left_panel.addWidget(self.undo_btn)
        left_panel.addWidget(self.save_btn)
        left_panel.addWidget(QLabel("Wire Radius"))
        left_panel.addWidget(self.radius_box)
        left_panel.addWidget(self.info_label)


        left_panel.addStretch()

       
        # 3D 视图
        self.plotter = QtInteractor(self)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        point_icon_path = os.path.join(base_dir, "icons", "point.png")
        curve_icon_path = os.path.join(base_dir, "icons", "curve.png")

        print("脚本目录:", base_dir)
        print("point路径:", point_icon_path)
        print("curve路径:", curve_icon_path)
        print("point存在:", os.path.exists(point_icon_path))
        print("curve存在:", os.path.exists(curve_icon_path))

        self.mode_point_btn = QToolButton(self.plotter)
        self.mode_point_btn.setIcon(QIcon(point_icon_path))
        self.mode_point_btn.setIconSize(QSize(30, 30))
        self.mode_point_btn.setFixedSize(44, 44)
        self.mode_point_btn.setFocusPolicy(Qt.NoFocus)
        self.mode_point_btn.setToolTip("单点模式")
        self.mode_point_btn.clicked.connect(self.set_point_mode)

        self.mode_curve_btn = QToolButton(self.plotter)
        self.mode_curve_btn.setIcon(QIcon(curve_icon_path))
        self.mode_curve_btn.setIconSize(QSize(30, 30))
        self.mode_curve_btn.setFixedSize(44, 44)
        self.mode_curve_btn.setFocusPolicy(Qt.NoFocus)
        self.mode_curve_btn.setToolTip("曲线模式")
        self.mode_curve_btn.clicked.connect(self.set_curve_mode)

        # ===== 设置样式 =====
        self.normal_btn_style = """
        QToolButton {
            background-color: yellow;
            border: none;
            outline: none;
            border-radius: 0px;
            padding: 4px;
        }
        """

        self.active_btn_style = """
        QToolButton {
            background-color: lime;
            border: none;
            outline: none;
            border-radius: 0px;
            padding: 4px;
        }
        """
        self.mode_point_btn.setStyleSheet(self.active_btn_style)   # 默认point模式高亮
        self.mode_curve_btn.setStyleSheet(self.normal_btn_style)

        # 初始位置（先写死，后面 resizeEvent 里动态调整）
        self.mode_point_btn.move(20, 60)
        self.mode_curve_btn.move(20, 120)

        self.mode_point_btn.show()
        self.mode_curve_btn.show()

        layout.addLayout(left_panel, 1)
        layout.addWidget(self.plotter, 4)
 


        self.points_actor = None
        self.undo_stack = []
        self.color = None

    def update_mode_buttons(self):
        if self.extract_mode == "point":
            self.mode_point_btn.setStyleSheet(self.active_btn_style)
            self.mode_curve_btn.setStyleSheet(self.normal_btn_style)
        else:
            self.mode_point_btn.setStyleSheet(self.normal_btn_style)
            self.mode_curve_btn.setStyleSheet(self.active_btn_style)

    def set_point_mode(self):
        self.extract_mode = "point"
        print("当前模式：单点提取")
        self.update_mode_buttons()

    def set_curve_mode(self):
        self.extract_mode = "curve"
        print("当前模式：曲线/多点提取")
        self.update_mode_buttons()

    def save_point_cloud(self):
        if self.current_cloud is None:
            print("当前没有点云可保存")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Point Cloud",
            "",
            "PointCloud (*.las *.ply *.pcd *.xyz)"
        )

        if not path:
            return

        # 只保存当前有效点
        valid_indices = np.where(self.valid_mask)[0]

        if len(valid_indices) == 0:
            print("没有可保存的点")
            return

        save_points = self.current_points[valid_indices]

        # ======================
        # 保存 LAS
        # ======================
        if path.lower().endswith(".las"):
            header = laspy.LasHeader(point_format=3, version="1.2")
            las = laspy.LasData(header)

            las.x = save_points[:, 0]
            las.y = save_points[:, 1]
            las.z = save_points[:, 2]

            # 如果有颜色就保存颜色
            if self.color_array_name is not None and self.color_array_name in self.current_cloud.array_names:
                colors = self.current_cloud[self.color_array_name][valid_indices]

                # 兼容 rgb / rgba
                if colors.ndim == 2 and colors.shape[1] >= 3:
                    rgb = colors[:, :3].copy()

                    # 如果颜色是 0~1 浮点，转成 LAS 常用 0~65535
                    if np.max(rgb) <= 1.0:
                        rgb = (rgb * 65535).clip(0, 65535).astype(np.uint16)
                    else:
                        rgb = rgb.clip(0, 65535).astype(np.uint16)

                    las.red = rgb[:, 0]
                    las.green = rgb[:, 1]
                    las.blue = rgb[:, 2]

            las.write(path)
            print(f"点云已保存到: {path}")
            return

        # ======================
        # 保存 PLY / PCD / XYZ
        # ======================
        save_cloud = pv.PolyData(save_points)

        # 如果有颜色则一起写入
        if self.color_array_name is not None and self.color_array_name in self.current_cloud.array_names:
            colors = self.current_cloud[self.color_array_name][valid_indices].copy()
            save_cloud[self.color_array_name] = colors

        try:
            save_cloud.save(path)
            print(f"点云已保存到: {path}")
        except Exception as e:
            print("保存失败:", e)

    
    def update_radius(self, value):
        self.radus = value
        print("Current radius:", self.radus)

    def change_point_color(self):

        if self.current_cloud is None:
            return

        color = QColorDialog.getColor()
        if not color.isValid():
            return

        r = color.red() / 255.0
        g = color.green() / 255.0
        b = color.blue() / 255.0
        self.color = (r, g, b)
        n_points = len(self.current_points)

        # 如果没有颜色数组，创建
        if self.color_array_name is None:
            self.color_array_name = "rgb"
            self.current_cloud[self.color_array_name] = np.zeros((n_points, 4))

        indices = np.where(self.valid_mask)[0]

        # 整体赋值
        self.current_cloud[self.color_array_name][indices] = [r, g, b, 1.0]

        self.points_actor.GetMapper().ScalarVisibilityOn()
        self.plotter.render()

    ### 点模式
    def extract_wire_point_mode(self):
        visited = set(self.seed_id)
        queued = set(self.seed_id)

        # 队列里存 (点id, 父方向)
        stack = deque((pid, None) for pid in self.seed_id)

        start_time = time.perf_counter()

        # 参数：方向连续性阈值
        direction_cos_thresh = 0.8   # 越接近1越严格，可调

        while stack:
            end_time = time.perf_counter()
            if end_time - start_time > 10:
                print("10 second\n")
                break

            pid, parent_dir = stack.popleft()
            p = self.current_points[pid]

            # 1. 当前点邻域
            neighbors = self.kdtree.query_ball_point(p, r=self.radus)
            neighbors = [i for i in neighbors if self.valid_mask[i]]

            if len(neighbors) < 5:
                continue

            pts = self.current_points[neighbors]
            center = pts.mean(axis=0)

            # 2. PCA
            cov = np.cov((pts - center).T)
            eigvals, eigvecs = np.linalg.eigh(cov)

            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            if eigvals[0] < 1e-12:
                continue

            direction = eigvecs[:, 0]
            linearity = (eigvals[0] - eigvals[1]) / eigvals[0]

            print(f"线性度：{linearity}")

            # 当前局部不像导线，就不扩
            if linearity < 0.75:
                continue

            # 3. 新增：和父节点方向做连续性约束
            if parent_dir is not None:
                cos_theta = abs(np.dot(direction, parent_dir))
                print(f"方向一致性：{cos_theta}")

                if cos_theta < direction_cos_thresh:
                    continue

            # 4. 用主轴做几何筛选，而不是逐点PCA
            vecs = pts - center
            ts = vecs @ direction
            perp = vecs - np.outer(ts, direction)
            d_perp = np.linalg.norm(perp, axis=1)

            # 圆柱约束：离主轴不能太远
            mask = d_perp < (0.5 * self.radus)

            accepted_ids = np.array(neighbors)[mask]
            accepted_ts = ts[mask]

            if len(accepted_ids) == 0:
                continue

            # 5. 批量加入 visited
            new_ids = [i for i in accepted_ids if i not in visited]
            visited.update(new_ids)

            # 6. 只选前沿点入队
            frontier = set()

            if len(accepted_ids) == 1:
                frontier.add(int(accepted_ids[0]))
            else:
                idx_max = np.argmax(accepted_ts)
                idx_min = np.argmin(accepted_ts)

                frontier.add(int(accepted_ids[idx_max]))
                frontier.add(int(accepted_ids[idx_min]))

            # 7. 子节点继承当前方向
            for fid in frontier:
                if fid not in queued:
                    stack.append((fid, direction.copy()))
                    queued.add(fid)

        wire_ids = list(visited)
        return wire_ids
    ### 曲线提取模式
    def extract_wire_curve_mode(self):
        min_seed_num = 2

        if len(self.seed_id) < min_seed_num:
            QMessageBox.warning(
                self,
                "提示",
                f"曲线模式至少需要 {min_seed_num} 个种子点"
            )
            return None

        start_time = time.perf_counter()

        # =========================
        # 参数设置
        # =========================

        linearity_thresh = 0.75                      # 局部线性度阈值
        dist_thresh =  self.radus /2              # 点到拟合曲线对应点的距离阈值
        direction_cos_thresh = 0.85
        refit_batch_size = 50                        # 每新增多少点重拟合一次
        frontier_k = 5                               # 每端选几个前沿点

        # =========================
        # 1. 取出种子点
        # =========================
        seed_ids = list(self.seed_id)
        seed_points = self.current_points[seed_ids]

        # =========================
        # 2. 初始拟合
        #    方案A：t 就是 PCA 投影坐标
        # =========================
        pts = seed_points
        centroid_fit = pts.mean(axis=0)

        cov = np.cov((pts - centroid_fit).T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        order_idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order_idx]
        eigvecs = eigvecs[:, order_idx]

        if eigvals[0] < 1e-12:
            print("种子点分布异常，无法拟合曲线")
            return list(self.seed_id)

        direction_fit = eigvecs[:, 0]

        # t = PCA投影坐标
        t = (pts - centroid_fit) @ direction_fit

        if np.max(t) - np.min(t) < 1e-8:
            print("种子点投影范围过小，无法拟合曲线")
            return list(self.seed_id)

        n = len(pts)
        order = min(3, max(1, n // 5))

        A = np.vstack([t**i for i in range(order + 1)]).T

        ax = np.linalg.lstsq(A, pts[:, 0], rcond=None)[0]
        ay = np.linalg.lstsq(A, pts[:, 1], rcond=None)[0]
        az = np.linalg.lstsq(A, pts[:, 2], rcond=None)[0]

        # =========================
        # 3. 初始化
        # =========================
        visited = set(seed_ids)
        queue = deque(seed_ids)

        new_added_since_refit = 0
        refit_count = 0

        # =========================
        # 4. 区域生长
        # =========================
        while queue:
            if time.perf_counter() - start_time > 10:
                print("曲线提取超时 10 秒，提前结束")
                break

            pid = queue.popleft()
            p = self.current_points[pid]


            # ==========================================
            # 4.2 判断当前中心点局部区域是不是像一条线
            # ==========================================
            local_neighbors = self.kdtree.query_ball_point(p, r=self.radus)
            local_neighbors = [i for i in local_neighbors if self.valid_mask[i]]

            if len(local_neighbors) < 5:
                continue

            local_pts = self.current_points[local_neighbors]
            local_center = local_pts.mean(axis=0)

            local_cov = np.cov((local_pts - local_center).T)
            local_eigvals, local_eigvecs = np.linalg.eigh(local_cov)

            local_order_idx = np.argsort(local_eigvals)[::-1]
            local_eigvals = local_eigvals[local_order_idx]
            local_eigvecs = local_eigvecs[:, local_order_idx]

            if local_eigvals[0] < 1e-12:
                continue

            linearity = (local_eigvals[0] - local_eigvals[1]) / local_eigvals[0]

            if linearity < linearity_thresh:
                continue

            local_dir = local_eigvecs[:, 0]


            # ==========================================
            # 4.4 用牛顿迭代法计算点到曲线的近似最短距离
            # ==========================================
            dist_to_curve, q, t_p = point_to_curve_distance_newton(
                p,
                direction_fit,
                centroid_fit,
                ax, ay, az
            )
            print("点到曲线的距离：", dist_to_curve)
            print("\n")
            if dist_to_curve > dist_thresh:
                continue

              # ==========================================
            # 当前 t_p 处的曲线切向
            # ==========================================
            curve_tangent = np.zeros(3)
            for i in range(1, order + 1):
                curve_tangent[0] += i * ax[i] * t_p**(i - 1)
                curve_tangent[1] += i * ay[i] * t_p**(i - 1)
                curve_tangent[2] += i * az[i] * t_p**(i - 1)

            tangent_norm = np.linalg.norm(curve_tangent)
            if tangent_norm < 1e-12:
                continue

            curve_tangent = curve_tangent / tangent_norm

            # ==========================================
            # 局部方向和曲线切向保持一致，避免走分支
            # ==========================================
            cos_theta = abs(np.dot(local_dir, curve_tangent))
            if cos_theta < direction_cos_thresh:
                continue


            # ==========================================
            # 4.6 当前中心点满足条件，则这一球邻域默认都接收
            # ==========================================
            accepted_ids = [i for i in local_neighbors if i not in visited]

            if len(accepted_ids) == 0:
                continue

            visited.update(accepted_ids)
            new_added_since_refit += len(accepted_ids)

            # ==========================================
            # 4.7 从当前邻域里选前沿点
            #     选距离中心较远，并且在局部PCA方向两端的点
            # ==========================================
            accepted_ids = np.array(accepted_ids)
            accepted_pts = self.current_points[accepted_ids]

            vecs = accepted_pts - p
            proj = vecs @ local_dir

            frontier = set()

            if len(accepted_ids) <= 2:
                for idx in accepted_ids:
                    frontier.add(int(idx))
            else:
                sort_idx = np.argsort(proj)
                k = min(frontier_k, len(sort_idx))

                for idx in sort_idx[:k]:
                    frontier.add(int(accepted_ids[idx]))

                for idx in sort_idx[-k:]:
                    frontier.add(int(accepted_ids[idx]))

            for fid in frontier:
                queue.append(fid)

            # ==========================================
            # 4.8 生长一定数量后重新拟合
            #     更新 centroid_fit, direction_fit, 曲线参数方程
            # ==========================================
            if new_added_since_refit >= refit_batch_size:
                pts = self.current_points[list(visited)]

                if len(pts) >= 3:
                    centroid_fit = pts.mean(axis=0)

                    cov = np.cov((pts - centroid_fit).T)
                    eigvals, eigvecs = np.linalg.eigh(cov)

                    order_idx = np.argsort(eigvals)[::-1]
                    eigvals = eigvals[order_idx]
                    eigvecs = eigvecs[:, order_idx]

                    if eigvals[0] > 1e-12:
                        direction_fit = eigvecs[:, 0]

                        # 方案A：重新用新的 PCA 投影坐标作为参数 t
                        t = (pts - centroid_fit) @ direction_fit

                        if np.max(t) - np.min(t) > 1e-8:
                            n = len(pts)
                            order = min(3, max(1, n // 20))

                            A = np.vstack([t**i for i in range(order + 1)]).T

                            ax = np.linalg.lstsq(A, pts[:, 0], rcond=None)[0]
                            ay = np.linalg.lstsq(A, pts[:, 1], rcond=None)[0]
                            az = np.linalg.lstsq(A, pts[:, 2], rcond=None)[0]

                            refit_count += 1
                            print(f"完成第 {refit_count} 次重拟合，当前点数: {len(visited)}")

                new_added_since_refit = 0

        wire_ids = list(visited)
        return wire_ids
    
    def extract_wire(self):
        if len(self.seed_id) == 0:
            return

        if self.extract_mode == "point":
            wire_ids = self.extract_wire_point_mode()
        elif self.extract_mode == "curve":
            wire_ids = self.extract_wire_curve_mode()
        else:
            return
        if wire_ids is None:
            return 
        seed_indices= list(self.seed_id)
        # 清空种子点
        self.seed_id.clear()
        self.info_label.setText(
           f"Seed IDs:\n"
        )
        if  len(wire_ids) == len(seed_indices):
            print("没有提取成功\n")
            # 如果没有提取出点的话，把种子点变成正常的颜色
            self.current_cloud[self.color_array_name][seed_indices,0:3] = self.color[0:3]
            # 只更新 mapper，不要重新 add_mesh
            self.points_actor.GetMapper().ScalarVisibilityOn()
            self.points_actor.GetMapper().SetScalarModeToUsePointData()
            self.points_actor.GetProperty().SetOpacity(1.0)
            self.plotter.render()
            return 
        # 将导线标红
        self.highlight_points(wire_ids)
        # 逻辑删除
        self.valid_mask[wire_ids] = False
        # 记录这次删除
        self.undo_stack.append(wire_ids)


    def delete_wire(self):
        indices = np.where(~self.valid_mask)[0] 
        # 写回同一个数组名
        self.current_cloud[self.color_array_name][indices,3] = 0.0  # 设置为完全透明
        # 只更新 mapper，不要重新 add_mesh
        self.points_actor.GetMapper().ScalarVisibilityOn()
        self.points_actor.GetMapper().SetScalarModeToUsePointData()
        self.points_actor.GetProperty().SetOpacity(1.0)
        self.plotter.render()




    # 撤销删除
    def undo_last(self):

        if not self.undo_stack:
            return

        last_indices = self.undo_stack.pop()

        # 恢复
        self.valid_mask[last_indices] = True

        # 恢复颜色
    
        self.current_cloud[self.color_array_name][last_indices,:] = [self.color[0],self.color[1],self.color[2],1]


        self.points_actor.GetMapper().ScalarVisibilityOn()
        self.points_actor.GetMapper().SetScalarModeToUsePointData()
        self.plotter.render()
        

  
    def highlight_points(self, indices):

        if self.current_cloud is None:
            return

        n_points = len(self.current_points)

        # ===== 情况1：原本有颜色 =====
        if self.color_array_name is not None:
            colors = self.current_cloud[self.color_array_name].copy()

        # ===== 情况2：原本没颜色 =====
        else:
            # 深灰色，不要白色
            colors = np.full((n_points, 3), 0.3)
            self.color_array_name = "rgb"
            self.current_cloud[self.color_array_name] = colors

        # ===== 设置导线为红色 =====
        colors[indices] = [1, 0, 0, 1]   # 红色，完全不透明

        # 写回同一个数组名
        self.current_cloud[self.color_array_name] = colors

        # 只更新 mapper，不要重新 add_mesh
        self.points_actor.GetMapper().ScalarVisibilityOn()
        self.points_actor.GetMapper().SetScalarModeToUsePointData()

        self.plotter.render()

    ### 加载点云
    def load_point_cloud(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open", "", "PointCloud (*.ply *.pcd *.xyz *.las)"
        )

        if not path:
            return

        if path.lower().endswith(".las"):
            las = laspy.read(path)

            points = np.vstack((las.x, las.y, las.z)).T

            cloud = pv.PolyData(points)

            # 如果有颜色
            if hasattr(las, "red"):
                rgb = np.vstack((las.red, las.green, las.blue)).T
                rgb = rgb / 65535.0  # LAS 通常是 16bit
                rgba = np.ones((rgb.shape[0], 4))
                rgba[:, :3] = rgb
                cloud["rgb"] = rgba     

        else:
            cloud = pv.read(path)

        # ======================
        # 显示
        # ======================
        self.plotter.clear()

        self.points_actor = self.plotter.add_mesh(
            cloud,
            scalars="rgb" if "rgb" in cloud.array_names else None,
            rgb=True if "rgb" in cloud.array_names else False,
            opacity=1.0,
            render_points_as_spheres=True,
            point_size=5
        )
        if "rgb" in cloud.array_names:
            self.color_array_name = "rgb"
        else:
            self.color_array_name = None

        self.plotter.reset_camera()
        self.plotter.camera.clipping_range = (0.0001, 1e8)

     
     

        self.plotter.enable_point_picking(
            callback=self.point_picked,
            show_point=False,
            left_clicking=False,   # 关键：关闭左键拾取
            use_picker=True
        )



        self.current_cloud = cloud
        self.current_points = cloud.points.copy()
        self.kdtree = cKDTree(self.current_points)
        self.valid_mask = np.ones(len(self.current_points), dtype=bool)
        self.current_cloud["opacity"] = np.ones(len(self.current_points))
        # 初始化当前颜色
        if "rgb" in cloud.array_names:
            rgb = cloud["rgb"]
            self.color = tuple(rgb[0])   # 取第一个点的颜色作为当前颜色
        else:
            self.color = (0.3, 0.3, 0.3)  # 默认灰色
    



    def point_picked(self, point, picker):

        point_id = picker.GetPointId()

        if point_id < 0:
            return

        if not self.valid_mask[point_id]:
            print("点击了无效点，忽略")
            return

        if point_id not in self.seed_id:
            self.seed_id.add(point_id)
            self.current_cloud[self.color_array_name][point_id] = [1, 0, 0, 1]
        else:
            self.seed_id.discard(point_id)
            self.current_cloud[self.color_array_name][point_id] = [self.color[0],self.color[1],self.color[2],1]

        self.points_actor.GetMapper().ScalarVisibilityOn()
        self.points_actor.GetMapper().SetScalarModeToUsePointData()
        self.plotter.render()

        # 所有种子点ID
        seed_list = sorted(self.seed_id)
        seed_text = "\n".join(map(str, seed_list))

        self.info_label.setText(
           f"Seed IDs:\n{seed_text}"
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())