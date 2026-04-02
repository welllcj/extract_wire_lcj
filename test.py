from configparser import NoOptionError
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
    QDoubleSpinBox, QToolButton, QFrame, QMessageBox,QCheckBox
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

        self.adaptive_radius_checkbox = QCheckBox("是否自适应半径")
        self.adaptive_radius_checkbox.setChecked(False)   # 默认不勾选



        self.delete_btn.clicked.connect(self.delete_wire)
        self.extract_btn.clicked.connect(self.extract_wire)
        self.color_btn.clicked.connect(self.change_point_color)
        self.undo_btn.clicked.connect(self.undo_last)
        self.save_btn.clicked.connect(self.save_point_cloud)
        self.load_btn.clicked.connect(self.load_point_cloud)
        self.adaptive_radius_checkbox.stateChanged.connect(self.on_adaptive_radius_changed)

        left_panel.addWidget(self.delete_btn)
        left_panel.addWidget(self.extract_btn)
        left_panel.addWidget(self.load_btn)
        left_panel.addWidget(self.color_btn)
        left_panel.addWidget(self.undo_btn)
        left_panel.addWidget(self.save_btn)
        left_panel.addWidget(QLabel("Wire Radius"))
        left_panel.addWidget(self.radius_box)
        left_panel.addWidget(self.adaptive_radius_checkbox)
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


        self.current_cloud = None
        self.current_points = None
        self.kdtree = None
        self.valid_mask = None
        self.color_array_name = None
        self.use_adaptive_radius = False  ###是否是自适应半径模式

    def on_adaptive_radius_changed(self, state):
        self.use_adaptive_radius = self.adaptive_radius_checkbox.isChecked()
        print("是否自适应半径：", self.use_adaptive_radius)

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

    ### 估计导线半径
    def estimate_structure_radius(self):
        “””
        估计导线半径（优化版：更高效的半径枚举 + 中空导管支持）
        返回: radius
        “””

        seed_ids = list(self.seed_id)
        radius_list = []

        for pid in seed_ids:
            p = self.current_points[pid]

            # 1) 优化的半径扫描：先粗扫描找范围，再细扫描
            direction = None
            linearity_best = -np.inf
            best_rr = None
            selected_rr = None

            # 线性度阈值
            linearity_threshold = 0.82  # 稍微降低以适应更多场景

            # 第一阶段：粗扫描（步长0.1）
            coarse_r_list = np.arange(0.1, 3.0, 0.1)
            coarse_best_rr = None
            coarse_best_linearity = -np.inf

            for rr in coarse_r_list:
                sphere_idxs = self.kdtree.query_ball_point(p, r=rr)
                sphere_idxs = [i for i in sphere_idxs if self.valid_mask[i]]

                if len(sphere_idxs) < 10:
                    continue

                pts = self.current_points[sphere_idxs]
                center = pts.mean(axis=0)
                cov = np.cov((pts - center).T)
                eigvals, eigvecs = np.linalg.eigh(cov)

                order = np.argsort(eigvals)[::-1]
                eigvals = eigvals[order]
                eigvecs = eigvecs[:, order]

                if eigvals[0] < 1e-12:
                    continue

                denom = max(eigvals[0], 1e-12)
                linearity = (eigvals[0] - eigvals[1]) / denom

                if linearity > coarse_best_linearity:
                    coarse_best_linearity = linearity
                    coarse_best_rr = rr
                    direction = eigvecs[:, 0]

                # 如果线性度足够高，提前停止
                if linearity > linearity_threshold:
                    selected_rr = float(rr)
                    linearity_best = linearity
                    best_rr = rr
                    print(f”[radius] seed={pid} 粗扫描命中: rr={rr:.2f}, linearity={linearity:.6f}”)
                    break

            # 第二阶段：如果没有命中阈值，在最佳范围附近细扫描
            if selected_rr is None and coarse_best_rr is not None:
                fine_start = max(0.05, coarse_best_rr - 0.15)
                fine_end = min(3.0, coarse_best_rr + 0.15)
                fine_r_list = np.arange(fine_start, fine_end, 0.02)

                for rr in fine_r_list:
                    sphere_idxs = self.kdtree.query_ball_point(p, r=rr)
                    sphere_idxs = [i for i in sphere_idxs if self.valid_mask[i]]

                    if len(sphere_idxs) < 10:
                        continue

                    pts = self.current_points[sphere_idxs]
                    center = pts.mean(axis=0)
                    cov = np.cov((pts - center).T)
                    eigvals, eigvecs = np.linalg.eigh(cov)

                    order = np.argsort(eigvals)[::-1]
                    eigvals = eigvals[order]
                    eigvecs = eigvecs[:, order]

                    if eigvals[0] < 1e-12:
                        continue

                    denom = max(eigvals[0], 1e-12)
                    linearity = (eigvals[0] - eigvals[1]) / denom

                    if linearity > linearity_threshold:
                        selected_rr = float(rr)
                        linearity_best = linearity
                        direction = eigvecs[:, 0]
                        best_rr = rr
                        print(f"[radius] seed={pid} 细扫描命中: rr={rr:.2f}, linearity={linearity:.6f}")
                        break

                    if linearity > linearity_best:
                        linearity_best = linearity
                        direction = eigvecs[:, 0]
                        best_rr = rr

            # 若扫描失败，则跳过该种子点
            if direction is None:
                continue

            print(f"[radius] seed={pid} 最佳线性度={linearity_best:.6f} best_rr={best_rr:.2f}")

            # 如果命中线性度阈值，直接返回该半径
            if selected_rr is not None:
                radius_list.append(selected_rr)
                print(f"[radius] seed={pid} 直接返回半径: {selected_rr:.4f}")
                continue

            # 2) 使用较大球查询获取邻居点（优化：使用best_rr的3倍而非固定5m）
            search_radius = min(5.0, max(1.0, best_rr * 3.0))
            sphere_idxs = self.kdtree.query_ball_point(p, r=search_radius)
            sphere_idxs = [i for i in sphere_idxs if self.valid_mask[i]]

            if len(sphere_idxs) < 30:  # 降低要求
                continue

            sphere_pts = self.current_points[sphere_idxs]

            # 3) 截面平面：过种子点p，法向量为主方向direction
            #    距离该平面的距离越小，越接近“截面薄片”
            sphere_vecs = sphere_pts - p
            plane_dist = np.abs(sphere_vecs @ direction)

            order_idx = np.argsort(plane_dist)
            if len(order_idx) < 10:
                print(f"[radius] seed={pid} 有效截面候选点不足10，跳过")
                continue

            # 4) 构造截面平面两个正交基 e1, e2
            tmp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(tmp, direction)) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0])

            e1 = np.cross(direction, tmp)
            e1_norm = np.linalg.norm(e1)
            if e1_norm < 1e-12:
                continue
            e1 = e1 / e1_norm

            e2 = np.cross(direction, e1)
            e2_norm = np.linalg.norm(e2)
            if e2_norm < 1e-12:
                continue
            e2 = e2 / e2_norm

            # 5) 优化n_fit扫描范围：基于点数动态调整
            candidate_radius = []
            max_n_fit = min(80, len(order_idx) // 2)  # 不超过可用点数的一半
            n_fit_list = list(range(10, max_n_fit + 1, 5))
            print(f"[radius] seed={pid} 开始扫描 n_fit: {n_fit_list[:5]}...{n_fit_list[-3:] if len(n_fit_list) > 5 else ''}")

            for n_fit in n_fit_list:
                if len(order_idx) < n_fit:
                    continue

                fit_idx = order_idx[:n_fit]
                slice_pts = sphere_pts[fit_idx]
                if len(slice_pts) < 10:
                    continue

                slice_vecs = slice_pts - p
                u = slice_vecs @ e1
                v = slice_vecs @ e2
                pts2d = np.column_stack([u, v])
                n2 = len(pts2d)

                if n2 < 10:
                    continue

                # ---- DBSCAN（纯numpy实现，优化参数）----
                nearest_dists = np.empty(n2, dtype=float)
                for i in range(n2):
                    diff = pts2d - pts2d[i]
                    d2 = np.sum(diff * diff, axis=1)
                    d2[i] = np.inf
                    nearest_dists[i] = np.sqrt(np.min(d2))

                eps = float(np.percentile(nearest_dists, 25) * 2.0)  # 调整参数
                eps = max(0.03, eps)  # 降低最小值
                eps = min(eps, max(0.25, 2.5 * self.radus))  # 增加上限
                eps2 = eps * eps

                min_samples = max(5, int(0.10 * n2))  # 降低最小样本数
                min_samples = min(min_samples, 15)  # 增加上限

                labels = -np.ones(n2, dtype=int)
                visited = np.zeros(n2, dtype=bool)
                cluster_id = 0

                def region_query(idx):
                    diff = pts2d - pts2d[idx]
                    d2 = np.sum(diff * diff, axis=1)
                    return np.where(d2 <= eps2)[0]

                for i in range(n2):
                    if visited[i]:
                        continue
                    visited[i] = True

                    neighbors = region_query(i)
                    if len(neighbors) < min_samples:
                        labels[i] = -1
                        continue

                    labels[i] = cluster_id
                    seed_queue = list(neighbors.tolist())
                    q_pos = 0

                    while q_pos < len(seed_queue):
                        j = seed_queue[q_pos]
                        q_pos += 1

                        if not visited[j]:
                            visited[j] = True
                            neighbors_j = region_query(j)
                            if len(neighbors_j) >= min_samples:
                                for jj in neighbors_j.tolist():
                                    if labels[jj] == -1:
                                        labels[jj] = cluster_id
                                    if jj not in seed_queue:
                                        seed_queue.append(jj)

                        if labels[j] == -1:
                            labels[j] = cluster_id

                    cluster_id += 1

                best_label = -1
                best_count = 0
                for cid in range(cluster_id):
                    cnt = int(np.sum(labels == cid))
                    if cnt > best_count:
                        best_count = cnt
                        best_label = cid

                if best_label < 0 or best_count < 8:  # 降低要求
                    pts_cluster = pts2d
                else:
                    pts_cluster = pts2d[labels == best_label]

                m = len(pts_cluster)
                if m < 3:
                    continue

                # ==========================================
                # 中空导管检测：检查点是否呈环形分布
                # ==========================================
                is_hollow = False
                if m >= 8:
                    # 计算点到质心的距离
                    center_2d = pts_cluster.mean(axis=0)
                    radial_dists = np.linalg.norm(pts_cluster - center_2d, axis=1)
                    radial_mean = np.mean(radial_dists)
                    radial_std = np.std(radial_dists)

                    # 如果径向距离标准差小，说明点分布在圆环上
                    if radial_std < radial_mean * 0.3 and radial_mean > 0.05:
                        is_hollow = True

                        # 对于中空导管，半径估计为径向距离的平均值
                        r_est = radial_mean
                        if r_est > 0 and np.isfinite(r_est):
                            candidate_radius.append((n_fit, r_est))
                            if n_fit % 20 == 10 or len(candidate_radius) <= 3:
                                print(f"[radius] seed={pid}, n_fit={n_fit}, r_est={r_est:.4f} (中空导管)")
                        continue

                # ==========================================
                # 实心导线：使用最小外接矩形
                # ==========================================

                best_area = np.inf
                best_diag = np.inf
                for i in range(m):
                    for j in range(i + 1, m):
                        vec = pts_cluster[j] - pts_cluster[i]
                        norm = np.linalg.norm(vec)
                        if norm < 1e-12:
                            continue

                        ux = vec / norm
                        uy = np.array([-ux[1], ux[0]], dtype=float)
                        xproj = pts_cluster @ ux
                        yproj = pts_cluster @ uy

                        dx = float(np.max(xproj) - np.min(xproj))
                        dy = float(np.max(yproj) - np.min(yproj))
                        area = dx * dy
                        diag = float(np.hypot(dx, dy))
                        if area < best_area:
                            best_area = area
                            best_diag = diag

                if not np.isfinite(best_diag):
                    continue

                r_est = 0.5 * best_diag
                if r_est <= 0 or not np.isfinite(r_est):
                    continue

                candidate_radius.append((n_fit, r_est))
                # 只在关键点打印，减少日志
                if n_fit % 20 == 10 or len(candidate_radius) <= 3:
                    print(f"[radius] seed={pid}, n_fit={n_fit}, r_est={r_est:.4f}")

            if len(candidate_radius) == 0:
                print(f"[radius] seed={pid} 没有可用候选半径，跳过")
                continue

            # 6) 在所有n_fit候选中，选”最合适”的半径：
            #    采用稳健准则：离候选半径中位数最近
            cand_vals = np.array([x[1] for x in candidate_radius], dtype=float)
            cand_med = float(np.median(cand_vals))
            best_idx = int(np.argmin(np.abs(cand_vals - cand_med)))
            best_n_fit, best_r = candidate_radius[best_idx]

            print(f”[radius] seed={pid} 最终: n_fit={best_n_fit}, r={best_r:.4f}, 候选数={len(candidate_radius)}”)
            radius_list.append(best_r)

        if len(radius_list) == 0:
            return self.radus

        radius = np.median(radius_list)
        print(f”[radius] 所有种子点估计完成，最终半径: {radius:.4f}”)

        return radius


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
            if linearity < 0.7:
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
    
    def process_representative_points_and_neighbors(
        self,
        rep_local_idx,
        local_pts,
        local_neighbors,
        visited,
        local_dir,
        direction_fit,
        centroid_fit,
        ax, ay, az,
        order,
        dist_thresh,
        direction_cos_thresh
    ):
        """
        处理代表点与邻居点（优化版：支持中空导管）
        1. 先检查所有代表点是否都满足条件
        2. 如果全部满足，则当前邻域整体接收
        3. 如果有任意一个不满足，则对所有邻居点逐个检查
        4. 增加中空导管检测：如果代表点分布呈环形，放宽距离要求
        返回:
            accepted_ids: 最终接收的点编号列表
            t_values: 通过检查时对应的曲线参数t列表
            all_rep_pass: 代表点是否全部通过
        """

        rep_t_values = []
        rep_dists = []
        all_rep_pass = True

        # =========================
        # 1) 先检查所有代表点
        # =========================
        for ridx in rep_local_idx:
            rp = local_pts[ridx]

            dist_r, q_r, t_r = point_to_curve_distance_newton(
                rp,
                direction_fit,
                centroid_fit,
                ax, ay, az
            )

            rep_dists.append(dist_r if np.isfinite(dist_r) else 1e6)

            if not np.isfinite(dist_r) or dist_r > dist_thresh:
                all_rep_pass = False
                break

            # 计算 t_r 处曲线切向
            curve_tangent_r = np.zeros(3)
            for i in range(1, order + 1):
                curve_tangent_r[0] += i * ax[i] * (t_r ** (i - 1))
                curve_tangent_r[1] += i * ay[i] * (t_r ** (i - 1))
                curve_tangent_r[2] += i * az[i] * (t_r ** (i - 1))

            tangent_norm_r = np.linalg.norm(curve_tangent_r)
            if tangent_norm_r < 1e-12:
                all_rep_pass = False
                break

            curve_tangent_r = curve_tangent_r / tangent_norm_r
            cos_r = abs(np.dot(local_dir, curve_tangent_r))

            if cos_r < direction_cos_thresh:
                all_rep_pass = False
                break

            rep_t_values.append(t_r)

        # =========================
        # 1.5) 中空导管检测：如果代表点距离较一致且较大，可能是中空导管
        # =========================
        is_hollow_pipe = False
        if len(rep_dists) >= 3:
            rep_dists_arr = np.array(rep_dists)
            valid_dists = rep_dists_arr[rep_dists_arr < 1e5]
            if len(valid_dists) >= 3:
                dist_std = np.std(valid_dists)
                dist_mean = np.mean(valid_dists)
                # 如果距离标准差小，且平均距离接近阈值，判定为中空导管
                if dist_std < dist_thresh * 0.3 and dist_mean > dist_thresh * 0.5:
                    is_hollow_pipe = True

        accepted_ids = []

        # =========================
        # 2) 如果所有代表点都通过，或检测到中空导管，则整团接收
        # =========================
        if all_rep_pass or is_hollow_pipe:
            for nid in local_neighbors:
                if nid not in visited:
                    accepted_ids.append(nid)

            return accepted_ids, rep_t_values, True

        # =========================
        # 3) 如果代表点没有全部通过，则逐点检查所有邻居点
        # =========================
        t_values = []

        # 如果检测到可能是中空导管，放宽距离阈值
        effective_dist_thresh = dist_thresh * 1.3 if is_hollow_pipe else dist_thresh

        for nid in local_neighbors:
            if nid in visited:
                continue

            np_point = self.current_points[nid]

            dist_i, q_i, t_i = point_to_curve_distance_newton(
                np_point,
                direction_fit,
                centroid_fit,
                ax, ay, az
            )

            if not np.isfinite(dist_i) or dist_i > effective_dist_thresh:
                continue

            curve_tangent_i = np.zeros(3)
            for j in range(1, order + 1):
                curve_tangent_i[0] += j * ax[j] * (t_i ** (j - 1))
                curve_tangent_i[1] += j * ay[j] * (t_i ** (j - 1))
                curve_tangent_i[2] += j * az[j] * (t_i ** (j - 1))

            tangent_norm_i = np.linalg.norm(curve_tangent_i)
            if tangent_norm_i < 1e-12:
                continue

            curve_tangent_i = curve_tangent_i / tangent_norm_i
            cos_i = abs(np.dot(local_dir, curve_tangent_i))

            if cos_i < direction_cos_thresh:
                continue

            accepted_ids.append(nid)
            t_values.append(t_i)

        return accepted_ids, t_values, False

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

        linearity_thresh = 0.60                   # 局部线性度阈值（降低以适应更多场景）
        direction_cos_thresh = 0.75               # 方向一致性阈值（降低以适应弯曲）
        refit_batch_size = 50                     # 每新增多少点重拟合一次（增加以减少重拟合次数）
        frontier_k = 8                            # 每端选几个前沿点（增加以提高覆盖）
        max_iterations = 15000                    # 最大迭代次数，防止无限循环
        gap_tolerance = 2.5                       # 断裂容忍倍数（相对于半径）

        # =========================
        # 1. 取出种子点并进行质量检查
        # =========================
        seed_ids = list(self.seed_id)
        seed_points = self.current_points[seed_ids]

        # 种子点质量检查：移除离群种子点
        if len(seed_ids) >= 3:
            seed_center = seed_points.mean(axis=0)
            seed_dists = np.linalg.norm(seed_points - seed_center, axis=1)
            seed_median_dist = np.median(seed_dists)
            seed_mad = np.median(np.abs(seed_dists - seed_median_dist))

            # 移除距离中位数超过3倍MAD的离群点
            outlier_threshold = seed_median_dist + 3.0 * seed_mad
            valid_seed_mask = seed_dists <= outlier_threshold
            if np.sum(valid_seed_mask) >= 2:
                seed_ids = [sid for sid, valid in zip(seed_ids, valid_seed_mask) if valid]
                seed_points = self.current_points[seed_ids]
                if np.sum(~valid_seed_mask) > 0:
                    print(f"移除 {np.sum(~valid_seed_mask)} 个离群种子点")

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

        # 检查种子点的线性度，如果过低则提前警告并尝试调整
        seed_linearity = (eigvals[0] - eigvals[1]) / eigvals[0]
        if seed_linearity < 0.4:
            print(f"警告：种子点线性度过低 ({seed_linearity:.3f})，提取可能失败")
            return list(self.seed_id)
        elif seed_linearity < 0.6:
            print(f"警告：种子点线性度较低 ({seed_linearity:.3f})，将降低约束条件")
            # 动态调整阈值
            linearity_thresh = max(0.50, linearity_thresh - 0.1)
            direction_cos_thresh = max(0.65, direction_cos_thresh - 0.1)

        direction_fit = eigvecs[:, 0]

        # t = PCA投影坐标
        t = (pts - centroid_fit) @ direction_fit

        if np.max(t) - np.min(t) < 1e-8:
            print("种子点投影范围过小，无法拟合曲线")
            return list(self.seed_id)

        n = len(pts)
        # 动态调整多项式阶数：点数少用低阶，点数多用高阶
        if n < 5:
            order = 1
        elif n < 15:
            order = 2
        else:
            order = min(4, max(2, n // 15))

        A = np.vstack([t**i for i in range(order + 1)]).T

        ax = np.linalg.lstsq(A, pts[:, 0], rcond=None)[0]
        ay = np.linalg.lstsq(A, pts[:, 1], rcond=None)[0]
        az = np.linalg.lstsq(A, pts[:, 2], rcond=None)[0]

        # =========================
        # 2.5 基于种子点估计结构厚度，用于自适应距离阈值
        # =========================
        seed_radial_dists = []
        for sid in seed_ids:
            sp = self.current_points[sid]
            vecs = pts - centroid_fit
            proj = vecs @ direction_fit
            perp = vecs - np.outer(proj, direction_fit)
            d_perp = np.linalg.norm(perp, axis=1)
            seed_radial_dists.extend(d_perp)

        if len(seed_radial_dists) > 0:
            # 使用中位数绝对偏差(MAD)提高鲁棒性
            median_dist = np.median(seed_radial_dists)
            mad = np.median(np.abs(seed_radial_dists - median_dist))
            structure_thickness = median_dist + 2.0 * mad  # 增加容忍度
            dist_thresh = max(structure_thickness * 1.5, self.radus * 0.4)  # 更宽松的阈值
            print(f"初始距离阈值: {dist_thresh:.4f} (结构厚度: {structure_thickness:.4f})")
        else:
            dist_thresh = self.radus * 0.6  # 更宽松的默认值
            print(f"初始距离阈值: {dist_thresh:.4f} (默认值)")

        # =========================
        # 3. 初始化
        # =========================
        visited = set(seed_ids)
        queue = deque(seed_ids)
        queued = set(seed_ids)

        new_added_since_refit = 0
        refit_count = 0

        # 记录生长统计信息
        total_checked = 0
        total_accepted = 0
        iteration_count = 0

        # 断裂桥接机制：记录最近N次失败的前沿点
        failed_frontier_history = deque(maxlen=20)
        consecutive_failures = 0

        # =========================
        # 4. 区域生长
        # =========================
        while queue and iteration_count < max_iterations:
            iteration_count += 1

            pid = queue.popleft()
            p = self.current_points[pid]


            # ==========================================
            # 4.2 判断当前中心点局部区域是不是像一条线
            # ==========================================
            local_neighbors = self.kdtree.query_ball_point(p, r=self.radus)
            local_neighbors = [i for i in local_neighbors if self.valid_mask[i]]

            # 如果邻域点数过少，尝试扩大搜索半径（断裂容忍）
            if len(local_neighbors) < 5:
                extended_neighbors = self.kdtree.query_ball_point(p, r=self.radus * gap_tolerance)
                extended_neighbors = [i for i in extended_neighbors if self.valid_mask[i]]
                if len(extended_neighbors) >= 5:
                    local_neighbors = extended_neighbors
                    consecutive_failures = 0  # 找到点，重置失败计数
                else:
                    consecutive_failures += 1
                    if consecutive_failures < 5:  # 允许少量连续失败
                        continue
                    else:
                        failed_frontier_history.append(pid)
                        continue
            else:
                consecutive_failures = 0  # 重置失败计数

            local_pts = self.current_points[local_neighbors]
            local_center = local_pts.mean(axis=0)

            # 快速检查：如果邻域点数过少，降低要求
            min_pts_for_pca = 8 if len(local_neighbors) >= 10 else 5
            if len(local_pts) < min_pts_for_pca:
                continue

            local_cov = np.cov((local_pts - local_center).T)
            local_eigvals, local_eigvecs = np.linalg.eigh(local_cov)

            local_order_idx = np.argsort(local_eigvals)[::-1]
            local_eigvals = local_eigvals[local_order_idx]
            local_eigvecs = local_eigvecs[:, local_order_idx]

            if local_eigvals[0] < 1e-12:
                continue

            linearity = (local_eigvals[0] - local_eigvals[1]) / local_eigvals[0]

            # 动态调整线性度阈值：如果连续失败多次，降低要求
            effective_linearity_thresh = linearity_thresh
            if consecutive_failures > 0:
                effective_linearity_thresh = max(0.50, linearity_thresh - 0.05 * consecutive_failures)

            if linearity < effective_linearity_thresh:
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

            # 动态调整距离阈值：如果连续失败，放宽距离要求
            effective_dist_thresh = dist_thresh
            if consecutive_failures > 0:
                effective_dist_thresh = dist_thresh * (1.0 + 0.2 * min(consecutive_failures, 3))

            if not np.isfinite(dist_to_curve) or dist_to_curve > effective_dist_thresh:
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

            # 动态调整方向阈值
            effective_direction_thresh = direction_cos_thresh
            if consecutive_failures > 0:
                effective_direction_thresh = max(0.65, direction_cos_thresh - 0.05 * min(consecutive_failures, 2))

            if cos_theta < effective_direction_thresh:
                continue


            # ==========================================
            # 4.6 引入代表点预检机制
            # ==========================================
            candidate_ids = [i for i in local_neighbors if i not in visited]

            if len(candidate_ids) == 0:
                continue

            # 选择代表点：优先选择距离中心较远且角度分布均匀的点
            candidate_pts = self.current_points[candidate_ids]
            dists_to_center = np.linalg.norm(candidate_pts - p, axis=1)

            # 动态调整代表点数量：邻域越大，代表点越多
            num_representatives = min(max(4, len(candidate_ids) // 8), 10)  # 增加代表点数量

            # 改进代表点选择：结合距离和角度分布
            if len(candidate_ids) <= num_representatives:
                rep_indices = np.arange(len(candidate_ids))
            else:
                # 先选择距离最远的2倍候选
                far_candidates = np.argsort(dists_to_center)[-(num_representatives * 2):]

                # 在远点中选择角度分布均匀的点
                if len(far_candidates) > num_representatives:
                    far_pts = candidate_pts[far_candidates]
                    vecs = far_pts - p
                    # 投影到垂直于局部方向的平面
                    perp_vecs = vecs - np.outer(vecs @ local_dir, local_dir)
                    norms = np.linalg.norm(perp_vecs, axis=1)

                    # 计算角度（避免除零）
                    angles = np.zeros(len(far_candidates))
                    for i, (vec, norm) in enumerate(zip(perp_vecs, norms)):
                        if norm > 1e-9:
                            angles[i] = np.arctan2(vec[1], vec[0])

                    # 按角度排序，均匀采样
                    sorted_by_angle = np.argsort(angles)
                    step = max(1, len(sorted_by_angle) // num_representatives)
                    rep_indices = far_candidates[sorted_by_angle[::step][:num_representatives]]
                else:
                    rep_indices = far_candidates

            rep_local_idx = rep_indices

            # 调用代表点检查函数（使用动态阈值）
            accepted_ids, t_values, all_rep_pass = self.process_representative_points_and_neighbors(
                rep_local_idx,
                candidate_pts,
                candidate_ids,
                visited,
                local_dir,
                direction_fit,
                centroid_fit,
                ax, ay, az,
                order,
                effective_dist_thresh,  # 使用动态阈值
                effective_direction_thresh  # 使用动态阈值
            )

            if len(accepted_ids) == 0:
                continue

            # 更新统计信息
            total_checked += len(candidate_ids)
            total_accepted += len(accepted_ids)

            visited.update(accepted_ids)
            new_added_since_refit += len(accepted_ids)

            # ==========================================
            # 4.7 优化前沿点选择：基于曲线切向，增加优先级队列机制
            # ==========================================
            accepted_ids_arr = np.array(accepted_ids)
            accepted_pts = self.current_points[accepted_ids_arr]

            vecs = accepted_pts - p
            proj = vecs @ curve_tangent

            frontier = set()

            if len(accepted_ids_arr) <= 2:
                for idx in accepted_ids_arr:
                    frontier.add(int(idx))
            else:
                sort_idx = np.argsort(proj)
                k = min(frontier_k, len(sort_idx))

                # 优先选择投影距离较大的点（远离中心）
                for idx in sort_idx[:k]:
                    frontier.add(int(accepted_ids_arr[idx]))

                for idx in sort_idx[-k:]:
                    frontier.add(int(accepted_ids_arr[idx]))

                # 额外添加：如果接收点数较多，增加中间区域的采样
                if len(accepted_ids_arr) > 20:
                    mid_k = min(3, len(sort_idx) // 4)  # 增加中间采样
                    mid_start = len(sort_idx) // 2 - mid_k // 2
                    for i in range(mid_start, mid_start + mid_k):
                        if 0 <= i < len(sort_idx):
                            frontier.add(int(accepted_ids_arr[sort_idx[i]]))

            # 优先级队列：将前沿点添加到队列前端（双端队列的优势）
            for fid in frontier:
                if fid not in queued:
                    # 如果是大量接收，说明生长顺利，优先处理
                    if len(accepted_ids) > 15:
                        queue.appendleft(fid)  # 添加到队列前端
                    else:
                        queue.append(fid)  # 添加到队列后端
                    queued.add(fid)

            # ==========================================
            # 4.8 智能重拟合：只在必要时重拟合
            # ==========================================
            should_refit = False

            # 条件1：累积足够多的新点
            if new_added_since_refit >= refit_batch_size:
                should_refit = True

            # 条件2：连续失败次数较多，可能需要更新曲线
            if consecutive_failures >= 3 and new_added_since_refit >= refit_batch_size // 2:
                should_refit = True

            if should_refit:
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

                        t = (pts - centroid_fit) @ direction_fit

                        if np.max(t) - np.min(t) > 1e-8:
                            n = len(pts)
                            # 重拟合时也使用动态阶数
                            if n < 50:
                                order = 2
                            elif n < 200:
                                order = 3
                            else:
                                order = min(4, max(2, n // 50))

                            A = np.vstack([t**i for i in range(order + 1)]).T

                            ax = np.linalg.lstsq(A, pts[:, 0], rcond=None)[0]
                            ay = np.linalg.lstsq(A, pts[:, 1], rcond=None)[0]
                            az = np.linalg.lstsq(A, pts[:, 2], rcond=None)[0]

                            # 重新估计结构厚度（使用MAD）
                            vecs = pts - centroid_fit
                            proj = vecs @ direction_fit
                            perp = vecs - np.outer(proj, direction_fit)
                            d_perp = np.linalg.norm(perp, axis=1)
                            median_dist = np.median(d_perp)
                            mad = np.median(np.abs(d_perp - median_dist))
                            structure_thickness = median_dist + 2.0 * mad  # 增加容忍度
                            dist_thresh = max(structure_thickness * 1.5, self.radus * 0.4)  # 更宽松

                            refit_count += 1
                            consecutive_failures = 0  # 重拟合后重置失败计数
                            print(f"完成第 {refit_count} 次重拟合，当前点数: {len(visited)}, 距离阈值: {dist_thresh:.4f}")

                new_added_since_refit = 0

        wire_ids = list(visited)

        # 输出最终统计信息
        if total_checked > 0:
            acceptance_rate = total_accepted / total_checked * 100
            print(f"提取完成：迭代 {iteration_count} 次，检查 {total_checked} 个候选点，接收 {total_accepted} 个 ({acceptance_rate:.1f}%)")

        if iteration_count >= max_iterations:
            print(f"警告：达到最大迭代次数 {max_iterations}，提前终止")

        return wire_ids
    
    def extract_wire(self):
        if len(self.seed_id) == 0:
            return


        # =========================
        # 自适应：直接估计导线半径
        # =========================
        if self.use_adaptive_radius:
            new_radius = self.estimate_structure_radius()
            print("估计半径：", new_radius)

            self.radus = new_radius 
            self.radius_box.blockSignals(True)
            self.radius_box.setValue(self.radus)
            self.radius_box.blockSignals(False)

            QMessageBox.information(
                self,
                "半径估计成功",
                f"估计导线半径为：{new_radius:.4f}"
            )


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


        # 写回同一个数组名
        self.current_cloud[self.color_array_name][indices,:] = [1, 0, 0, 1]

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

       
        if self.color_array_name is None:
            self.color_array_name = "rgb"
            n_points = len(self.current_points)
            rgba = np.ones((n_points, 4))
            rgba[:, :3] = 0.3   # 默认灰色
            self.current_cloud[self.color_array_name] = rgba

       

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