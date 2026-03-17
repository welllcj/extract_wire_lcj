import sys
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
import laspy
from scipy.spatial import cKDTree
import time


from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QPushButton, QFileDialog,
    QLabel, QHBoxLayout, QColorDialog,
    QDoubleSpinBox
)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt + PyVista PointCloud Platform")
        self.resize(1400, 900)
        self.seed_id = set()
        central = QWidget()
        self.setCentralWidget(central)
        self.radus = 0.3
        layout = QHBoxLayout(central)

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

        layout.addLayout(left_panel, 1)
        layout.addWidget(self.plotter, 4)


        self.points_actor = None
        self.undo_stack = []
        self.color = None

    def save_point_cloud(self):
        pass

    
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

    ### 提取线缆
    def extract_wire(self):
        num_seed = len(self.seed_id)
        if num_seed == 0:
            return

        max_points = 10000

        seed_point = self.current_points[list(self.seed_id)]

        if num_seed <= 3:
            visited = set(self.seed_id)
            stack = list(self.seed_id)
            start_time = time.perf_counter()

            while stack and len(visited) < max_points:
                end_time = time.perf_counter()
                if end_time - start_time > 10:
                    print("10 second\n")
                    break
                pid = stack.pop(0)
                p = self.current_points[pid]

                # k = 30
                # _, neighbors = self.kdtree.query(p, k=k)
                neighbors = self.kdtree.query_ball_point(p, r=self.radus)
                for nid in neighbors:

                    if nid in visited:
                        continue

                    if self.valid_mask[nid] == False:
                        continue

                    point = self.current_points[nid]

                    flag = False
                    p1 = point.copy()
                    p2 = point.copy()
                    p1[2] += 1.3 * self.radus
                    p2[2] -= 1.3 * self.radus
                    idx1 = self.kdtree.query_ball_point(p1, r=0.3)
                    idx2 = self.kdtree.query_ball_point(p2, r=0.3) 
                    if(len(idx1)+len(idx2) <= 10):
                        flag = True

                    if flag == True:
                        visited.add(nid)
                        stack.append(nid)
            wire_ids = list(visited)
                        
        else:
            
            # ===== 1 PCA排序 =====
            pts = seed_point
            centroid = pts.mean(axis=0)

            cov = np.cov((pts - centroid).T)
            eigvals, eigvecs = np.linalg.eig(cov)
            direction = eigvecs[:, np.argmax(eigvals)]

            # 参数 t
            t = (pts - centroid) @ direction

            # ===== 根据点数量确定曲线阶数 =====
            n = len(pts)
            order = min(4, max(1, n // 3))

            # ===== 最小二乘曲线拟合 =====
            A = np.vstack([t**i for i in range(order+1)]).T

            ax = np.linalg.lstsq(A, pts[:,0], rcond=None)[0]
            ay = np.linalg.lstsq(A, pts[:,1], rcond=None)[0]
            az = np.linalg.lstsq(A, pts[:,2], rcond=None)[0]

            # ===== 采样曲线 =====
            length = t.max() - t.min()
            ts = np.linspace(t.min()-length, t.max()+length, 200)

            curve = np.zeros((len(ts),3))

            for i in range(order+1):
                curve[:,0] += ax[i] * ts**i
                curve[:,1] += ay[i] * ts**i
                curve[:,2] += az[i] * ts**i

            # ===== 4 区域生长 =====
            visited = set(self.seed_id)
            stack = list(self.seed_id)

            # 记录哪些seed已经扩展过
            expanded_seed = set()

            # 是否已经进行第二次拟合
            first_refit = True
            start_time = time.perf_counter()

            while stack and len(visited) < max_points:
                end_time = time.perf_counter()
                if end_time - start_time > 10:
                    print("10 second\n")
                    break
                pid = stack.pop(0)
                p = self.current_points[pid]

                # k = 30
                # _, neighbors = self.kdtree.query(p, k=k)
                neighbors = self.kdtree.query_ball_point(p, r=self.radus)
                for nid in neighbors:

                    if nid in visited:
                        continue

                    if self.valid_mask[nid] == False:
                        continue

                    point = self.current_points[nid]

                    # 点到曲线最小距离
                    d = np.linalg.norm(curve - point, axis=1).min()
                    flag = False
                    p1 = point.copy()
                    p2 = point.copy()
                    p1[2] += 1.414 * self.radus
                    p2[2] -= 1.414 * self.radus
                    idx1 = self.kdtree.query_ball_point(p1, r=0.15)
                    idx2 = self.kdtree.query_ball_point(p2, r=0.15) 
                    if(len(idx1)+len(idx2) <= 5):
                        flag = True

                    if d < self.radus and flag == True:
                        visited.add(nid)
                        stack.append(nid)

                # ===== 记录seed是否扩展过 =====
                if pid in self.seed_id:
                    expanded_seed.add(pid)

                # ===== 所有seed扩展一遍后进行第二次拟合 =====
                if first_refit and len(expanded_seed) == num_seed:

                    pts = self.current_points[list(visited)]

                    centroid = pts.mean(axis=0)

                    cov = np.cov((pts - centroid).T)
                    eigvals, eigvecs = np.linalg.eig(cov)
                    direction = eigvecs[:, np.argmax(eigvals)]

                    t = (pts - centroid) @ direction

                    # 根据点数量重新确定阶数
                    n = len(pts)
                    order = min(4, max(1, n // 3))

                    A = np.vstack([t**i for i in range(order+1)]).T

                    ax = np.linalg.lstsq(A, pts[:,0], rcond=None)[0]
                    ay = np.linalg.lstsq(A, pts[:,1], rcond=None)[0]
                    az = np.linalg.lstsq(A, pts[:,2], rcond=None)[0]

                    length = t.max() - t.min()
                    ts = np.linspace(t.min()-length, t.max()+length, 200)

                    curve = np.zeros((len(ts),3))

                    for i in range(order+1):
                        curve[:,0] += ax[i] * ts**i
                        curve[:,1] += ay[i] * ts**i
                        curve[:,2] += az[i] * ts**i

                    first_refit = False

            wire_ids = list(visited)

    
        print("Wire points:", len(visited))

        # 将导线标红
        self.highlight_points(wire_ids)
        # 逻辑删除
        self.valid_mask[wire_ids] = False
        # 记录这次删除
        self.undo_stack.append(wire_ids)
        # 清空种子点
        self.seed_id.clear()
        self.info_label.setText(
           f"Seed IDs:\n"
        )

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