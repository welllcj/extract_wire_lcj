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
        """
        估计导线半径（统一使用：5m球查询 + 截面最近点 + 圆拟合）
        返回: radius
        """

        seed_ids = list(self.seed_id)
        radius_list = []

        for pid in seed_ids:
            p = self.current_points[pid]

            # 1) 扫描球半径：用线性度选择主方向，并找出“候选导线半径”
            #    线性度定义： (eig0 - eig1) / eig0
            direction = None
            linearity_best = -np.inf
            best_rr = None
            selected_rr = None

            # 线性度阈值：你的大胆想法
            linearity_threshold = 0.85

            # 从 0.05 开始枚举，步长 0.05
            r_list = np.arange(0.05, 5.0 + 1e-9, 0.05)
            for rr in r_list:
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

                print(
                    f"[radius] seed={pid} rr={rr:.2f} sphere_n={len(sphere_idxs)} linearity={linearity:.6f} "
                    f"current_best={linearity_best:.6f}"
                )

                if linearity > linearity_threshold:
                    selected_rr = float(rr)
                    linearity_best = linearity
                    direction = eigvecs[:, 0]
                    best_rr = rr
                    print(
                        f"[radius] seed={pid} 命中阈值: rr={rr:.2f}, "
                        f"linearity={linearity:.6f} (> {linearity_threshold})，将 rr 视为导线半径并停止扫描"
                    )
                    break

                if linearity > linearity_best:
                    print(
                        f"[radius] seed={pid} 更新方向: rr={rr:.2f}, "
                        f"sphere_n={len(sphere_idxs)}, linearity={linearity:.6f}, "
                        f"prev_linear={linearity_best:.6f}"
                    )
                    linearity_best = linearity
                    direction = eigvecs[:, 0]
                    best_rr = rr

            # 若KNN和扫描都失败，则跳过该种子点
            if direction is None:
                continue

            # 不可视化时，打印方向（从种子点出发便于你核对）
            p_end = p + direction * 1.0
            print(
                f"[radius] seed={pid} direction={direction} p_end={p_end} "
                f"linearity_best={linearity_best:.6f} best_rr={best_rr}"
            )

            # 你的大胆想法：一旦命中线性度阈值，用该半径直接作为导线半径
            if selected_rr is not None:
                radius_list.append(selected_rr)
                print(f"[radius] seed={pid} 直接返回半径: {selected_rr:.6f}")
                continue

            # 2) 5m球查询获取很多邻居点
            sphere_idxs = self.kdtree.query_ball_point(p, r=5.0)
            sphere_idxs = [i for i in sphere_idxs if self.valid_mask[i]]

            if len(sphere_idxs) < 50:
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

            # 5) n_fit从10到80（步长5）全部尝试，得到多个半径候选
            candidate_radius = []
            n_fit_list = list(range(10, 81, 5))
            print(f"[radius] seed={pid} 开始扫描 n_fit: {n_fit_list}")

            for n_fit in n_fit_list:
                if len(order_idx) < n_fit:
                    print(f"[radius] seed={pid}, n_fit={n_fit}, 可用点不足，跳过")
                    continue

                fit_idx = order_idx[:n_fit]
                slice_pts = sphere_pts[fit_idx]
                if len(slice_pts) < 10:
                    print(f"[radius] seed={pid}, n_fit={n_fit}, slice点不足10，跳过")
                    continue

                slice_vecs = slice_pts - p
                u = slice_vecs @ e1
                v = slice_vecs @ e2
                pts2d = np.column_stack([u, v])
                n2 = len(pts2d)

                if n2 < 10:
                    print(f"[radius] seed={pid}, n_fit={n_fit}, 投影点不足10，跳过")
                    continue

                # ---- DBSCAN（纯numpy实现）----
                nearest_dists = np.empty(n2, dtype=float)
                for i in range(n2):
                    diff = pts2d - pts2d[i]
                    d2 = np.sum(diff * diff, axis=1)
                    d2[i] = np.inf
                    nearest_dists[i] = np.sqrt(np.min(d2))

                eps = float(np.percentile(nearest_dists, 20) * 2.5)
                eps = max(0.05, eps)
                eps = min(eps, max(0.2, 2.0 * self.radus))
                eps2 = eps * eps

                min_samples = max(6, int(0.12 * n2))
                min_samples = min(min_samples, 12)

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

                if best_label < 0 or best_count < 10:
                    pts_cluster = pts2d
                else:
                    pts_cluster = pts2d[labels == best_label]

                m = len(pts_cluster)
                if m < 3:
                    print(f"[radius] seed={pid}, n_fit={n_fit}, 聚类后点过少，跳过")
                    continue

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
                    print(f"[radius] seed={pid}, n_fit={n_fit}, best_diag无效，跳过")
                    continue

                r_est = 0.5 * best_diag
                if r_est <= 0 or not np.isfinite(r_est):
                    print(f"[radius] seed={pid}, n_fit={n_fit}, r_est无效，跳过")
                    continue

                candidate_radius.append((n_fit, r_est))
                print(
                    f"[radius] seed={pid}, n_fit={n_fit}, "
                    f"eps={eps:.4f}, cluster_size={m}, r_est={r_est:.6f}"
                )

            if len(candidate_radius) == 0:
                print(f"[radius] seed={pid} 没有可用候选半径，跳过")
                continue

            # 6) 在所有n_fit候选中，选“最合适”的半径：
            #    采用稳健准则：离候选半径中位数最近
            cand_vals = np.array([x[1] for x in candidate_radius], dtype=float)
            cand_med = float(np.median(cand_vals))
            best_idx = int(np.argmin(np.abs(cand_vals - cand_med)))
            best_n_fit, best_r = candidate_radius[best_idx]

            print(
                f"[radius] seed={pid} 选择结果: n_fit={best_n_fit}, "
                f"r={best_r:.6f}, cand_median={cand_med:.6f}, "
                f"candidate_num={len(candidate_radius)}, linearity_best={linearity_best:.6f}"
            )
            radius_list.append(best_r)

        if len(radius_list) == 0:
            return self.radus

        radius = np.median(radius_list)

        
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
        处理代表点与邻居点：
        1. 先检查所有代表点是否都满足条件
        2. 如果全部满足，则当前邻域整体接收
        3. 如果有任意一个不满足，则对所有邻居点逐个检查
        返回:
            accepted_ids: 最终接收的点编号列表
            t_values: 通过检查时对应的曲线参数t列表
            all_rep_pass: 代表点是否全部通过
        """

        rep_t_values = []
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

            if not np.isfinite(dist_r):
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

            if dist_r > dist_thresh or cos_r < direction_cos_thresh:
                all_rep_pass = False
                break

            rep_t_values.append(t_r)

        accepted_ids = []

        # =========================
        # 2) 如果所有代表点都通过，则整团接收
        # =========================
        if all_rep_pass:
            for nid in local_neighbors:
                if nid not in visited:
                    accepted_ids.append(nid)

            return accepted_ids, rep_t_values, True

        # =========================
        # 3) 如果代表点没有全部通过，则逐点检查所有邻居点
        # =========================
        t_values = []

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

            if not np.isfinite(dist_i):
                continue

            if dist_i > dist_thresh:
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

        linearity_thresh = 0.7                    # 局部线性度阈值
        dist_thresh =  self.radus / 2              # 点到拟合曲线对应点的距离阈值
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
        queued = set(seed_ids)

        new_added_since_refit = 0
        refit_count = 0

        # =========================
        # 4. 区域生长
        # =========================
        while queue:
            # if time.perf_counter() - start_time > 10:
            #     print("曲线提取超时 10 秒，提前结束")
            #     break

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
                if fid not in queued:
                    queue.append(fid)
                    queued.add(fid)

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