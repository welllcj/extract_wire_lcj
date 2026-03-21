import numpy as np
def point_to_curve_distance_newton( p, direction_fit, centroid_fit, ax, ay, az,
                                   max_iter=8, tol=1e-6):
    """
    用牛顿迭代法计算点 p 到参数曲线的近似最短距离

    参数
    ----
    p : ndarray shape (3,)
        原始点坐标
    direction_fit : ndarray shape (3,)
        当前拟合得到的 PCA 主方向（单位向量更好）
    centroid_fit : ndarray shape (3,)
        当前拟合点集中心
    ax, ay, az : ndarray
        参数方程 x(t), y(t), z(t) 的多项式系数
    max_iter : int
        牛顿迭代最大次数
    tol : float
        收敛阈值

    返回
    ----
    dist_to_curve : float
        点 p 到曲线最近点 q 的距离
    q : ndarray shape (3,)
        曲线上距离 p 最近的点坐标
    t_cur : float
        最近点对应的参数值
    """

    # =========================
    # 1. 用 PCA 投影作为初值
    # =========================
    t_cur = np.dot(p - centroid_fit, direction_fit)

    order = len(ax) - 1

    for _ in range(max_iter):
        # r(t)
        r = np.zeros(3)
        for i in range(order + 1):
            r[0] += ax[i] * t_cur**i
            r[1] += ay[i] * t_cur**i
            r[2] += az[i] * t_cur**i

        # r'(t)
        r1 = np.zeros(3)
        for i in range(1, order + 1):
            r1[0] += i * ax[i] * t_cur**(i - 1)
            r1[1] += i * ay[i] * t_cur**(i - 1)
            r1[2] += i * az[i] * t_cur**(i - 1)

        # r''(t)
        r2 = np.zeros(3)
        for i in range(2, order + 1):
            r2[0] += i * (i - 1) * ax[i] * t_cur**(i - 2)
            r2[1] += i * (i - 1) * ay[i] * t_cur**(i - 2)
            r2[2] += i * (i - 1) * az[i] * t_cur**(i - 2)

        diff = r - p

        # g(t) = (r(t) - p) · r'(t)
        g = np.dot(diff, r1)

        # g'(t) = r'(t)·r'(t) + (r(t)-p)·r''(t)
        gp = np.dot(r1, r1) + np.dot(diff, r2)

        # 防止分母过小
        if abs(gp) < 1e-12:
            break

        t_next = t_cur - g / gp

        if abs(t_next - t_cur) < tol:
            t_cur = t_next
            break

        t_cur = t_next

    # =========================
    # 2. 用最终 t_cur 计算最近点 q
    # =========================
    q = np.zeros(3)
    for i in range(order + 1):
        q[0] += ax[i] * t_cur**i
        q[1] += ay[i] * t_cur**i
        q[2] += az[i] * t_cur**i

    dist_to_curve = np.linalg.norm(p - q)

    return dist_to_curve, q, t_cur