import numpy as np
import sympy as sp
import open3d as o3d

from math import sqrt, pi, sin, cos, atan
from scipy.spatial import KDTree

np.set_printoptions(suppress=True)

def random_translation_transform(tx_range=(-1, 1), ty_range=(-1, 1), tz_range=(-1, 1)):
    tx = np.random.uniform(tx_range[0], tx_range[1])
    ty = np.random.uniform(ty_range[0], ty_range[1])
    tz = np.random.uniform(tz_range[0], tz_range[1])
    return np.array([[1, 0, 0, tx],
                     [0, 1, 0, ty],
                     [0, 0, 1, tz],
                     [0, 0, 0, 1]])

def generate_transform(t, a, b, x, y, z):
    theta_x = t
    theta_y = a
    theta_z = b
    Rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x), 0],
                   [0, np.sin(theta_x), np.cos(theta_x), 0],
                   [0, 0, 0, 1]])

    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y), 0],
                   [0, 1, 0, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y), 0],
                   [0, 0, 0, 1]])

    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0, 0],
                   [np.sin(theta_z), np.cos(theta_z), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    trans = np.array([[1, 0, 0, x],
                      [0, 1, 0, y],
                      [0, 0, 1, z],
                      [0, 0, 0, 1]])
    return trans.dot(Rx).dot(Ry).dot(Rz)

def random_rotation_transform(theta_range=(-np.pi, np.pi)):
    theta_x = np.random.uniform(theta_range[0], theta_range[1])
    theta_y = np.random.uniform(theta_range[0], theta_range[1])
    theta_z = np.random.uniform(theta_range[0], theta_range[1])

    Rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x), 0],
                   [0, np.sin(theta_x), np.cos(theta_x), 0],
                   [0, 0, 0, 1]])

    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y), 0],
                   [0, 1, 0, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y), 0],
                   [0, 0, 0, 1]])

    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0, 0],
                   [np.sin(theta_z), np.cos(theta_z), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    return Rx.dot(Ry).dot(Rz)

def random_rotation(x_range=(-np.pi, np.pi), y_range=(-np.pi, np.pi), z_range=(-np.pi, np.pi)):
    theta_x = np.random.uniform(x_range[0], x_range[1])
    theta_y = np.random.uniform(y_range[0], y_range[1])
    theta_z = np.random.uniform(z_range[0], z_range[1])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])

    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])

    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])

    return Rx.dot(Ry).dot(Rz)

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, m

# 定义函数：计算点到平面的距离
def point_to_plane_distance(point, plane_equation):
    normal, d = plane_equation
    return np.abs(np.dot(point, normal) + d) / np.linalg.norm(normal)

# 定义函数：计算点到平面的距离
def points_to_plane_distance(points, plane_equation):
    a, b, c, d = plane_equation
    normal = np.array([a, b, c])
    return np.abs(np.dot(points, normal) + d) / np.linalg.norm(normal)

# 定义函数：计算点到平面的距离
def points_to_plane_sign_distance(points, plane_equation):
    a, b, c, d = plane_equation
    normal = np.array([a, b, c])
    return (np.dot(points, normal) + d) / np.linalg.norm(normal)

# 定义函数：计算平面法向量
def compute_normal(plane_points):
    v1 = plane_points[1] - plane_points[0]
    v2 = plane_points[2] - plane_points[0]
    normal = np.cross(v1, v2)
    if np.all(normal == 0):
        return np.array([0,0,0])
    return normal / np.linalg.norm(normal)

# 定义函数：检查三个平面是否相互垂直
def check_orthogonal_planes(planes):
    for i in range(3):
        for j in range(i + 1, 3):
            dot_product = np.dot(planes[i][0], planes[j][0])
            if abs(dot_product) > 0.1:  # 不垂直
                return False
    return True

def check_2_orthogonal(plane1, plane2):
    dot_product = np.dot(plane1[0], plane2[0])
    if abs(dot_product) > 0.1:  # 不垂直
        return False
    return True

def check_2_Parallel(plane1, plane2):
    normal1_normalized = plane1[0] / np.linalg.norm(plane1[0])
    normal2_normalized = plane2[0] / np.linalg.norm(plane2[0])
    # 判断两个法向量是否平行
    if np.allclose(normal1_normalized, normal2_normalized, rtol=0.08) or np.allclose(normal1_normalized, -normal2_normalized, rtol=0.08):
        return True
    else:
        return False

def check_3_orthogonal(plane1, plane2, plane3):
    dot_product1 = np.dot(plane1[0], plane2[0])
    if abs(dot_product1) > 0.1:  # 不垂直
        return False
    dot_product2 = np.dot(plane1[0], plane3[0])
    if abs(dot_product2) > 0.1:  # 不垂直
        return False
    return True

# 定义函数：计算平面方程
def compute_plane_equation(plane_points):
    normal = compute_normal(plane_points)
    d = -np.dot(normal, plane_points[0])
    return normal, d

def point_to_plane_sign_distance(point, plane_equation):
    normal, d = plane_equation
    return np.dot(point, normal) + d / np.linalg.norm(normal)

def compute_plane_distance(plane1, plane2):
    # 提取法向量和偏移量
    normal1 = plane1[0]
    normal2 = plane2[0]
    D1 = plane1[1]
    D2 = plane2[1]
    # 计算距离
    distance = abs(np.dot(normal1, normal2) * D2 - D1) / np.linalg.norm(normal1)
    return distance

def angle_between_vectors(v1, v2):
    # 计算向量的点积
    dot_product = np.dot(v1, v2)
    # 计算向量的模
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # 计算夹角（弧度）
    angle_rad = np.arccos(dot_product / (norm_v1 * norm_v2))
    # 将弧度转换为角度
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def project_point_to_plane(points, plane_params):
    points = np.asarray(points)
    # 平面的法向量
    normal_vector = np.array(plane_params[:3])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    # 平面到原点的距离
    plane_distance = plane_params[3]
    # 点到平面的距离
    distance_to_plane = np.dot(points, normal_vector) + plane_distance
    # 投影点的坐标
    projected_points = points - np.outer(np.array([distance_to_plane]).T, np.array([normal_vector]))
    return projected_points

def points_to_point_distance(points, point):
    return np.linalg.norm(points - point, ord=2, axis=1) # 按行求坐标差的 2 范数，为点到点的距离


def rotation_matrix_from_z_axis(z_axis):
    # 1. 与 z 轴正交的任意向量
    print(np.abs(z_axis[0]))
    any_vector = np.array([1, 0, 0]) if np.abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])

    # 2. 第三个向量是 z 轴向量和任意向量的叉乘
    y_axis = np.cross(z_axis, any_vector)
    y_axis /= np.linalg.norm(y_axis)  # 归一化

    # 3. 第二个向量是 z 轴向量和第三个向量的叉乘
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)  # 归一化

    # 构建旋转变换矩阵
    rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T

    return rotation_matrix


# TODO: Distance calculate speed test, modified to matrix multiply whether better?
def point_to_general_quadratic_surface_dist(coeff, x):
    return abs(coeff[0]*x[:,0]**2 + coeff[1]*x[:,1]**2 + coeff[2]*x[:,2]**2 + coeff[3]*x[:,0]*x[:,1] + coeff[4]*x[:,0]*x[:,2] +
               coeff[5]*x[:,1]*x[:,2] + coeff[6]*x[:,0] + coeff[7]*x[:,1] + coeff[8]*x[:,2] + coeff[9])

def generate_general_quadratic_surface_funcA(points):
    A = []
    for x in points:
        A.append([x[0]**2, x[1]**2, x[2]**2, x[0] * x[1], x[0] * x[2], x[1] * x[2], x[0], x[1], x[2], 1])
    return A

def generate_ellipsoid(x0, y0, z0, a, b, c):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = x0 + a * np.outer(np.cos(u), np.sin(v))
    y = y0 + b * np.outer(np.sin(u), np.sin(v))
    z = z0 + c * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

def generate_ellipsoid_general(X):
    A, B, C, D, E, F, G, H, I, J = X
    # 根据参数方程生成椭球上的点
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # 根据椭球的一般方程进行转换
    equation = A*x**2 + B*y**2 + C*z**2 + 2*D*x*y + 2*E*x*z + 2*F*y*z + 2*G*x + 2*H*y + 2*I*z + J

    # 令方程等于零，得到椭球上的点
    x = x[equation <= 0]
    y = y[equation <= 0]
    z = z[equation <= 0]
    return x, y, z

def circle_from_three_points_in_plane(points, plane):
    x1, x2, x3 = points[0][0], points[1][0], points[2][0]
    y1, y2, y3 = points[0][1], points[1][1], points[2][1]
    z1, z2, z3 = points[0][2], points[1][2], points[2][2]
    A = np.array([[2 * (x1 - x2), 2 * (y1 - y2), 2 * (z1 - z2)],
                  [2 * (x1 - x3), 2 * (y1 - y3), 2 * (z1 - z3)],
                  [2 * (x2 - x3), 2 * (y2 - y3), 2 * (z2 - z3)],
                  [plane[0], plane[1], plane[2]]])
    B = np.array([[x1**2 + y1**2 + z1**2 - x2**2 - y2**2 - z2**2],
                  [x1**2 + y1**2 + z1**2 - x3**2 - y3**2 - z3**2],
                  [x2**2 + y2**2 + z2**2 - x3**2 - y3**2 - z3**2],
                  [-plane[3]]])
    X = np.linalg.lstsq(A, B, rcond=None)[0]
    h, k, l = X[0][0], X[1][0], X[2][0]
    r = np.linalg.norm(points[0] - np.array([h, k, l]))
    return (h, k, l), r

def circle_from_three_points_2D(points):
    x1, x2, x3 = points[0][0], points[1][0], points[2][0]
    y1, y2, y3 = points[0][1], points[1][1], points[2][1]
    A = np.array([[2 * (x1 - x2), 2 * (y1 - y2)],
                  [2 * (x1 - x3), 2 * (y1 - y3)],
                  [2 * (x2 - x3), 2 * (y2 - y3)]])
    B = np.array([[x1**2 + y1**2 - x2**2 - y2**2],
                  [x1**2 + y1**2 - x3**2 - y3**2],
                  [x2**2 + y2**2 - x3**2 - y3**2],])
    X = np.linalg.lstsq(A, B, rcond=None)[0]
    h, k = X[0][0], X[1][0]
    r = np.linalg.norm(points[0] - np.array([h, k]))
    return (h, k), r

def fit_circumcircle_in_plane(points, num_iterations, plane):
    points = np.asarray(points)
    num_total_points = len(points)
    best_circle = None
    best_circle_points = None
    best_num_inliers = 0
    remained_points_indices = None
    for _ in range(num_iterations):
        # 随机采样3组3个点
        random_indices = np.random.choice(num_total_points, 3, replace=False)
        circle_points = points[random_indices]
        center, r = circle_from_three_points_in_plane(circle_points, plane)
        outliers = np.where(abs(points_to_point_distance(points, center) - r) > 0.005)[0]
        # outliers = np.where(points_to_point_distance(points, center) - r < 0.01)[0]
        num_inliers = num_total_points - len(outliers)
        if num_inliers > best_num_inliers:
            best_circle = (center, r)
            best_num_inliers = num_inliers
            best_circle_points = circle_points
            remained_points_indices = outliers
    return best_circle, best_num_inliers, remained_points_indices, best_circle_points

def create_plane(A, B, C, D, rgb):
    normal = np.array([A, B, C])
    v1, v2 = find_orthogonal_vectors(normal)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    normal = normal / np.linalg.norm(normal)
    R = np.array([v1, v2, normal])
    x, y, z = sp.symbols('x y z')
    xp, yp, zp = sp.symbols('xp yp zp')
    plane_expr = A * x + B * y + C * z + D
    plane_xyz = sp.Matrix([[xp, yp, zp]]) * R
    trans_plane_expr = plane_expr.subs({x: plane_xyz[0], y: plane_xyz[1], z: plane_xyz[2]})
    plane_expr_rot = trans_plane_expr.expand()
    plane_coefficients_dict = plane_expr_rot.as_coefficients_dict(xp, yp, zp, 1)
    for term, coefficient in plane_coefficients_dict.items():
        if abs(coefficient) < 1e-10:
            plane_expr_rot = plane_expr_rot.subs(term, 0)
    [A, B, C, D] = [plane_coefficients_dict[xp], plane_coefficients_dict[yp], plane_coefficients_dict[zp], plane_coefficients_dict[1]]
    # 生成网格的顶点坐标
    times = 1
    num_grid = 11
    if abs(A) > 1e-10:
        y = np.linspace(-1, 1, num_grid)*times  # y轴上的坐标范围
        z = np.linspace(-1, 1, num_grid)*times  # z轴上的坐标范围
        Y, Z = np.meshgrid(y, z)
        X = (- B * Y - C * Z - D) / A
    elif abs(B) > 1e-10:
        x = np.linspace(-1, 1, num_grid)*times  # x轴上的坐标范围
        z = np.linspace(-1, 1, num_grid)*times  # z轴上的坐标范围
        X, Z = np.meshgrid(x, z)
        Y = (-A * X - C * Z - D) / B
    elif abs(C) > 1e-10:
        x = np.linspace(-1, 1, num_grid)*times  # x轴上的坐标范围
        y = np.linspace(-1, 1, num_grid)*times  # y轴上的坐标范围
        X, Y = np.meshgrid(x, y)
        Z = (-A * X - B * Y - D) / C
    else:
        print("平面方程错误")
    points = np.array([X.flatten(), Y.flatten(), Z.flatten()])
    points_stack = np.column_stack(points)
    # 创建平面
    triangles = []
    for i in range(num_grid - 1):
        for j in range(num_grid - 1):
            idx1 = i * num_grid + j
            idx2 = idx1 + 1
            idx3 = (i + 1) * num_grid + j
            idx4 = idx3 + 1
            triangles.append([idx1, idx2, idx4])
            triangles.append([idx1, idx4, idx3])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points_stack)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.paint_uniform_color(rgb)
    mesh.rotate(R.T, [0,0,0])
    return mesh

def find_orthogonal_vectors(normal_vector):
    """
    根据平面的法向量找到相互垂直的两个向量
    """
    # 生成两个随机向量
    v1 = np.random.rand(3)
    v2 = np.random.rand(3)
    # 计算第一个向量在法向量上的投影
    v1_proj = np.dot(v1, normal_vector) / np.linalg.norm(normal_vector) * normal_vector
    # 计算第一个向量与投影的差，得到第一个相互垂直的向量
    v1_ortho = v1 - v1_proj
    # 计算第二个向量与第一个相互垂直的向量的叉乘，得到第二个相互垂直的向量
    v2_ortho = np.cross(normal_vector, v1_ortho)
    return v1_ortho, v2_ortho

def get_plane(pcd, dist_threshold, n, num_it):
    plane_model, inliers = pcd.segment_plane(distance_threshold=dist_threshold, ransac_n=n, num_iterations=num_it)
    num_inliers = len(inliers)
    plane_cloud = pcd.select_by_index(inliers)
    remained_cloud = pcd.select_by_index(inliers, invert=True)
    return plane_model, inliers, num_inliers, plane_cloud, remained_cloud

def get_plane_intersection_point(planes):
    assert len(planes) == 3, "Only three planes can calculate intersection point"
    A = np.array([[planes[0][0], planes[0][1], planes[0][2]],
                  [planes[1][0], planes[1][1], planes[1][2]],
                  [planes[2][0], planes[2][1], planes[2][2]]])
    b = np.array([-planes[0][3], -planes[1][3], -planes[2][3]])
    intersection_point = np.linalg.solve(A, b)
    return intersection_point

def trans_points_coordinate(points, T):
    pointsTransed = []
    for p in points:
        pointsTransed.append(T.dot(np.insert(p, 3, 1).T)[:3])
    return np.array(pointsTransed)


def fit_ellipse_by_formula(points, num_iterations, threshold=0.01):
    num_total_points = len(points)
    best_num_inliers = 0
    best_ellipse = None
    remained_points_indices = None
    # 迭代拟合
    for _ in range(num_iterations):
        random_indices = np.random.choice(len(points), 10, replace=False)
        sample_points = points[random_indices]
        A = np.array(generate_ellipse_general_funcA(sample_points))
        B = np.zeros(A.shape[0])
        _, _, V = np.linalg.svd(A)
        X = V[-1]
        outliers = np.where(point_to_ellipse_dist_general(X, points) > threshold)[0]
        num_inliers = num_total_points - len(outliers)
        # 更新最佳平面
        if num_inliers > best_num_inliers:
            [a,b,c,d,e,f] = X
            xc = (2*c*d-b*e) / (b**2-4*a*c)
            yc = (2*a*e-b*d) / (b**2-4*a*c)
            center = [xc, yc]
            t = (pi - atan((c-a)/b)) / 2 # arccot(x) = pi/2 - arctan(x)
            a_square = (a*xc**2+b*xc*yc+c*yc**2-f) / (a*cos(t)**2+b*sin(t)*cos(t)+c*sin(t)**2)
            b_square = (a*xc**2+b*xc*yc+c*yc**2-f) / (c*cos(t)**2-b*sin(t)*cos(t)+a*sin(t)**2)
            if (a_square < 0 or b_square < 0):
                continue
            a = sqrt(a_square)
            b = sqrt(b_square)
            # if (a < b):
            #     tmp = a
            #     a = b
            #     b = tmp
            # print("update1")
            best_ellipse = (X, center, a, b, t, sample_points, num_inliers)
            best_num_inliers = num_inliers
            remained_points_indices = outliers
    return best_ellipse, best_num_inliers, remained_points_indices

def fit_circle(points, num_iterations, threshold=0.01):
    points = np.asarray(points)
    num_total_points = len(points)
    best_circle = None
    # best_circle_points = None
    best_num_inliers = 0
    remained_points_indices = None
    for _ in range(num_iterations):
        # 随机采样3组3个点
        random_indices = np.random.choice(num_total_points, 3, replace=False)
        circle_points = points[random_indices]
        x1, x2, x3 = circle_points[0][0], circle_points[1][0], circle_points[2][0]
        y1, y2, y3 = circle_points[0][1], circle_points[1][1], circle_points[2][1]
        A = np.array([[2 * (x1 - x2), 2 * (y1 - y2)],
                      [2 * (x1 - x3), 2 * (y1 - y3)],
                      [2 * (x2 - x3), 2 * (y2 - y3)]])
        B = np.array([[x1**2 + y1**2 - x2**2 - y2**2],
                      [x1**2 + y1**2 - x3**2 - y3**2],
                      [x2**2 + y2**2 - x3**2 - y3**2]])
        X = np.linalg.lstsq(A, B, rcond=None)[0]
        center = np.array([X[0][0], X[1][0]])
        r = np.linalg.norm(circle_points[0] - center)
        outliers = np.where(abs(points_to_point_distance(points, center) - r) > threshold)[0]
        num_inliers = num_total_points - len(outliers)
        if num_inliers > best_num_inliers:
            best_circle = (center, r)
            best_num_inliers = num_inliers
            # best_circle_points = circle_points
            remained_points_indices = outliers
    return best_circle, best_num_inliers, remained_points_indices

def fit_ellipse(points, num_iterations, threshold=0.01):
    num_total_points = len(points)
    best_num_inliers = 0
    best_ellipse = None
    remained_points_indices = None
    x, y = sp.symbols('x y')
    xp, yp = sp.symbols('xp yp')
    for _ in range(num_iterations):
        random_indices = np.random.choice(num_total_points, 20, replace=False)
        sample_points = points[random_indices]
        A = np.array(generate_ellipse_general_funcA(sample_points))
        B = np.zeros(A.shape[0])
        _, _, V = np.linalg.svd(A)
        X = V[-1]
        outliers = np.where(point_to_ellipse_dist_general(X, points) > threshold)[0]
        num_inliers = num_total_points - len(outliers)
        if num_inliers > best_num_inliers:
            [a,b,c,d,e,f] = X
            t = (pi - atan((c-a)/b)) / 2 # arccot(x) = pi/2 - arctan(x)
            expr = a*x**2 + b*x*y + c*y**2 + d*x + e*y + f
            R = sp.Matrix([[cos(t), -sin(t)],
                           [sin(t),  cos(t)]])
            xyz = sp.Matrix([[xp, yp]]) * R
            trans = expr.subs({x: xyz[0], y: xyz[1]})
            var = (xp**2, yp**2, xp*yp, xp, yp, 1)
            expr_trans = trans.expand()
            coefficients_dict = expr_trans.as_coefficients_dict(*var)
            for term, coefficient in coefficients_dict.items():
                if abs(coefficient) < 1e-5:
                    expr_trans = expr_trans.subs(term, 0)
            if (coefficients_dict[xp**2]*coefficients_dict[yp**2] < 0):
                continue
            if (abs(coefficients_dict[xp**2]*coefficients_dict[yp**2]) < 1e-5):
                continue
            if (coefficients_dict[xp**2] < 0):
                expr_trans = expr_trans * (-1)
            coefficients_dict = expr_trans.as_coefficients_dict(*var)
            coeff_xp2 = coefficients_dict[xp**2]
            coeff_xp = coefficients_dict[xp]
            coeff_yp2 = coefficients_dict[yp**2]
            coeff_yp = coefficients_dict[yp]
            x0 = -0.5 * coeff_xp / coeff_xp2
            y0 = -0.5 * coeff_yp / coeff_yp2
            Cx = coeff_xp2 * x0**2
            Cy = coeff_yp2 * y0**2
            constJ = coefficients_dict[1] - Cx - Cy
            if constJ > 0:
                continue
            print("update")
            a = sqrt(-constJ / coeff_xp2)
            b = sqrt(-constJ / coeff_yp2)
            center = sp.Matrix([[x0, y0]]) * R
            best_ellipse = (X, center, a, b, t, sample_points, num_inliers)
            best_num_inliers = num_inliers
            remained_points_indices = outliers
    return best_ellipse, best_num_inliers, remained_points_indices

def generate_ellipse_general_funcA(points): # 椭圆的一般方程为：F(x,y)=ax^2+bxy+cy^2+dx+ey+f=0
    # https://blog.csdn.net/weixin_41674673/article/details/135864871
    A = []
    for x in points:
        A.append([x[0]**2, x[0]*x[1], x[1]**2, x[0], x[1], 1])
    return A

def points_to_2D_coord(points, plane_origin, x_axis, y_axis):
    plane_coordinates = []
    for point in points:
        # 在平面坐标系中计算投影点的坐标
        x_coord = np.dot(point - plane_origin, x_axis)
        y_coord = np.dot(point - plane_origin, y_axis)
        plane_coordinates.append([x_coord, y_coord])
    return np.array(plane_coordinates)

def convert_2d_to_3d_coord(points, plane_origin, x_axis, y_axis):
    coord_3D = []
    # 将二维坐标转换为平面坐标系中的向量
    for point in points:
        plane_vectors = point[0] * x_axis + point[1] * y_axis
        # 将平面坐标系中的向量转换为平面上的投影点
        coord_3D.append(plane_origin + plane_vectors)
    return coord_3D

def point_to_ellipse_dist_general(coeff, x):
    return abs(coeff[0]*x[:,0]**2 + coeff[1]*x[:,0]*x[:,1] + coeff[2]*x[:,1]**2 + coeff[3]*x[:,0] + coeff[4]*x[:,1] + coeff[5])

def generate_ellipse_xy(x0, y0, a, b, angle):
    theta = np.linspace(0, 2*np.pi, 100)
    x = x0 + a * np.cos(theta) * np.cos(angle) - b * np.sin(theta) * np.sin(angle)
    y = y0 + a * np.cos(theta) * np.sin(angle) + b * np.sin(theta) * np.cos(angle)
    return x,y

def generate_circle_xy(x0, y0, r):
    theta = np.linspace(0, 2 * np.pi, 100)
    x = x0 + r * np.cos(theta)
    y = y0 + r * np.sin(theta)
    return x,y

def generate_cube(center, R, whd):
    cube = o3d.geometry.TriangleMesh.create_box(width=whd[0], height=whd[1], depth=whd[2])
    cube.translate(center)
    cube.rotate(R, center)
    return cube
def plane_filter_radius(point_cloud, r=0.05, dist_threshold=0.1):
    # 估计法向量
    point_cloud.estimate_normals()

    # K 最近邻点搜索
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)
    plane_points_idx = []
    for i in range(len(point_cloud.points)):
        _, k_indices, _ = kdtree.search_radius_vector_3d(point_cloud.points[i], r)
        k_points = point_cloud.select_by_index(k_indices)
        avg_normal_diff = np.linalg.norm(np.asarray(k_points.normals) - np.mean(np.asarray(k_points.normals), axis=0), axis=1)
        if len(np.where(avg_normal_diff < dist_threshold)[0]) > len(k_indices) / 2:
            for idx in k_indices:
                plane_points_idx.append(idx)
    # 创建新的点云对象并设置法向量
    filtered_pcd = point_cloud.select_by_index(plane_points_idx)

    return filtered_pcd

def plane_filter_o3d(pcd, distance_threshold=0.001,ransac_n=3,num_iterations=1000):
    pcd.estimate_normals()
    # 提取平面点
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
    # 提取平面点和非平面点
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return outlier_cloud

def quad_curve_filter(point_cloud, num_iterations=1000,cone_dist_threshold=5e-4):
    num_total_points = len(np.asarray(point_cloud.points))
    best_num_inliers = 0
    best_cone = None
    remained_points_indices = None
    x, y, z = sp.symbols('x y z')
    xp, yp, zp = sp.symbols('xp yp zp')
    for _ in range(num_iterations):
        random_indices = np.random.choice(len(point_cloud.points), 10, replace=False)
        sample_points = np.asarray(point_cloud.points)[random_indices]
        A_Matrix = np.array(generate_general_quadratic_surface_funcA(sample_points))
        U, S, V = np.linalg.svd(A_Matrix)
        X = V[-1]
        [a,b,c,d,e,f,g,h,i,j] = X
        outliers = np.where(point_to_general_quadratic_surface_dist(X, np.asarray(point_cloud.points)) > cone_dist_threshold)[0]
        num_inliers = num_total_points - len(outliers)
        if num_inliers > best_num_inliers:
            expr = a*x**2 + b*y**2 + c*z**2 + d*x*y + e*x*z + f*y*z + g*x + h*y + i*z + j
            A0 = sp.Matrix([[a, d/2, e/2],
                            [d/2, b, f/2],
                            [e/2, f/2, c]])
            eigenvalues = sorted(A0.eigenvals())
            eigenvectors = sorted(A0.eigenvects(), key=lambda x: abs(x[0]))
            T = eigenvectors[2][2][0].col_insert(0, eigenvectors[1][2][0]).col_insert(0, eigenvectors[0][2][0])
            xyz = sp.Matrix([[xp, yp, zp]]) * T.T
            trans = expr.subs({x: xyz[0], y: xyz[1], z: xyz[2]})
            if sum(1 for e in eigenvalues if e < 0) == 2:
                trans = trans * (-1)
            if sum(1 for e in eigenvalues if e > 0) != 2:
                if sum(1 for e in eigenvalues if e < 0) != 2:
                    continue
            var = (xp**2, yp**2, zp**2, xp*yp, xp*zp, yp*zp, xp, yp, zp, 1)
            expr_rot = trans.expand()
            coefficients_dict = expr_rot.as_coefficients_dict(*var)
            for term, coefficient in coefficients_dict.items():
                if abs(coefficient) < 1e-10:
                    expr_rot = expr_rot.subs(term, 0)
            coeff_xp2 = expr_rot.coeff(xp**2)
            coeff_xp = expr_rot.coeff(xp)
            coeff_yp2 = expr_rot.coeff(yp**2)
            coeff_yp = expr_rot.coeff(yp)
            coeff_zp2 = expr_rot.coeff(zp**2)
            coeff_zp = expr_rot.coeff(zp)
            x0 = -0.5 * coeff_xp / coeff_xp2
            y0 = -0.5 * coeff_yp / coeff_yp2
            z0 = -0.5 * coeff_zp / coeff_zp2
            Cx = coeff_xp2 * x0**2
            Cy = coeff_yp2 * y0**2
            Cz = coeff_zp2 * z0**2
            constJ = coefficients_dict[1] - Cx - Cy - Cz
            a = sqrt(abs(constJ) / abs(coeff_xp2))
            b = sqrt(abs(constJ) / abs(coeff_yp2))
            c = sqrt(abs(constJ) / abs(coeff_zp2))
            best_cone = (X, [x0, y0, z0], [a, b, c], expr_rot, T, outliers)
            best_num_inliers = num_inliers
    # R = best_cone[4]
    # cone_cloud = point_cloud.select_by_index(best_cone[5], invert=True)
    # cone_cloud.paint_uniform_color([0, 0, 1])
    # cone_outlier_cloud = point_cloud.select_by_index(best_cone[5])
    # cone_outlier_cloud.paint_uniform_color([1, 1, 0])
    return best_cone

def nearest_neighbor_distance(source_points, target_points):
    # 构建 KD 树
    kd_tree = KDTree(target_points)

    # 查询每个源点云中每个点的最近邻点及其距离
    distances, _ = kd_tree.query(source_points)

    # 返回最近点距离
    return distances

if __name__ == "__main__":
    # 生成随机平移矩阵
    translation_matrix = random_translation_transform((-2000, 2000), (-2000, 2000), (0, 2000))

    # 生成随机旋转矩阵
    rotation_matrix = random_rotation_transform()

    # 合并平移和旋转
    transformation_matrix = translation_matrix.dot(rotation_matrix)

    print("随机生成的变换矩阵：")
    print(transformation_matrix)
