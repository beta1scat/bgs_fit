import os
import open3d as o3d
import numpy as np
from spatialmath import SO3, SE3
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
# For ROS2
# from .utils import *
# from .generatePickPoses import *
from utils import *
from generatePickPoses import *

def align_vector_to_z(v):
    """ 生成一个旋转矩阵，使 Z 轴对齐到向量 v """
    v = np.array(v, dtype=np.float64)
    v = v / np.linalg.norm(v)  # 归一化

    # 选择辅助向量，确保它与 v 不共线
    if np.abs(v[0]) < np.abs(v[1]):
        aux = np.array([1, 0, 0], dtype=np.float64)
    else:
        aux = np.array([0, 1, 0], dtype=np.float64)

    # 计算 X 轴
    x_axis = np.cross(aux, v)
    x_axis /= np.linalg.norm(x_axis)

    # 计算 Y 轴
    y_axis = np.cross(v, x_axis)

    # 旋转矩阵
    R = np.column_stack((x_axis, y_axis, v))  # R = [X' Y' Z']
    return R

def fit_cuboid_obb2(pcd, dist_threshold=0.001, n=3, num_it=500):
    '''
        Fit cuboid by "Oriented Bounding Box"
    Params:
    - pcd: input point cloud
    Return:
    - a: side length along X axis
    - b: side length along Y axis
    - c: side length along Z axis
    - T: rigid transform of the cuboid
    '''
    plane_model, inliers = pcd.segment_plane(distance_threshold=dist_threshold, ransac_n=n, num_iterations=num_it)
    remained_cloud = pcd.select_by_index(inliers, invert=True)
    remained_cloud.paint_uniform_color([0, 1, 0])
    plane_cloud = pcd.select_by_index(inliers)
    plane_cloud.paint_uniform_color([1, 1, 0])
    plane_model2, inliers2 = remained_cloud.segment_plane(distance_threshold=dist_threshold, ransac_n=n, num_iterations=num_it)
    # o3d.visualization.draw_geometries([plane_cloud, remained_cloud], point_show_normal=True)
    v1 = plane_model[:3]
    v2 = plane_model2[:3]
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    R = SO3.TwoVectors(v1, v2)
    pcd_transed = o3d.geometry.PointCloud()
    pcd_transed.points = o3d.utility.Vector3dVector(pcd.points)
    pcd_transed.rotate(R.inv(), [0,0,0])
    aabb = pcd_transed.get_axis_aligned_bounding_box()
    aabb.color = [0,0,1]
    a, b, c = aabb.get_half_extent()
    cube_center = SO3(R) * aabb.get_center()
    T = SE3.Rt(R, cube_center)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd_transed, aabb, coord_frame], point_show_normal=True)
    return a, b, c, T

def fit_cuboid_obb(pcd):
    '''
        Fit cuboid by "Oriented Bounding Box"
    Params:
    - pcd: input point cloud
    Return:
    - a: side length along X axis
    - b: side length along Y axis
    - c: side length along Z axis
    - T: rigid transform of the cuboid
    '''
    obb = pcd.get_minimal_oriented_bounding_box()
    T = SE3.Rt(obb.R, obb.center)
    pcd.transform(T.inv())
    aabb = pcd.get_axis_aligned_bounding_box()
    pcd.transform(T)
    a = aabb.max_bound[0]
    b = aabb.max_bound[1]
    c = aabb.max_bound[2]
    return a, b, c, T

def fit_frustum_cone_by_slice_linear(points, num_layers=10):
    '''
        Fit frustum cone by slice points along Z-axis
    Params:
    - points: input points
    - num_layers: the number of slice layers
    Return:
    - r1: radius of max Z slice layers
    - r2: radius of min Z slice layers
    - height: max Z - min Z
    - center: center coordinate at X-Y plane [x,y]
    '''
    x, y, z = points[:,0], points[:,1], points[:,2]
    z_min = z.min()
    z_max = z.max()
    height = z_max - z_min
    z_step = height / num_layers
    layer_pts = []
    for i in range(num_layers):
        layer_min = z_min + i * height / num_layers
        layer_max = z_min + (i + 1) * height / num_layers
        mask = (z >= layer_min) & (z < layer_max)
        layer_pts.append(points[mask])
    layer_radius = {}
    layer_center = {}
    for layer_idx in range (num_layers):
        layer_idx_pts = layer_pts[layer_idx]
        cirModel = CircleLeastSquaresModel()
        bestmodel, inliers = ransac(layer_idx_pts[:,:2], cirModel, 4, 10, 0.01, 10, debug=False, return_all=True)
        if bestmodel is not None:
            layer_radius[layer_idx] = bestmodel[2]
            layer_center[layer_idx] = bestmodel[:2]
    ransac_linear = linear_model.RANSACRegressor()
    X_to_fit = np.array(list(layer_radius.keys()))[:, np.newaxis]
    y_to_fit = np.array(list(layer_radius.values()))
    ransac_linear.fit(X_to_fit, y_to_fit)
    inlier_mask = np.where(ransac_linear.inlier_mask_)[0]
    outlier_mask = np.where(np.logical_not(ransac_linear.inlier_mask_))[0]
    line_x_ransac = np.array(range(num_layers))[:, np.newaxis]
    line_y_ransac = ransac_linear.predict(line_x_ransac)
    r2 = line_y_ransac[0] # From min to max, so r2 is bottom
    r1 = line_y_ransac[-1]
    layer_center = np.array(list(layer_center.values()))
    center = np.array([np.mean(layer_center[inlier_mask, 0]), np.mean(layer_center[inlier_mask, 1]), (z_min + z_max) / 2])
    return r1, r2, height, center

def fit_frustum_cone_by_slice_poly(points, num_layers=10):
    '''
        Fit frustum cone by slice points along Z-axis
    Params:
    - points: input points
    - num_layers: the number of slice layers
    Return:
    - r1: radius of max Z slice layers
    - r2: radius of min Z slice layers
    - height: max Z - min Z
    - center: center coordinate at X-Y plane [x,y]
    '''
    x, y, z = points[:,0], points[:,1], points[:,2]
    z_min = z.min()
    z_max = z.max()
    height = z_max - z_min
    z_step = height / num_layers
    layer_pts = []
    for i in range(num_layers):
        layer_min = z_min + i * height / num_layers
        layer_max = z_min + (i + 1) * height / num_layers
        mask = (z >= layer_min) & (z < layer_max)
        layer_pts.append(points[mask])
    layer_radius = {}
    layer_center = {}
    for layer_idx in range (num_layers):
        layer_idx_pts = layer_pts[layer_idx]
        cirModel = CircleLeastSquaresModel()
        bestmodel, inliers = ransac(layer_idx_pts[:,:2], cirModel, 3, 100, 0.01, 10, debug=False, return_all=True)
        if bestmodel is not None:
            layer_radius[layer_idx] = bestmodel[2]
            layer_center[layer_idx] = bestmodel[:2]
    X_to_fit = np.array(list(layer_radius.keys()))[:, np.newaxis]
    y_to_fit = np.array(list(layer_radius.values()))
    model = make_pipeline(PolynomialFeatures(degree=2), RANSACRegressor(LinearRegression()))
    model.fit(X_to_fit, y_to_fit.ravel())
    ransacModel = model.named_steps['ransacregressor']
    inlier_mask = ransacModel.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    # inlier_mask = np.where(ransac.inlier_mask_)[0]
    # outlier_mask = np.where(np.logical_not(ransac.inlier_mask_))[0]
    line_x_ransac = np.array(range(num_layers))[:, np.newaxis]
    line_y_ransac = model.predict(line_x_ransac)
    r2 = line_y_ransac[0] # From min to max, so r2 is bottom
    r1 = line_y_ransac[-1]
    layer_center = np.array(list(layer_center.values()))
    center = np.array([np.mean(layer_center[inlier_mask, 0]), np.mean(layer_center[inlier_mask, 1]), (z_min + z_max) / 2])
    return r1, r2, height, center

def fit_frustum_cone_ransac(pcd):
    '''
        Fit frustum cone by RANSAC method
    Params:
    - pcd: input point cloud
    Return:
    - r1: radius 1
    - r2: radius 2
    - height: height of the frustum cone
    - T: rigid transform of the frustum cone at center
    '''
    points = np.asarray(pcd.points)
    x, y, z = points[:,0].copy(), points[:,1], points[:,2]
    normals = np.asarray(pcd.normals)
    num_points = len(points)
    normalModel = NormalLeastSquaresModel()
    bestNormalModel, normalInliers = ransac(normals, normalModel, 3, 1000, 0.02, num_points*0.2, inliers_ratio=0.8, debug=False, return_all=True)
    best_normal, best_angle = bestNormalModel
    if best_normal is None:
        print("Normal fit error")
        exit()
    dir_x = np.cross(best_normal, [0,0,1])
    if (np.allclose(dir_x, np.array([0,0,0]))):
        dir_x = np.cross(best_normal, [1,0,0])
    R = SO3.TwoVectors(x=dir_x, z=best_normal)
    pcd.rotate(R.inv(),center=[0,0,0])

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, coord_frame], point_show_normal=True)

    r1, r2, height, center = fit_frustum_cone_by_slice(points, 30)
    return r1, r2, height, SE3.Rt(R, center)

def fit_frustum_cone_normal(pcd, use_poly=False, plane_t=0.001, normal_t=0.02, use_plane_normal=True):
    pcd_normalized = o3d.geometry.PointCloud(pcd)
    pts, m, centroid = pc_normalize(np.asarray(pcd.points))
    pcd_normalized.points = o3d.utility.Vector3dVector(pts)
    points = np.asarray(pcd_normalized.points)
    normals = np.asarray(pcd_normalized.normals)
    num_points = len(points)
    num_clusters = 10
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, tol = 0.01).fit(normals)
    labels = kmeans.labels_
    cluster_normals_list = []
    plane_normals_list = []
    plane_points_ratio_list = []
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster_indices_size = len(cluster_indices)
        if cluster_indices_size < num_points / 10:
            continue
        cluster_pcd = pcd_normalized.select_by_index(cluster_indices)
        cluster_normal = np.mean(normals[cluster_indices], axis=0)
        cluster_normals_list.append(cluster_normal)
        plane_model, plane_inliers = cluster_pcd.segment_plane(distance_threshold=plane_t, ransac_n=3, num_iterations=1000)
        plane_normals_list.append(plane_model[:3])
        ratio = len(plane_inliers) / cluster_indices_size
        plane_points_ratio_list.append(ratio)
        # print(f"idx: {i}, plane_normals_list: {plane_model[:3]}, cluster_normal: {cluster_normal}")
        # print(f"pts: {cluster_indices_size}, Ratio: {ratio}")
        # plane_pcd = cluster_pcd.select_by_index(plane_inliers)
        # plane_pcd.paint_uniform_color([1,0,0])
        # o3d.visualization.draw_geometries([cluster_pcd, plane_pcd], window_name='Classified Point Clouds', point_show_normal=True)
    print(plane_points_ratio_list)
    # if np.max(plane_points_ratio_list) > 0.7 and len([x for x in plane_points_ratio_list if x > 0.7]) < 3:
    if use_plane_normal:
    # if np.max(plane_points_ratio_list) > 0.7:
        print(f"使用平面法向量")
        max_plane_ratio_idx = np.argmax(plane_points_ratio_list)
        cone_normal = plane_normals_list[max_plane_ratio_idx]
    else:
        print(f"使用估计法向量")
        cone_axis_model = ConeAxisLeastSquaresModel()
        cluster_normals_list = np.array(cluster_normals_list)
        best_fit, best_inlier_idxs = ransac(normals, cone_axis_model, 10, 1000, normal_t, 1, inliers_ratio=0.9, debug=False, return_all=True)
        vector, best_angle = best_fit
        cone_normal = vector
    vec_x = np.cross(cone_normal, [0,0,1])
    R = SO3.TwoVectors(x=vec_x, z=cone_normal)
    pcd_normalized.rotate(R.inv(), center=[0,0,0])
    """ Fit cone radius  """
    if use_poly:
        r1, r2, height, center = fit_frustum_cone_by_slice_poly(points, 30)
    else:
        r1, r2, height, center = fit_frustum_cone_by_slice_linear(points, 30)
    return r1 * m, r2 * m, height * m, SE3(centroid) * SE3(R) * SE3(np.asarray(center) * m)

def get_plane_axis(point_cloud, halfDim):
    pts = np.asarray(point_cloud.points)
    dist_sum_list = []
    for idx in np.arange(3):
        idx1, idx2 = [i for i in range(3) if i != idx]
        corner = np.array([[halfDim[idx1], halfDim[idx2]], [-1 * halfDim[idx1], -1 * halfDim[idx2]],
                           [-1 * halfDim[idx1], halfDim[idx2]], [halfDim[idx1], -1 * halfDim[idx2]]])
        pts2d = np.array([pts[:, idx1], pts[:, idx2]])
        dist_sum = 0
        for i in range(4):
            min_corner = min(np.linalg.norm(pts2d - corner[i].reshape(2,1), ord=2, axis=0))
            if min_corner > halfDim[idx1] or min_corner > halfDim[idx2]: # 避免偏离情况
                min_corner = 0
            dist_sum += min_corner
        print(dist_sum)
        dist_sum_list.append(dist_sum)
    return np.argmax(dist_sum_list)

def get_max_num_cluster(pcd, eps=0.1, min_points=10, print_progress=True):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=print_progress))
    max_label = labels.max()
    # print(f"点云分为 {max_label + 1} 类")
    # 统计每个簇中点的数量
    unique_labels, counts = np.unique(labels, return_counts=True)
    # 找到点数最多的簇
    max_cluster_label = unique_labels[np.argmax(counts)]
    max_cluster_indices = np.where(labels == max_cluster_label)[0]
    max_pcd = pcd.select_by_index(max_cluster_indices)
    return max_pcd

def get_adjust_transform(pts, numPoints, halfDim, planeAxis, pcd):
    idx1, idx2 = [i for i in range(3) if i != planeAxis]
    top_plane_points_idx = [i for i in range(numPoints) if pts[i][planeAxis] > 0.9 * halfDim[planeAxis]]
    if len(top_plane_points_idx) < 500:
        top_plane_points_idx = np.argsort(pts[:, planeAxis])[-500:]
    bottom_plane_points_idx = [i for i in range(numPoints) if pts[i][planeAxis] < -1 * 0.9 * halfDim[planeAxis]]
    if len(bottom_plane_points_idx) < 500:
        bottom_plane_points_idx = np.argsort(pts[:, planeAxis])[:500]
    top_pcd = pcd.select_by_index(top_plane_points_idx)
    top_pts = np.asarray(top_pcd.points)
    top_pcd.paint_uniform_color(np.random.rand(3))
    bottom_pcd = pcd.select_by_index(bottom_plane_points_idx)
    bottom_pts = np.asarray(bottom_pcd.points)
    bottom_pcd.paint_uniform_color(np.random.rand(3))
    top_max_1 = np.max(top_pts[:, idx1])
    top_min_1 = np.min(top_pts[:, idx1])
    top_max_2 = np.max(top_pts[:, idx2])
    top_min_2 = np.min(top_pts[:, idx2])
    bottom_max_1 = np.max(bottom_pts[:, idx1])
    bottom_min_1 = np.min(bottom_pts[:, idx1])
    bottom_max_2 = np.max(bottom_pts[:, idx2])
    bottom_min_2 = np.min(bottom_pts[:, idx2])
    top_size_1 = abs(top_max_1 - top_min_1)
    top_size_2 = abs(top_max_2 - top_min_2)
    top_size_ratio = top_size_1 / top_size_2 if top_size_1 < top_size_2 else top_size_2 / top_size_1
    top_ratio_1 = top_size_1 / (2 * halfDim[idx1])
    top_ratio_2 = top_size_2 / (2 * halfDim[idx2])
    bottom_size_1 = abs(bottom_max_1 - bottom_min_1)
    bottom_size_2 = abs(bottom_max_2 - bottom_min_2)
    bottom_size_ratio = bottom_size_1 / bottom_size_2 if bottom_size_1 < bottom_size_2 else bottom_size_2 / bottom_size_1
    bottom_ratio_1 = bottom_size_1 / (2 * halfDim[idx1])
    bottom_ratio_2 = bottom_size_2 / (2 * halfDim[idx2])

    # print("拟合顶面圆")
    top_cluster_pcd = get_max_num_cluster(top_pcd, 0.1, 10, True)
    top_cluster_pts = np.asarray(top_cluster_pcd.points)
    # print("拟合底面圆")
    bottom_cluster_pcd = get_max_num_cluster(bottom_pcd, 0.1, 10, True)
    bottom_cluster_pts = np.asarray(bottom_cluster_pcd.points)
    o3d.visualization.draw_geometries([bottom_cluster_pcd, top_cluster_pcd])
    top_circle, _, _ = fit_circle(top_cluster_pts[:, [idx1, idx2]], 1000, 0.01)
    bottom_circle, _, _ = fit_circle(bottom_cluster_pts[:, [idx1, idx2]], 1000, 0.01)
    if abs(top_size_ratio - 1) < 0.05:
        center_top = np.array([0.5 * (top_max_1 + top_min_1),
                               0.5 * (top_max_2 + top_min_2)])
    else:
        center_top = top_circle[0]
    if abs(bottom_size_ratio - 1) < 0.05:
        center_bottom = np.array([0.5 * (bottom_max_1 + bottom_min_1),
                                  0.5 * (bottom_max_2 + bottom_min_2)])
    else:
        center_bottom = bottom_circle[0]

    center_top = top_circle[0]
    center_bottom = bottom_circle[0]
    top_center = [0,0,0]
    top_center[planeAxis] = halfDim[planeAxis]
    top_center[idx1] = center_top[0]
    top_center[idx2] = center_top[1]
    bottom_center = [0,0,0]
    bottom_center[planeAxis] = -1.0 * halfDim[planeAxis]
    bottom_center[idx1] = center_bottom[0]
    bottom_center[idx2] = center_bottom[1]

    if abs(top_ratio_1/top_ratio_2 - 1) < 0.01 and abs(bottom_ratio_1/bottom_ratio_2 - 1) < 0.01:
        T = SE3.Tx(0)
    else:
        v1 = np.asarray(top_center) - np.asarray(bottom_center)
        v1 = v1 / np.linalg.norm(v1)
        v2, v3 = find_orthogonal_vectors(v1)
        if planeAxis == 0:
            T = SE3.Rt(SO3.TwoVectors(x=v1, y=v2), 0.5 * (np.asarray(top_center) + np.asarray(bottom_center)))
        if planeAxis == 1:
            T = SE3.Rt(SO3.TwoVectors(x=v2, y=v1), 0.5 * (np.asarray(top_center) + np.asarray(bottom_center)))
        if planeAxis == 2:
            T = SE3.Rt(SO3.TwoVectors(x=v2, z=v1), 0.5 * (np.asarray(top_center) + np.asarray(bottom_center)))

    # print(f"top center: {center_top}")
    # print(f"bottom center: {center_bottom}")
    # print(f"top norm: {np.linalg.norm(center_top)}")
    # print(f"bottom norm: {np.linalg.norm(center_bottom)}")
    # print(f"top size ratio: {top_size_ratio}")
    # print(f"bottom size ratio: {bottom_size_ratio}")
    # print(f"top ratio: {top_ratio_1}， {top_ratio_2}")
    # print(f"bottom ratio: {bottom_ratio_1}， {bottom_ratio_2}")

    top_center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01, resolution=100)
    top_center_sphere.paint_uniform_color([1.0,1.0,0.0]) # Yellow
    top_center_sphere.translate(top_center)
    bottom_center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01, resolution=100)
    bottom_center_sphere.paint_uniform_color([1.0,0.0,1.0])
    bottom_center_sphere.translate(bottom_center)
    # coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    # TobbOri = SE3.Rt(obbOri.R, obbOri.center)
    ceter = np.asarray(top_center)
    ceter[planeAxis] = 0
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    origin_trans = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    origin_trans.transform(T)
    # o3d.visualization.draw_geometries([aabb, top_pcd, bottom_pcd, top_center_sphere, bottom_center_sphere, origin, origin_trans])
    top_circle_r = max(np.linalg.norm(top_pts[:, [idx1, idx2]] - np.array(top_circle[0]), ord=2, axis=1))
    bottom_circle_r = max(np.linalg.norm(bottom_pts[:, [idx1, idx2]] - np.array(bottom_circle[0]), ord=2, axis=1))
    # print(top_circle_r)
    # print(bottom_circle_r)
    # print(max(top_size_1, top_size_2)/2)
    # print(max(bottom_size_1, bottom_size_2)/2)

    if top_circle_r > max(top_size_1, top_size_2)/2 and top_circle[1] < max(halfDim)*1.2:
        top_r = top_circle_r
    else:
        top_r = max(top_size_1, top_size_2)/2
    if bottom_circle_r > max(bottom_size_1, bottom_size_2)/2 and bottom_circle[1] < max(halfDim)*1.2:
        bottom_r = bottom_circle_r
    else:
        bottom_r = max(bottom_size_1, bottom_size_2)/2
    return T, top_r, bottom_r

def fit_frustum_cone_obb(pcd):
    pcd_fit = o3d.geometry.PointCloud()
    pcd_fit.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    pts = np.asarray(pcd_fit.points)
    obbOri = pcd_fit.get_minimal_oriented_bounding_box()
    TobbOri = SE3.Rt(obbOri.R, obbOri.center)
    pcd_fit.transform(TobbOri.inv())
    halfDim = [max(pts[:, 0]), max(pts[:, 1]), max(pts[:, 2])]
    planeAxis = get_plane_axis(pcd_fit, halfDim)
    if planeAxis == -1:
        print("判断平面轴方向错误")
        return SE3(), 0, 0
    height = halfDim[planeAxis] * 2
    T, top_r, bottom_r = get_adjust_transform(pts, pts.shape[0], halfDim, planeAxis, pcd_fit)
    # TODO: T is not used
    return top_r, bottom_r, height, TobbOri

def fit_ellipsoid(pcd, t=0.01):
    points, m, centroid = pc_normalize(np.asarray(pcd.points))
    num_points = len(points)
    ellipsoid_model = EllipsoidLeastSquaresModel()
    best_fit, _ = ransac(points, ellipsoid_model, 10, 500, t, 1, inliers_ratio=0.9, debug=False, return_all=True)
    params = ellipsoid_model.get_ellipsoid_params(best_fit)
    if params is not None:
        x0t, y0t, z0t, a, b, c, R = params
    else:
        return None
    center = np.array([x0t, y0t, z0t]) * m + centroid
    T = SE3.Rt(SO3(np.array(R, dtype=np.float64)), center)
    return a*m, b*m, c*m, T

if __name__ == "__main__":
    test = 1 # 0: cube, 1: cone, 2: ellipsoid
    # point_cloud = o3d.io.read_point_cloud("../../../data/outputs/pick.ply")
    point_cloud = o3d.io.read_point_cloud("../../../data/0005/8.ply")
    # point_cloud = o3d.io.read_point_cloud("../../../data/outputs/test.ply")
    point_cloud.estimate_normals()
    # camera = [0,0,800]
    # point_cloud.orient_normals_towards_camera_location(camera)
    if len(np.asarray(point_cloud.points)) > 5000:
        pcd_fps = point_cloud.farthest_point_down_sample(5000)
    else:
        pcd_fps = point_cloud
    cl, ind = pcd_fps.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.0)
    pcd = pcd_fps.select_by_index(ind)
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    if test == 0:
        pass
    elif test == 1:
        r1, r2, height, T = fit_frustum_cone_normal(pcd)
        poses_geo = []
        poses1 = gen_cone_side_pick_poses(height, r1, r2, 10)
        poses2 = gen_cone_center_pick_poses(height, 10)
        poses = poses1 + poses2
        for pose in poses:
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])
            coord.transform(T*pose)
            poses_geo.append(coord)
        fit_cone_points = generate_cone_points(r_bottom=r2, r_top_ratio=r1/r2, height=height, delta=0.0, points_density=0, total_points=5000)
        fit_cone_pcd = o3d.geometry.PointCloud()
        fit_cone_pcd.points = o3d.utility.Vector3dVector(fit_cone_points)
        fit_cone_pcd.paint_uniform_color([0, 0, 1])
        fit_cone_pcd.transform(T)
        o3d.io.write_point_cloud(os.path.join("/root/ros_ws//src/data/0005/fit_8.ply"), fit_cone_pcd)
        coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        coord_frame.transform(T)
        o3d.visualization.draw_geometries([*poses_geo, fit_cone_pcd, point_cloud, coord_frame, coord_frame_origin], point_show_normal=False)
    elif test == 2:
        a, b, c, T = fit_ellipsoid(pcd)
        poses_geo = []
        poses = gen_ellipsoid_center_pick_poses(10, [0, 0, 0])
        poses2 = gen_ellipsoid_side_pick_poses(10, a, b, c, T, pcd)
        for pose in poses:
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])
            coord.transform(T*pose)
            poses_geo.append(coord)
        for pose in poses2:
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])
            coord.transform(pose)
            poses_geo.append(coord)
        fit_ellipsoid_points = generate_ellipsoid_points(a, b, c, total_points=5000)
        fit_ellipsoid_pcd = o3d.geometry.PointCloud()
        fit_ellipsoid_pcd.points = o3d.utility.Vector3dVector(fit_ellipsoid_points)
        fit_ellipsoid_pcd.paint_uniform_color([0, 0, 1])
        fit_ellipsoid_pcd.transform(T)
        coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        coord_frame.transform(T)
        o3d.visualization.draw_geometries([*poses_geo, fit_ellipsoid_pcd, point_cloud, coord_frame, coord_frame_origin], point_show_normal=False)
