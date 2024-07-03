import os
import open3d as o3d
import numpy as np
from spatialmath import SO3, SE3
from sklearn import linear_model
from sklearn.cluster import KMeans
# For ROS2
from .utils import *
from .generatePickPoses import *
# from utils import *
# from generatePickPoses import *

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
    a = aabb.max_bound[0]
    b = aabb.max_bound[1]
    c = aabb.max_bound[2]
    return a, b, c, T

def fit_frustum_cone_by_slice(points, num_layers=10):
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

def fit_frustum_cone_normal(pcd):
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
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
        if cluster_indices_size < 100:
            continue
        cluster_pcd = pcd.select_by_index(cluster_indices)
        cluster_normal = np.mean(normals[cluster_indices], axis=0)
        cluster_normals_list.append(cluster_normal)
        plane_model, plane_inliers = cluster_pcd.segment_plane(distance_threshold=0.001, ransac_n=3, num_iterations=1000)
        plane_normals_list.append(plane_model[:3])
        ratio = len(plane_inliers) / cluster_indices_size
        plane_points_ratio_list.append(ratio)
    if np.max(plane_points_ratio_list) > 0.6:
        print(f"使用平面法向量")
        max_plane_ratio_idx = np.argmax(plane_points_ratio_list)
        cone_normal = plane_normals_list[max_plane_ratio_idx]
    else:
        print(f"使用估计法向量")
        cone_axis_model = ConeAxisLeastSquaresModel()
        cluster_normals_list = np.array(cluster_normals_list)
        best_fit, _ = ransac(cluster_normals_list, cone_axis_model, 3, 100, 0.02, num_points*0.2, inliers_ratio=0.8, debug=False, return_all=True)
        vector, best_angle = best_fit
        cone_normal = vector
    vec_x = np.cross(cone_normal, [0,0,1])
    R = SO3.TwoVectors(x=vec_x, z=cone_normal)
    pcd.rotate(R.inv(), center=[0,0,0])
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = [0,0,1]
    coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    coord_frame_cone = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    coord_frame_cone.rotate(R,center=[0,0,0])
    o3d.visualization.draw_geometries([coord_frame_origin, coord_frame_cone, aabb, pcd], point_show_normal=True)
    """ Fit cone radius  """
    r1, r2, height, center = fit_frustum_cone_by_slice(points, 30)
    return r1, r2, height, SE3(R) * SE3(center)

def fit_ellipsoid(pcd):
    points, m, centroid = pc_normalize(np.asarray(pcd.points))
    num_points = len(points)
    ellipsoid_model = EllipsoidLeastSquaresModel()
    best_fit, _ = ransac(points, ellipsoid_model, 10, 100, 0.02, num_points*0.2, inliers_ratio=0.8, debug=False, return_all=True)
    x0t, y0t, z0t, a, b, c, R = ellipsoid_model.get_ellipsoid_params(best_fit)
    center = np.array([x0t, y0t, z0t]) * m + centroid
    return a*m, b*m, c*m, SE3.Rt(SO3(np.array(R, dtype=np.float64)), center)

if __name__ == "__main__":
    test = 2 # 0: cube, 1: cone, 2: ellipsoid
    point_cloud = o3d.io.read_point_cloud("../../../data/outputs/test2.ply")
    point_cloud.estimate_normals()
    camera = [0,0,800]
    point_cloud.orient_normals_towards_camera_location(camera)
    if len(np.asarray(point_cloud.points)) > 5000:
        pcd = point_cloud.farthest_point_down_sample(5000)
    else:
        pcd = point_cloud
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    if test == 0:
        pass
    elif test == 1:
        r1, r2, height, T = fit_frustum_cone_normal(pcd)
        poses_geo = []
        poses1 = gen_cone_side_pick_poses(height, r1, r2, 100)
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
        coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        coord_frame.transform(T)
        o3d.visualization.draw_geometries([*poses_geo, fit_cone_pcd, point_cloud, coord_frame, coord_frame_origin], point_show_normal=False)
    elif test == 2:
        a, b, c, T = fit_ellipsoid(pcd)
        poses_geo = []
        poses = gen_ellipsoid_center_pick_poses(10, [0, 0, 0])
        for pose in poses:
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])
            coord.transform(T*pose)
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
