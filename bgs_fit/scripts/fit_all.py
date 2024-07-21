import json
import os
import cv2
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from utils import *
from fit_bgspcd_noros import *

# For X error of failed request badwindow
import matplotlib
matplotlib.use('agg')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, m

data_path = "../../../data/0005"
preds = np.loadtxt(os.path.join(data_path, 'class.txt'), dtype=int)
print(preds)
# exit()
num_class = len(preds)
print(f"total num: {num_class}")
for idx in range(num_class):
    pcd = o3d.io.read_point_cloud(os.path.join(data_path, f"{idx}.ply"))
    pcd_fps = pcd.farthest_point_down_sample(5000)
    cl, ind = pcd_fps.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
    pcd_fit = pcd_fps.select_by_index(ind)
    o3d.visualization.draw_geometries([pcd_fit], point_show_normal=True)
    pred_choice = preds[idx]
    print(f"pred_choice: {pred_choice}")
    if pred_choice == 0:
        a,b,c,T_cube = fit_cuboid_obb(pcd_fit)
        coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        coord_frame.transform(T_cube)
        obb = pcd_fit.get_minimal_oriented_bounding_box()
        cube = o3d.geometry.TriangleMesh.create_box(width=a*2, height=b*2, depth=c*2)
        cube.translate(-1*np.array([a,b,c]))
        cube.transform(T_cube)
        fit_pcd = cube.sample_points_poisson_disk(5000)
        fit_pcd.paint_uniform_color([0, 0, 1])
        o3d.visualization.draw_geometries([fit_pcd, obb, pcd_fit, coord_frame, coord_frame_origin])
        o3d.io.write_point_cloud(os.path.join(data_path, f"fit_{idx}.ply"), fit_pcd)
    elif pred_choice == 1:
        r1, r2, height, T_cone = fit_frustum_cone_normal(pcd_fit)
        fit_cone_points = generate_cone_points(r_bottom=r2, r_top_ratio=r1/r2, height=height, delta=0.0, points_density=0, total_points=5000)
        fit_cone_pcd = o3d.geometry.PointCloud()
        fit_cone_pcd.points = o3d.utility.Vector3dVector(fit_cone_points)
        fit_cone_pcd.paint_uniform_color([0, 0, 1])
        fit_cone_pcd.transform(T_cone)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        coord_frame.transform(T_cone)
        coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([fit_cone_pcd, pcd_fit, coord_frame, coord_frame_origin], point_show_normal=False)
        o3d.io.write_point_cloud(os.path.join(data_path, f"fit_{idx}.ply"), fit_cone_pcd)
    elif pred_choice == 2:
        a, b, c, T_ellip = fit_ellipsoid(pcd_fit)
        fit_ellipsoid_points = generate_ellipsoid_points(a, b, c, total_points=5000)
        fit_ellipsoid_pcd = o3d.geometry.PointCloud()
        fit_ellipsoid_pcd.points = o3d.utility.Vector3dVector(fit_ellipsoid_points)
        fit_ellipsoid_pcd.paint_uniform_color([0, 0, 1])
        fit_ellipsoid_pcd.transform(T_ellip)
        coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        coord_frame.transform(T_ellip)
        o3d.visualization.draw_geometries([fit_ellipsoid_pcd, pcd_fit, coord_frame, coord_frame_origin], point_show_normal=False)
        o3d.io.write_point_cloud(os.path.join(data_path, f"fit_{idx}.ply"), fit_ellipsoid_pcd)
    else:
        print(f"类型错误")