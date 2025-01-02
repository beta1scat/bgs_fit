import sys
from PyQt5 import QtWidgets
QtWidgets.QApplication(sys.argv)
import os
import cv2
import json
import torch
import importlib
import numpy as np
import open3d as o3d

from utils import *
from fit_bgspcd_noros import *


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, m

model_path = "../../../data/models"
# base_path = "/root/ros_ws/src/data/saved/scatter_pll"
base_path = "/root/ros_ws/src/data/sim/scatter_pll"
pcd_path = os.path.join(base_path, "pcd")
fit_pcd_path = os.path.join(base_path, "fit")
class_path = os.path.join(base_path, "classes")

classifier_model_type = "pointnet2_cls_ssg"
num_class = 3
use_normals = True
model = importlib.import_module(classifier_model_type)
classifier = model.get_model(num_class, normal_channel=use_normals)
classifier = classifier.cuda()
classifier_checkpoint_path = os.path.join(model_path,  "best_model_5000.pth")  # 替换为你的模型路径
classifier_checkpoint = torch.load(classifier_checkpoint_path)
classifier.load_state_dict(classifier_checkpoint['model_state_dict'])
classifier.eval()

files = sorted(os.listdir(pcd_path))
for file in files:
    isOK = False
    # if file != "0_4.ply":
    #     continue
    while not isOK:
        print(file)
        fileName, suffix = file.split('.')
        if suffix == "txt":
            continue
        pcd = o3d.io.read_point_cloud(os.path.join(pcd_path, file))
        pcd_normalized = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pointcloud)
        pts_normalized, normalized_scalse = pc_normalize(np.asarray(pcd.points))
        pcd_normalized.points = o3d.utility.Vector3dVector(pts_normalized)
        pcd_normalized.estimate_normals()
        # pcd_normalized.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(100))
        camera = [0,0,800]
        pcd_normalized.orient_normals_towards_camera_location(camera)
        o3d.visualization.draw_geometries([pcd_normalized], point_show_normal=True)
        # o3d.io.write_point_cloud("../../../data/outputs/"+"test.ply", pcd)
        if len(np.asarray(pcd_normalized.points)) > 5000:
            pcd_normalized = pcd_normalized.farthest_point_down_sample(5000)
        # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        # pcd = o3d.io.read_point_cloud("/home/niu/Downloads/BGSPCD/ellipsoid/ellipsoid_0006.ply")
        points = torch.from_numpy(np.asarray(pcd_normalized.points))
        normals = torch.from_numpy(np.asarray(pcd_normalized.normals))
        ptsWithN = torch.cat((points, normals), dim=1)
        ptsWithNT = torch.unsqueeze(ptsWithN.permute(1, 0), 0)
        ptsWithNT = ptsWithNT.cuda()
        # print(pointsNp)
        print(points.shape)
        print(normals.shape)
        print(ptsWithN.shape)
        print(ptsWithNT.shape)
        print(ptsWithNT)
        print("========")
        vote_num = 1
        vote_pool = torch.zeros(1, num_class).cuda()
        pred, _ = classifier(ptsWithNT.float())
        vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]
        print(pred)
        print(pred_choice)
        realClass = input("Real class is: ")
        # np.savetxt(os.path.join(class_path, f"{fileName}.txt"), np.array([pred_choice.item(), int(realClass)]), fmt="%.0e")
        if len(pcd.points) > 5000:
            pcd_fps = pcd.farthest_point_down_sample(5000)
            cl, ind = pcd_fps.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
            pcd_fit = pcd_fps.select_by_index(ind)
        else:
            pcd_fit = pcd
        if realClass == '0':
            a,b,c,T_cube = fit_cuboid_obb(pcd_fit)
            coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            coord_frame.transform(T_cube)
            obb = pcd_fit.get_minimal_oriented_bounding_box()
            cube = o3d.geometry.TriangleMesh.create_box(width=a*2, height=b*2, depth=c*2)
            cube.translate(-1*np.array([a,b,c]))
            cube.transform(T_cube)
            fit_cube_pcd = cube.sample_points_poisson_disk(5000)
            fit_cube_pcd.paint_uniform_color([0, 0, 1])
            o3d.visualization.draw_geometries([fit_cube_pcd, obb, pcd, coord_frame, coord_frame_origin])
            with open(os.path.join(class_path, f"{fileName}.json"), "w") as outfile:
                json.dump({"pred_type": pred_choice.item(), "input_type": int(realClass), "size": [a,b,c], "T": T_cube.A.tolist()}, outfile)
            o3d.io.write_point_cloud(os.path.join(fit_pcd_path, f"{fileName}.ply"), fit_cube_pcd)
        elif realClass == '1':
            # r1, r2, height, T_cone = fit_frustum_cone_normal(pcd_fit, plane_t=0.001, normal_t=0.02)
            r1, r2, height, T_cone = fit_frustum_cone_normal(pcd_fit, plane_t=0.01, normal_t=0.5)
            fit_cone_points = generate_cone_points(r_bottom=r2, r_top_ratio=r1/r2, height=height, delta=0.0, points_density=0, total_points=5000)
            fit_cone_pcd = o3d.geometry.PointCloud()
            fit_cone_pcd.points = o3d.utility.Vector3dVector(fit_cone_points)
            fit_cone_pcd.paint_uniform_color([0, 0, 1])
            fit_cone_pcd.transform(T_cone)
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            coord_frame.transform(T_cone)
            coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([fit_cone_pcd, pcd, coord_frame, coord_frame_origin], point_show_normal=False)
            with open(os.path.join(class_path, f"{fileName}.json"), "w") as outfile:
                json.dump({"pred_type": pred_choice.item(), "input_type": int(realClass), "size": [r1, r2, height], "T": T_cone.A.tolist()}, outfile)
            o3d.io.write_point_cloud(os.path.join(fit_pcd_path, f"{fileName}.ply"), fit_cone_pcd)
        elif realClass == '2':
            a, b, c, T_ellip = fit_ellipsoid(pcd_fit, 0.01)
            # a, b, c, T_ellip = fit_ellipsoid(pcd_fit, 0.1)
            fit_ellipsoid_points = generate_ellipsoid_points(a, b, c, total_points=5000)
            fit_ellipsoid_pcd = o3d.geometry.PointCloud()
            fit_ellipsoid_pcd.points = o3d.utility.Vector3dVector(fit_ellipsoid_points)
            fit_ellipsoid_pcd.paint_uniform_color([0, 0, 1])
            fit_ellipsoid_pcd.transform(T_ellip)
            coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            coord_frame.transform(T_ellip)
            o3d.visualization.draw_geometries([fit_ellipsoid_pcd, pcd_fit, coord_frame, coord_frame_origin], point_show_normal=False)
            with open(os.path.join(class_path, f"{fileName}.json"), "w") as outfile:
                json.dump({"pred_type": pred_choice.item(), "input_type": int(realClass), "size": [a, b, c], "T": T_ellip.A.tolist()}, outfile)
            o3d.io.write_point_cloud(os.path.join(fit_pcd_path, f"{fileName}.ply"), fit_ellipsoid_pcd)
        elif realClass == '11':
            r1, r2, height, T_cone = fit_frustum_cone_obb(pcd_fit)
            fit_cone_points = generate_cone_points(r_bottom=r2, r_top_ratio=r1/r2, height=height, delta=0.0, points_density=0, total_points=5000)
            fit_cone_pcd = o3d.geometry.PointCloud()
            fit_cone_pcd.points = o3d.utility.Vector3dVector(fit_cone_points)
            fit_cone_pcd.paint_uniform_color([0, 0, 1])
            fit_cone_pcd.transform(T_cone)
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            coord_frame.transform(T_cone)
            coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([fit_cone_pcd, pcd, coord_frame, coord_frame_origin], point_show_normal=False)
            with open(os.path.join(class_path, f"{fileName}.json"), "w") as outfile:
                json.dump({"pred_type": pred_choice.item(), "input_type": int(realClass), "size": [r1, r2, height], "T": T_cone.A.tolist()}, outfile)
            o3d.io.write_point_cloud(os.path.join(fit_pcd_path, f"{fileName}.ply"), fit_cone_pcd)
        else:
            print(f"类型错误")
        isOkStr = input("This file is OK? 'n' for 'not'")
        if isOkStr == "n":
            continue
        isOK = True