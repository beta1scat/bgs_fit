
import sys
from PyQt5 import QtWidgets
QtWidgets.QApplication(sys.argv)

import os
import cv2
import torch
import importlib
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
# from fit_bgspcd import *
# from generatePickPoses import *
from math import pi

def mouse_callback(event, x, y, flags, param):
    # img, img_bak, click_pos = param
    if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键点击事件
        # 记录点击的位置
        param[2].append((x, y))
        # 在图像上绘制圆点
        cv2.circle(param[0], (x, y), 5, (0, 0, 255), -1)  # 红色圆点
        cv2.imshow("Image with Clicks", param[0])  # 显示更新后的图像

    if event == cv2.EVENT_RBUTTONDOWN:  # 鼠标左键点击事件
        param[0] = param[1].copy()
        param[2].clear()
        cv2.imshow("Image with Clicks", param[0])  # 显示更新后的图像

def depth_to_pointcloud(depth_image, fx, fy, cx, cy):
    height, width = depth_image.shape
    u, v = np.meshgrid(np.arange(1, width+1), np.arange(1, height+1))
    z = depth_image
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pointcloud = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
    nonzero_indices = np.all(pointcloud != [0, 0, 0], axis=1)
    filteredPCD = pointcloud[nonzero_indices]
    return filteredPCD

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, m


# 载入 SAM 模型
sam_model_type = "vit_h"  # or vit_b, vit_l based on the model you have
sam_checkpoint_path = "../../../data/models/sam_vit_h_4b8939.pth"  # 替换为你的模型路径
sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
sam = sam.to(device = "cuda")
# sam.to(device = "cpu")
predictor = SamPredictor(sam)

classifier_model_type = "pointnet2_cls_ssg"
num_class = 3
use_normals = True
model = importlib.import_module(classifier_model_type)
classifier = model.get_model(num_class, normal_channel=use_normals)
classifier = classifier.cuda()
classifier_checkpoint_path = '../../../data/models/best_model_ssg.pth'
classifier_checkpoint = torch.load(classifier_checkpoint_path)
classifier.load_state_dict(classifier_checkpoint['model_state_dict'])
classifier.eval()

# 读取图片 & 深度图
# image_path = "../../../data/0003/image.png"  # 替换为你的图片路径
# depth_image_path = '../../../data/0003/depth.tiff'

# base_path = "/root/ros_ws/src/data/saved/scatter_pll"
# base_path = "/root/ros_ws/src/data/sim/scatter_pll"
base_path = "/root/ros_ws/src/data/sim_flat"
# # Sim Camera Intrinsic parameters
fx = 2327.564263511396
fy = 2327.564263511396
cx = 720.5
cy = 540.5
tx = -162.92949844579772
ty = 0
# # Pro-S
# fx = 2422.5631472657747,
# fy = 2422.603833233446,
# cx = 962.2960960737917,
# cy = 631.7893849597299
# RealSense 415
# fx, fy = [895.176, 895.176]
# cx, cy = [630.254, 374.059]
image_path = os.path.join(base_path, "img")
depth_path = os.path.join(base_path, "depth")
seg_path = os.path.join(base_path, "segments")
pcd_path = os.path.join(base_path, "pcd")
masks_path = os.path.join(base_path, "masks")
files = sorted(os.listdir(image_path))
for file in files:
    print(file)
    fileName = file.split('.')[0]
    imageIsOk = False
    idx = 0
    while not imageIsOk:
        image = cv2.imread(os.path.join(image_path, file))
        image_bak = image.copy()
        print(f"image size: {image.shape}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth_image = cv2.imread(os.path.join(depth_path, fileName + ".tiff"), cv2.IMREAD_UNCHANGED)
        # click_positions = []
        image_bak = image.copy()
        # 使用 selectROI 在缩小后的图像上选择 ROI
        cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ROI", 1280, 800)
        roi = cv2.selectROI("ROI", image_bak)
        cv2.destroyWindow("ROI")
        if roi == (0, 0, 0, 0):
            print("No box was drawn.")
            break
        # 获取缩小后选择的 ROI 坐标
        x, y, w, h = roi

        # # 将 ROI 坐标转换回原始图像的坐标
        # x = int(x)
        # y = int(y)
        # w = int(w)
        # h = int(h)
        input_box = np.array([x, y, x + w, y + h])
        # 使用框提示进行分割
        predictor.set_image(image_rgb)
        masks, scores, logits = predictor.predict(
            box=input_box[None, :],  # 需要增加一个维度来匹配输入形状
            multimask_output=False  # 如果为 True，将返回多个可能的分割结果
        )

        print(masks)
        # 选择分割结果，并展示
        mask = masks[0]
        segmented_image = np.zeros_like(image_rgb)
        segmented_image[mask] = image_rgb[mask]

        segmented_depth_image = np.copy(depth_image)
        segmented_depth_image[~mask] = 0

        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(image_rgb)
        # plt.gca().add_patch(plt.Circle(click_positions[0], 5,
        #                                 edgecolor='red', facecolor='none', lw=2))
        plt.title("Original Image with Box")

        plt.subplot(1, 3, 2)
        plt.imshow(segmented_image)
        plt.title("Segmented Image")

        plt.subplot(1, 3, 3)
        plt.imshow(segmented_depth_image)
        plt.title("Segmented depth Image")
        plt.show()

        pointcloud = depth_to_pointcloud(segmented_depth_image, fx, fy, cx, cy)
        print(f"点数：{len(pointcloud)}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        pcd.estimate_normals()
        camera = [0,0,800]
        pcd.orient_normals_towards_camera_location(camera)
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        isSave = input("Is save result? 'n' for 'not'")
        if isSave == "n":
            continue
        o3d.io.write_point_cloud(os.path.join(pcd_path, f"{fileName}_{idx}.ply"), pcd)
        print(os.path.join(masks_path, f"{fileName}_{idx}.txt"))
        np.savetxt(os.path.join(masks_path, f"{fileName}_{idx}.txt"), masks[0], fmt='%s')
        # print(type(masks[0]))
        # print(masks[0].shape)
        # print(masks[0])
        cv2.imwrite(os.path.join(seg_path, f"{fileName}_{idx}_c.png"), cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(seg_path, f"{fileName}_{idx}_d.png"), np.uint8(cv2.normalize(segmented_depth_image, None, 0, 255, cv2.NORM_MINMAX)))
        idx = idx + 1





    # o3d.io.write_point_cloud("../../../data/outputs/"+"pick.ply", pcd)
# pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(100))
# pcd_normalized = o3d.geometry.PointCloud()
# # pcd.points = o3d.utility.Vector3dVector(pointcloud)
# pts_normalized, normalized_scalse = pc_normalize(pointcloud)
# pcd_normalized.points = o3d.utility.Vector3dVector(pts_normalized)
# pcd_normalized.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(100))
# camera = [0,0,800]
# pcd_normalized.orient_normals_towards_camera_location(camera)
# # o3d.io.write_point_cloud("../../../data/outputs/"+"test.ply", pcd)
# if len(np.asarray(pcd_normalized.points)) > 2000:
#     pcd_normalized = pcd_normalized.farthest_point_down_sample(2000)
# # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
# # o3d.io.write_point_cloud("../../../data/outputs/"+"pick.ply", pcd)

# # pcd = o3d.io.read_point_cloud("/home/niu/Downloads/BGSPCD/ellipsoid/ellipsoid_0006.ply")
# print(pointcloud.shape)
# points = torch.from_numpy(np.asarray(pcd_normalized.points))
# normals = torch.from_numpy(np.asarray(pcd_normalized.normals))
# ptsWithN = torch.cat((points, normals), dim=1)
# ptsWithNT = torch.unsqueeze(ptsWithN.permute(1, 0), 0)
# ptsWithNT = ptsWithNT.cuda()
# # print(pointsNp)
# print(points.shape)
# print(normals.shape)
# print(ptsWithN.shape)
# print(ptsWithNT.shape)
# print(ptsWithNT)
# print("========")
# vote_num = 1
# vote_pool = torch.zeros(1, num_class).cuda()
# pred, _ = classifier(ptsWithNT.float())
# vote_pool += pred
# pred = vote_pool / vote_num
# pred_choice = pred.data.max(1)[1]
# print(pred)
# print(pred_choice)

# pcd = pcd.farthest_point_down_sample(5000)
# r1, r2, height, T = fit_frustum_cone_ransac(pcd)
# fit_cone_points = generate_cone_points(r_bottom=r2, r_top_ratio=r1/r2, height=height, delta=0.0, points_density=0, total_points=5000)
# fit_cone_pcd = o3d.geometry.PointCloud()
# fit_cone_pcd.points = o3d.utility.Vector3dVector(fit_cone_points)
# fit_cone_pcd.paint_uniform_color([0, 0, 1])
# fit_cone_pcd.transform(T)
# coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
# coord_frame.transform(T)
# o3d.visualization.draw_geometries([fit_cone_pcd, pcd, coord_frame], point_show_normal=False)
# poses = gen_cone_side_pick_poses(height, r1, r2, 100)
# poses_file_path = "../../../poses.txt"
# with open(poses_file_path, 'w+') as file:
#     for matrix in poses:
#         mat = T * matrix * SE3.Rz(pi/2) # For gripper
#         cos_theta = np.dot(mat.a, np.array([0,0,1]))
#         angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
#         if angle_rad < pi / 4:
#             for row in mat.A:
#                 for element in row:
#                     file.write(str(element))
#                     file.write(" ")
#             file.write('\n')




# a,b,c,T_cube = fit_cuboid_obb(pcd)
# print(a)
# print(b)
# print(c)
# print(T_cube)
# poses = gen_cube_side_pick_poses([a*2,b*2,c*2], 1)
# poses_file_path = "../../../poses.txt"
# with open(poses_file_path, 'w+') as file:
#     for matrix in poses:
#         mat = T_cube * matrix * SE3.Rz(pi/2) # For gripper
#         cos_theta = np.dot(mat.a, np.array([0,0,1]))
#         angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
#         if angle_rad < pi / 4:
#             for row in mat.A:
#                 for element in row:
#                     file.write(str(element))
#                     file.write(" ")
#             file.write('\n')
