
import rclpy
from rclpy.node import Node
from tf_msgs.srv import StringBool

import os
import cv2
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from segment_anything import SamPredictor, sam_model_registry
from .scripts.fit_bgspcd import *
from .scripts.generatePickPoses import *
from .scripts import pointnet2_cls_ssg as pointnet2_cls_ssg

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

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(StringBool, 'shape_fit', self.shape_fit)

        # 载入 SAM 模型
        sam_model_type = "vit_h"  # or vit_b, vit_l based on the model you have
        sam_checkpoint_path = "/root/ros_ws/src/data/models/sam_vit_h_4b8939.pth"  # 替换为你的模型路径
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
        sam = sam.to(device = "cuda")
        # sam.to(device = "cpu")
        self.predictor = SamPredictor(sam)

        # 载入分类模型
        self.num_class = 3
        use_normals = True
        classifier = pointnet2_cls_ssg.get_model(self.num_class, normal_channel=use_normals)
        self.classifier = classifier.cuda()
        classifier_checkpoint_path = '/root/ros_ws/src/data/models/best_model_ssg.pth'
        classifier_checkpoint = torch.load(classifier_checkpoint_path)
        self.classifier.load_state_dict(classifier_checkpoint['model_state_dict'])
        self.classifier.eval()
        self.get_logger().info('Initialization completed!')

    def shape_fit(self, request, response):
        self.get_logger().info('Vision results path: %s' % (request.str))
        # 读取图片 & 深度图
        base_path = request.str
        if not os.path.exists(base_path):
            response.b = False
            return response
        image_path = os.path.join(base_path, "image.png")
        depth_image_path = os.path.join(base_path, "depth.tiff")

        image = cv2.imread(image_path)
        print(f"image size: {image.shape}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

        # 缩小图像
        scale_factor = 0.5  # 缩小比例
        small_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)

        # 使用 selectROI 在缩小后的图像上选择 ROI
        roi = cv2.selectROI('Select ROI', small_image)
        cv2.destroyWindow('Select ROI')

        if roi == (0, 0, 0, 0):
            print("No box was drawn.")
            exit()
        # 获取缩小后选择的 ROI 坐标
        x, y, w, h = roi

        # 将 ROI 坐标转换回原始图像的坐标
        x = int(x / scale_factor)
        y = int(y / scale_factor)
        w = int(w / scale_factor)
        h = int(h / scale_factor)
        input_box = np.array([x, y, x + w, y + h])
        print(f"prompt box: {input_box}")
        # 使用框提示进行分割
        self.predictor.set_image(image_rgb)
        input_point = np.array([[x+w/2, y+h/2]])
        input_label = np.array([0])
        masks, scores, logits = self.predictor.predict(
            box=input_box[None, :],  # 需要增加一个维度来匹配输入形状
            multimask_output=False  # 如果为 True，将返回多个可能的分割结果
        )

        # 选择分割结果，并展示
        mask = masks[0]
        segmented_image = np.zeros_like(image_rgb)
        segmented_image[mask] = image_rgb[mask]

        segmented_depth_image = np.copy(depth_image)
        segmented_depth_image[~mask] = 0

        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(image_rgb)
        plt.gca().add_patch(plt.Rectangle((input_box[0], input_box[1]),
                                        input_box[2] - input_box[0], input_box[3] - input_box[1],
                                        edgecolor='red', facecolor='none', lw=2))
        plt.title("Original Image with Box")

        plt.subplot(1, 3, 2)
        plt.imshow(segmented_image)
        plt.title("Segmented Image")

        plt.subplot(1, 3, 3)
        plt.imshow(segmented_depth_image)
        plt.title("Segmented depth Image")
        plt.show()

        # Camera Intrinsic parameters
        fx = 2327.564263511396
        fy = 2327.564263511396
        cx = 720.5
        cy = 540.5
        tx = -162.92949844579772
        ty = 0
        pointcloud = depth_to_pointcloud(segmented_depth_image, fx, fy, cx, cy)
        print(f"点数：{len(pointcloud)}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(100))
        pcd_normalized = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pointcloud)
        pts_normalized, normalized_scalse = pc_normalize(pointcloud)
        pcd_normalized.points = o3d.utility.Vector3dVector(pts_normalized)
        pcd_normalized.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(100))
        camera = [0,0,800]
        pcd_normalized.orient_normals_towards_camera_location(camera)
        o3d.io.write_point_cloud("/root/ros_ws/src/data/outputs/"+"test2.ply", pcd)
        if len(np.asarray(pcd_normalized.points)) > 2000:
            pcd_normalized = pcd_normalized.farthest_point_down_sample(2000)
        # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        # o3d.io.write_point_cloud("/root/ros_ws/src/data/outputs/"+"pick.ply", pcd)

        # pcd = o3d.io.read_point_cloud("/home/niu/Downloads/BGSPCD/ellipsoid/ellipsoid_0006.ply")
        print(pointcloud.shape)
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
        vote_pool = torch.zeros(1, self.num_class).cuda()
        pred, _ = self.classifier(ptsWithNT.float())
        vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]
        print(pred)
        print(pred_choice)
        pcd_fit = pcd.farthest_point_down_sample(5000)
        if pred_choice == 0:
            a,b,c,T_cube = fit_cuboid_obb(pcd)
            poses = gen_cube_side_pick_poses([a*2,b*2,c*2], 1)
        elif pred_choice == 1:
            r1, r2, height, T = fit_frustum_cone_normal(pcd_fit)
            fit_cone_points = generate_cone_points(r_bottom=r2, r_top_ratio=r1/r2, height=height, delta=0.0, points_density=0, total_points=5000)
            fit_cone_pcd = o3d.geometry.PointCloud()
            fit_cone_pcd.points = o3d.utility.Vector3dVector(fit_cone_points)
            fit_cone_pcd.paint_uniform_color([0, 0, 1])
            fit_cone_pcd.transform(T)
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            coord_frame.transform(T)
            coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([fit_cone_pcd, pcd, coord_frame, coord_frame_origin], point_show_normal=False)
            poses = gen_cone_side_pick_poses(height, r1, r2, 100)
        elif pred_choice == 2:
            a, b, c, T = fit_ellipsoid(pcd_fit)
            poses_geo = []
            poses = gen_ellipsoid_center_pick_poses(10, [0, 0, 0])
            for pose in poses:
                coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])
                coord.transform(T*pose)
                poses.append(T*pose)
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
        else:
            print(f"类型错误")

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
        response.b = True
        return response



def main():
    rclpy.init()

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()