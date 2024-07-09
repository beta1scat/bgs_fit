
import rclpy
from rclpy.node import Node
from tf_msgs.srv import StringBool
import json

import os
import cv2
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from segment_anything import SamPredictor, sam_model_registry
from .scripts.utils import *
from .scripts.fit_bgspcd import *
from .scripts.generatePickPoses import *
from .scripts import pointnet2_cls_ssg as pointnet2_cls_ssg

# For X error of failed request badwindow
import matplotlib
matplotlib.use('agg')

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
        plt.close()

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
        pcd.estimate_normals()
        pcd_normalized = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pointcloud)
        pts_normalized, normalized_scalse = pc_normalize(pointcloud)
        pcd_normalized.points = o3d.utility.Vector3dVector(pts_normalized)
        pcd_normalized.estimate_normals()
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
        pcd_fps = pcd.farthest_point_down_sample(5000)
        cl, ind = pcd_fps.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd_fit = pcd_fps.select_by_index(ind)
        o3d.visualization.draw_geometries([pcd_fit], point_show_normal=True)
        o3d.io.write_point_cloud("/root/ros_ws/src/data/outputs/"+"pick.ply", pcd_fit)
        if pred_choice == 0:
            a,b,c,T_cube = fit_cuboid_obb(pcd_fit)
            poses1 = gen_cube_center_pick_poses(a, b, c)
            poses2 = gen_cube_side_pick_poses([a*2,b*2,c*2], 1)
            poses_geo = []
            poses = []
            for pose in poses1:
                coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])
                coord.transform(T_cube*pose)
                poses.append(T_cube*pose)
                poses_geo.append(coord)
            for pose in poses2:
                coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])
                coord.transform(T_cube*pose)
                poses.append(T_cube*pose)
                poses_geo.append(coord)
            obb = pcd_fit.get_minimal_oriented_bounding_box()
            poses_filterd = checkPickPoseFor2FingerGripper(pcd_fit, poses, [0.02, 0.07, 0.02], 10)
            print(f"len(poses): {len(poses)}")
            print(f"len(poses_filterd): {len(poses_filterd)}")
            poses = poses_filterd
            coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            coord_frame.transform(T_cube)
            o3d.visualization.draw_geometries([*poses_geo, obb, pcd_fit, coord_frame, coord_frame_origin])
        elif pred_choice == 1:
            r1, r2, height, T_cone = fit_frustum_cone_normal(pcd_fit)
            poses1 = gen_cone_center_pick_poses(height, 4)
            poses2 = gen_cone_side_pick_poses(height, r1, r2, 4)
            poses_geo = []
            poses = []
            for pose in poses1:
                coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])
                coord.transform(T_cone*pose)
                poses.append(T_cone*pose)
                poses_geo.append(coord)
            for pose in poses2:
                coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])
                coord.transform(T_cone*pose)
                poses.append(T_cone*pose)
                poses_geo.append(coord)
            poses_filterd = checkPickPoseFor2FingerGripper(pcd_fit, poses, [0.02, 0.07, 0.02], 10)
            print(f"len(poses): {len(poses)}")
            print(f"len(poses_filterd): {len(poses_filterd)}")
            poses = poses_filterd
            fit_cone_points = generate_cone_points(r_bottom=r2, r_top_ratio=r1/r2, height=height, delta=0.0, points_density=0, total_points=5000)
            fit_cone_pcd = o3d.geometry.PointCloud()
            fit_cone_pcd.points = o3d.utility.Vector3dVector(fit_cone_points)
            fit_cone_pcd.paint_uniform_color([0, 0, 1])
            fit_cone_pcd.transform(T_cone)
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            coord_frame.transform(T_cone)
            coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([*poses_geo, fit_cone_pcd, pcd_fit, coord_frame, coord_frame_origin], point_show_normal=False)
        elif pred_choice == 2:
            a, b, c, T_ellip = fit_ellipsoid(pcd_fit)
            r1, r2, height, T_cone = fit_frustum_cone_normal(pcd_fit, True)
            poses_geo = []
            pose1 = gen_ellipsoid_center_pick_poses(4, [0, 0, 0])
            pose2 = gen_cone_side_pick_poses(height, r1, r2, 4)
            poses = []
            for pose in pose1:
                coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])
                coord.transform(T_ellip*pose)
                poses.append(SE3(T_ellip.t)*pose)
                poses.append(T_ellip*pose)
                poses_geo.append(coord)
            for pose in pose2:
                coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])
                coord.transform(T_cone*pose)
                poses.append(T_cone*pose)
                poses_geo.append(coord)
            poses_filterd = checkPickPoseFor2FingerGripper(pcd_fit, poses, [0.02, 0.07, 0.02], 10)
            print(f"len(poses): {len(poses)}")
            print(f"len(poses_filterd): {len(poses_filterd)}")
            poses = poses_filterd
            fit_ellipsoid_points = generate_ellipsoid_points(a, b, c, total_points=5000)
            fit_ellipsoid_pcd = o3d.geometry.PointCloud()
            fit_ellipsoid_pcd.points = o3d.utility.Vector3dVector(fit_ellipsoid_points)
            fit_ellipsoid_pcd.paint_uniform_color([0, 0, 1])
            fit_ellipsoid_pcd.transform(T_ellip)
            coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            coord_frame.transform(T_ellip)
            o3d.visualization.draw_geometries([*poses_geo, fit_ellipsoid_pcd, pcd_fit, coord_frame, coord_frame_origin], point_show_normal=False)
        else:
            print(f"类型错误")
        poses_geo = []
        for pose in poses:
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])
            coord.transform(pose)
            poses_geo.append(coord)
        o3d.visualization.draw_geometries([*poses_geo, pcd_fit], point_show_normal=False)
        # # Unit is meter
        # tool_coordinate = SE3.Trans([0, 0, 0.2])
        # cam_in_base = SE3.Trans([0.35, -0.3, 1]) * UQ([0.0, 0.707, -0.707, 0.0]).SE3()
        poses_diff_filter = filter_pose_by_axis_diff(poses, 2, [0, 0, 1], np.pi/4)
        print(f"Final len(poses): {len(poses_diff_filter)}")
        poses = poses_diff_filter
        poses_geo = []
        for pose in poses:
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])
            coord.transform(pose)
            poses_geo.append(coord)
        o3d.visualization.draw_geometries([*poses_geo, pcd_fit], point_show_normal=False)
        poses_path = os.path.join(base_path, "poses.txt")
        poses_AA = []
        for pose in poses:
            # pose = pose * SE3.Rt(SO3.Rz(np.pi/2), [0, 0, 0.01])
            pose = pose * SE3([0, 0, 0.01])
            angvec = pose.angvec()[0] * pose.angvec()[1]
            poses_AA.append([*(pose.t.tolist()), pose.UnitQuaternion().s, *(pose.UnitQuaternion().v)])
        with open(poses_path, 'w+') as f:
            json.dump(poses_AA, f)
        response.b = True
        return response



def main():
    rclpy.init()

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()