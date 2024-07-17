import sys
from PyQt5 import QtWidgets
QtWidgets.QApplication(sys.argv)
import os
import cv2
import json
import torch
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import random
import matplotlib.colors as mcolors

# 获取 CSS4 颜色名称列表
css4_colors = list(mcolors.CSS4_COLORS.keys())

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    numAnn= len(sorted_anns)
    for idx in range(numAnn):
        m = sorted_anns[idx]['segmentation']
        c = random.choice(css4_colors)
        color_mask = np.concatenate([mcolors.to_rgb(c), [0.5]])
        img[m] = color_mask
        # point_coords = sorted_anns[idx]['point_coords'][0]
        bbox = sorted_anns[idx]['bbox']
        ax.add_patch(
            patches.Rectangle(
                (bbox[0], bbox[1]),   # (x,y)
                bbox[2],          # width
                bbox[3],          # height
                facecolor='none', edgecolor=c, linewidth=2
            )
        )
        ax.text(bbox[0] + (bbox[2] / 2) - 15, bbox[1] + (bbox[3] / 2) - 15, f"{idx}", size = 15, c=c)
        ax.text(bbox[0] + (bbox[2] / 2), bbox[1] + (bbox[3] / 2), f"{sorted_anns[idx]['stability_score']:.2f}", size = 15)
    ax.imshow(img)

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

data_path = "../../../data/0005"
model_path = "../../../data/models"

# 载入 SAM 模型
sam_model_type = "vit_h"  # or vit_b, vit_l based on the model you have
sam_checkpoint_path = os.path.join(model_path,  "sam_vit_h_4b8939.pth")  # 替换为你的模型路径
sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
sam = sam.to(device = "cuda")
"""
def __init__(
    self,
    model: Sam,
    points_per_side: Optional[int] = 32,
    points_per_batch: int = 64,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    stability_score_offset: float = 1.0,
    box_nms_thresh: float = 0.7,
    crop_n_layers: int = 0,
    crop_nms_thresh: float = 0.7,
    crop_overlap_ratio: float = 512 / 1500,
    crop_n_points_downscale_factor: int = 1,
    point_grids: Optional[List[np.ndarray]] = None,
    min_mask_region_area: int = 0,
    output_mode: str = "binary_mask",
) -> None:

Using a SAM model, generates masks for the entire image.
Generates a grid of point prompts over the image, then filters
low quality and duplicate masks. The default settings are chosen
for SAM with a ViT-H backbone.

Arguments:
  model (Sam): The SAM model to use for mask prediction.
  points_per_side (int or None): The number of points to be sampled
    along one side of the image. The total number of points is
    points_per_side**2. If None, 'point_grids' must provide explicit
    point sampling.
  points_per_batch (int): Sets the number of points run simultaneously
    by the model. Higher numbers may be faster but use more GPU memory.
  pred_iou_thresh (float): A filtering threshold in [0,1], using the
    model's predicted mask quality.
  stability_score_thresh (float): A filtering threshold in [0,1], using
    the stability of the mask under changes to the cutoff used to binarize
    the model's mask predictions.
  stability_score_offset (float): The amount to shift the cutoff when
    calculated the stability score.
  box_nms_thresh (float): The box IoU cutoff used by non-maximal
    suppression to filter duplicate masks.
  crop_n_layers (int): If >0, mask prediction will be run again on
    crops of the image. Sets the number of layers to run, where each
    layer has 2**i_layer number of image crops.
  crop_nms_thresh (float): The box IoU cutoff used by non-maximal
    suppression to filter duplicate masks between different crops.
  crop_overlap_ratio (float): Sets the degree to which crops overlap.
    In the first crop layer, crops will overlap by this fraction of
    the image length. Later layers with more crops scale down this overlap.
  crop_n_points_downscale_factor (int): The number of points-per-side
    sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
  point_grids (list(np.ndarray) or None): A list over explicit grids
    of points used for sampling, normalized to [0,1]. The nth grid in the
    list is used in the nth crop layer. Exclusive with points_per_side.
  min_mask_region_area (int): If >0, postprocessing will be applied
    to remove disconnected regions and holes in masks with area smaller
    than min_mask_region_area. Requires opencv.
  output_mode (str): The form masks are returned in. Can be 'binary_mask',
    'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
    For large resolutions, 'binary_mask' may consume large amounts of
    memory.
"""
while True:
    user_input = input("请输入要调整的参数： ")
    if user_input == 'q':
        break
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=6,
        pred_iou_thresh=0.95,
        stability_score_thresh=0.98,
        crop_n_layers=0,
        box_nms_thresh=0.7,
        crop_nms_thresh=0.7,
        crop_overlap_ratio=512/1500,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=5000,  # Requires open-cv to run post-processing
    )

    # classifier_model_type = "pointnet2_cls_ssg"
    # num_class = 3
    # use_normals = True
    # model = importlib.import_module(classifier_model_type)
    # classifier = model.get_model(num_class, normal_channel=use_normals)
    # classifier = classifier.cuda()
    # classifier_checkpoint_path = os.path.join(model_path,  "best_model_ssg.pth")  # 替换为你的模型路径
    # classifier_checkpoint = torch.load(classifier_checkpoint_path)
    # classifier.load_state_dict(classifier_checkpoint['model_state_dict'])
    # classifier.eval()

    # 读取图片 & 深度图
    image_path = os.path.join(data_path, "image.png")  # 替换为你的图片路径
    depth_image_path = os.path.join(data_path, "depth.tiff")

    image = cv2.imread(image_path)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(f"image size: {image.shape}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow("image_rgb", image_rgb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    masks = mask_generator.generate(image)
    img = image
    for mask in masks:
        m = mask['segmentation']
        color_mask = np.array([0,0,0])
        img[m] = color_mask
    plt.imshow(img)
    masks2 = mask_generator.generate(img)
    total_mask = masks + masks2
    # with open(os.path.join(data_path, "mask.json"), 'w') as file:
    #     json.dump(masks, file, indent=4)
    # print(masks)
    print(f"len(masks): {len(masks)}")
    # print(f"masks[0].keys(): {masks[0].keys()}")
    # print(masks2)
    print(f"len(masks): {len(masks2)}")
    # print(f"masks[0].keys(): {masks2[0].keys()}")
    # print(total_mask)
    print(f"len(masks): {len(total_mask)}")
    print(f"masks[0].keys(): {total_mask[0].keys()}")
    plt.figure(figsize=(20,20))
    plt.imshow(image_rgb)
    show_anns(total_mask)
    plt.axis('off')
    plt.show()



# # 选择分割结果，并展示
# mask = masks[0]
# segmented_image = np.zeros_like(image_rgb)
# segmented_image[mask] = image_rgb[mask]

# segmented_depth_image = np.copy(depth_image)
# segmented_depth_image[~mask] = 0

# plt.figure(figsize=(10, 10))
# plt.subplot(1, 3, 1)
# plt.imshow(image_rgb)
# plt.gca().add_patch(plt.Rectangle((input_box[0], input_box[1]),
#                                   input_box[2] - input_box[0], input_box[3] - input_box[1],
#                                   edgecolor='red', facecolor='none', lw=2))
# plt.title("Original Image with Box")

# plt.subplot(1, 3, 2)
# plt.imshow(segmented_image)
# plt.title("Segmented Image")

# plt.subplot(1, 3, 3)
# plt.imshow(segmented_depth_image)
# plt.title("Segmented depth Image")
# plt.show()

# # Camera Intrinsic parameters
# fx = 2327.564263511396
# fy = 2327.564263511396
# cx = 720.5
# cy = 540.5
# tx = -162.92949844579772
# ty = 0
# pointcloud = depth_to_pointcloud(segmented_depth_image, fx, fy, cx, cy)
# print(f"点数：{len(pointcloud)}")
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pointcloud)
# pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(100))
# pcd_normalized = o3d.geometry.PointCloud()
# # pcd.points = o3d.utility.Vector3dVector(pointcloud)
# pts_normalized, normalized_scalse = pc_normalize(pointcloud)
# pcd_normalized.points = o3d.utility.Vector3dVector(pts_normalized)
# pcd_normalized.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(100))
# camera = [0,0,-800]
# pcd_normalized.orient_normals_towards_camera_location(camera)
# # o3d.io.write_point_cloud("../../data/outputs/"+str(mask['value'])+".ply", pcd)
# if len(np.asarray(pcd_normalized.points)) > 2000:
#     pcd_normalized = pcd_normalized.farthest_point_down_sample(2000)
# # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
# # o3d.io.write_point_cloud("../../data/outputs/"+"pick.ply", pcd)

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

# a,b,c,T_cube = fit_cuboid_obb(pcd)
# print(a)
# print(b)
# print(c)
# print(T_cube)
# poses = gen_cube_side_pick_poses([a*2,b*2,c*2], 1)
# poses_file_path = "../../poses.txt"
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
