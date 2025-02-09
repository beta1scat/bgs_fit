import os
import open3d as o3d
from itertools import groupby
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
# 设置随机种子保证每次生成相同的随机颜色（如果需要）
# random.seed(42)
# # 生成 10 个随机颜色（在 matplotlib 中随机选择颜色）
# def generate_random_colors(num_colors):
#     colors = []
#     for _ in range(num_colors):
#         color = np.random.rand(3,)  # 生成随机 RGB 颜色
#         colors.append(color)
#     return colors
# # 生成 10 个随机颜色
# colors = generate_random_colors(10)
colormap = plt.cm.get_cmap('tab10', 10)  # 选择 'tab10' 并限制为 10 种颜色
# base_path = "/root/ros_ws/src/data/saved/scatter_pll"
# base_path = "/root/ros_ws/src/data/sim/scatter_pll"
base_path = "/root/ros_ws/src/data/sim_flat"
pcd_path = os.path.join(base_path, "pcd")
fit_pcd_path = os.path.join(base_path, "fit")
img_path = os.path.join(base_path, "img")
masks_path = os.path.join(base_path, "masks")

pcd_files = sorted(os.listdir(pcd_path))
fit_files = sorted(os.listdir(fit_pcd_path))
masks_files = sorted(os.listdir(masks_path))

pcd_grouped_files = {}
for key, group in groupby(pcd_files, key=lambda x: x.split('_')[0]):
    pcd_grouped_files[key] = list(group)
fit_grouped_files = {}
for key, group in groupby(fit_files, key=lambda x: x.split('_')[0]):
    fit_grouped_files[key] = list(group)
masks_grouped_files = {}
for key, group in groupby(masks_files, key=lambda x: x.split('_')[0]):
    masks_grouped_files[key] = list(group)

# for key, files in pcd_grouped_files.items():
#     print(f"Group {key}: {files}")
#     scene = []

#     for pf in pcd_grouped_files[key]:
#         pcd = o3d.io.read_point_cloud(os.path.join(pcd_path, pf))
#         pcd.paint_uniform_color([1, 0, 0])
#         scene.append(pcd)
#     for ff in fit_grouped_files[key]:
#         pcd = o3d.io.read_point_cloud(os.path.join(fit_pcd_path, ff))
#         pcd.paint_uniform_color([0, 0, 1])
#         scene.append(pcd)
#     o3d.visualization.draw_geometries(scene, point_show_normal=False)

for key, files in masks_grouped_files.items():
    print(f"Group {key}: {files}")
    image = cv2.imread(os.path.join(img_path, f"{key}.png"))
    # 创建一个用于叠加所有 mask 的空白图像
    overlay = image.copy()
    # 将 mask 区域涂上随机颜色
    # colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask = np.ones_like(image, dtype=np.uint8) * 255
    # height, width = image.shape[:2]
    for idx, mask_file in enumerate(files):
        # 读取 mask 数据（True/False）
        mask_path = os.path.join(masks_path, mask_file)
        mask = np.loadtxt(mask_path, dtype=str)  # 读取为字符串类型
        mask = np.vectorize(lambda x: x == 'True')(mask)  # 将 'True' 转为 True，'False' 转为 False
        # # 确保 mask 大小与原始图像一致（如果不一致需要调整）
        # if mask.shape != (height, width):
        #     mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

        # 将 mask 区域填充为随机颜色
        # print(colormap(idx)[:3])
        color = (np.array(colormap(idx)[:3]) * 255).astype(np.uint8)

        colored_mask[mask] = (np.array(color)).astype(np.uint8)  # 转换为 0-255 范围
        # print(colored_mask.shape)

    cv2.imshow('colored_mask', colored_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 将颜色 mask 叠加到原图上（透明度可以调整）
    alpha = 0.8  # 叠加的透明度
    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
        # print(colored_mask)
        # print(image)

    cv2.imshow('Overlayed Image', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()