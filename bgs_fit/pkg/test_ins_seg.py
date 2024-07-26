import os
import cv2
import json
import random
import matplotlib
matplotlib.use('TkAgg') # Fix Qt conflict when use cv2 and matplotlib
import numpy as np
import matplotlib.pyplot as plt
from instance_segmentation import *
from pycocotools import mask as coco_mask

colors = plt.get_cmap("tab20")

def test_sam_by_roi(ins_seg, image_bgr, image, depth_image, use_coco_rle=False):
    """ Select ROI by cursor """
    cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI", 800, 800)
    cv2.imshow("ROI", image_bgr)
    x, y, w, h = cv2.selectROI("ROI", image_bgr)
    print(f"ROI: {[x, y, w, h]}")
    # cropped_image = image_bgr[int(y):int(y + h), int(x):int(x + w)]
    # cv2.imshow("ROI", cropped_image)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

    input_box = np.array([x, y, x + w, y + h]) # xyxy format

    mask = ins_seg.segment_by_roi(image, input_box)
    if use_coco_rle:
        mask = coco_mask.decode(mask).astype(bool) # coco mask decode results is 'int' type, transfrom to bool for numpy
    segmented_image = np.zeros_like(image)
    segmented_image[mask] = image[mask]

    segmented_depth_image = np.copy(depth_image)
    segmented_depth_image[~mask] = 0

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.gca().add_patch(plt.Rectangle((input_box[0], input_box[1]), input_box[2] - input_box[0], input_box[3] - input_box[1],
                                       edgecolor='red', facecolor='none', lw=2))
    plt.title("Original Image with Box")

    plt.subplot(1, 3, 2)
    plt.imshow(segmented_image)
    plt.title("Segmented Image")

    plt.subplot(1, 3, 3)
    plt.imshow(segmented_depth_image)
    plt.title("Segmented depth Image")
    plt.show()

def test_sam_all(ins_seg, image, depth_image, mask_save_path, use_coco_rle=False):
    masks = ins_seg.segment_all(image)
    with open(mask_save_path, 'w') as file:
        json.dump(masks, file, indent=4)
    print(masks)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.subplot(1, 3, 2)
    plt.imshow(depth_image)
    plt.title("Original Depth Image")
    plt.subplot(1, 3, 3)
    img = np.zeros((image.shape[0], image.shape[1], 4), int)
    # img = img * 10
    # img[:,:,3] = 255 # Opaque
    # img[:,:,3] = 127 # Translucent
    # img[:,:,3] = 0 # Transparent
    num = len(masks)
    print(f"number fo masks: {num}")
    for idx in range(num):
        if use_coco_rle:
            m = coco_mask.decode(masks[idx]['segmentation']).astype(bool) # coco mask decode results is 'int' type, transfrom to bool for numpy
        else:
            m = masks[idx]['segmentation']
        img[m] = np.array([int(x * 255) for x in colors(idx % 20)], int)
        bbox = masks[idx]['bbox']
        plt.gca().add_patch(
            matplotlib.patches.Rectangle(
                (bbox[0], bbox[1]),   # (x,y)
                bbox[2],          # width
                bbox[3],          # height
                facecolor='none', edgecolor=colors(idx % 20), linewidth=2
            )
        )
        plt.gca().text(bbox[0] + (bbox[2] / 2) - 30*len(str(idx)), bbox[1] + (bbox[3] / 2), f"{idx}", size = 10)
        plt.gca().text(bbox[0] + (bbox[2] / 2), bbox[1] + (bbox[3] / 2) + 20, f"{masks[idx]['stability_score']:.2f}", size = 10)
    plt.imshow(img)
    plt.title("Masks")
    plt.show()

ins_seg = InstanceSegmentation("vit_h", "../../../data/models/sam_vit_h_4b8939.pth", True, SAMParameters(
    points_per_side=6,
    points_per_batch = 64,
    pred_iou_thresh=0.95,
    stability_score_thresh=0.98,
    stability_score_offset = 1.0,
    crop_n_layers=0,
    box_nms_thresh=0.7,
    crop_nms_thresh=0.7,
    crop_overlap_ratio=512/1500,
    crop_n_points_downscale_factor=1,
    point_grids = None,
    min_mask_region_area=5000,
    output_mode = "coco_rle")
)

data_path = "../../../data/0005"
model_path = "../../../data/models"

image_bgr = cv2.imread(os.path.join(data_path, "image.png")) # For opencv, size: height, width
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)           # For others
depth_image = cv2.imread(os.path.join(data_path, "depth.tiff"), cv2.IMREAD_UNCHANGED)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(depth_image)
plt.title("Original Depth Image")
plt.show()

# test_sam_by_roi(ins_seg, image_bgr, image, depth_image, True)
test_sam_all(ins_seg, image, depth_image, os.path.join(data_path, "mask.json"), True)