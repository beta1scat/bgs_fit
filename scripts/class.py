import os
import sys
import cv2
import json
import torch
import importlib
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

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

def save_pointcloud_to_ply(pointcloud, file_path):
    with open(file_path, 'w') as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write("element vertex {}\n".format(pointcloud.shape[0]))
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")
        for point in pointcloud:
            file.write("{} {} {}\n".format(point[0], point[1], point[2]))

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)

# camera intrisic parameters
# fx = 517.2365030025323
# fy = 517.2365030025323
# cx = 160.5
# cy = 120.5
# tx = -36.20655521017727
# ty = 0

fx = 2327.564263511396
fy = 2327.564263511396
cx = 720.5
cy = 540.5
tx = -162.92949844579772
ty = 0

classDict = {0: "cube", 1: "cone", 2: "ellipsoid"}
maskFile = "../../data/outputs/label.json"
with open(maskFile) as f:
    masksData = json.load(f)

depth_image = cv2.imread('../../data/7/depth.tiff', cv2.IMREAD_UNCHANGED)
print(type(depth_image))
labels = []
boxes_filt = []

modelType = "pointnet2_cls_ssg"
num_class = 3
use_normals = True
model = importlib.import_module(modelType)
classifier = model.get_model(num_class, normal_channel=use_normals)
# classifier = classifier.cuda()
checkpoint = torch.load('../../data/models/best_model_ssg.pth')
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier.eval()
for mask in masksData['mask']:
    if mask['value'] == 0:
        continue
    mask1 = np.array(mask['mask'])[0]
    masked_depth_image = np.copy(depth_image)
    masked_depth_image[~mask1] = 0
    cv2.imshow("1", masked_depth_image)
    cv2.waitKey()

    # Convert depth image to point cloud
    pointcloud = depth_to_pointcloud(masked_depth_image, fx, fy, cx, cy)
    print(f"点数：{len(pointcloud)}")
    pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd.points = o3d.utility.Vector3dVector(pc_normalize(pointcloud))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(100))
    camera = [0,0,-800]
    pcd.orient_normals_towards_camera_location(camera)
    o3d.io.write_point_cloud("../../data/outputs/"+str(mask['value'])+".ply", pcd)
    if len(np.asarray(pcd.points)) > 2000:
        pcd = pcd.farthest_point_down_sample(2000)
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    # pcd = o3d.io.read_point_cloud("/home/niu/Downloads/BGSPCD/ellipsoid/ellipsoid_0006.ply")
    # pointcloud = np.asarray(pcd.points)
    print(pointcloud.shape)
    # pointcloudSampled = farthest_point_sample(pointcloud, 1024)
    # pointsNp = np.array([pc_normalize(pointcloudSampled).T])
    # pointsNp = pc_normalize(pointcloud).T
    points = torch.from_numpy(np.asarray(pcd.points))
    normals = torch.from_numpy(np.asarray(pcd.normals))
    ptsWithN = torch.cat((points, normals), dim=1)
    ptsWithNT = torch.unsqueeze(ptsWithN.permute(1, 0), 0)
    # print(pointsNp)
    print(points.shape)
    print(normals.shape)
    print(ptsWithN.shape)
    print(ptsWithNT.shape)
    print(ptsWithNT)
    print("========")
    vote_num = 1
    vote_pool = torch.zeros(1, num_class)
    pred, _ = classifier(ptsWithNT.float())
    vote_pool += pred
    pred = vote_pool / vote_num
    pred_choice = pred.data.max(1)[1]
    labels.append(classDict[int(pred_choice[0])])
    boxes_filt.append(mask['box'])
    print(pred)
    print(pred_choice)
print(labels)
print(boxes_filt)
image_path = "../../data/7/image.png"
output_dir = "../../data/outputs"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(image)
for box, label in zip(boxes_filt, labels):
    show_box(box, plt.gca(), label)

# plt.title('RAM-tags' + tags + '\n' + 'RAM-tags_chineseing: ' + tags_chinese + '\n')
plt.axis('off')
plt.savefig(
    os.path.join(output_dir, "fit_shapes.jpg"),
    bbox_inches="tight", dpi=300, pad_inches=0.0
)
# Save point cloud to PLY file
# save_pointcloud_to_ply(pointcloud, f'../../data/outputs/{1}.ply')


# for mask in masksData['mask']:
#     if mask['value'] == 0:
#         continue
#     print(mask['mask'])
#     masked_depth_image = np.copy(depth_image)
#     masked_depth_image[~mask['mask']] = 0
    # print(mask['mask'])

