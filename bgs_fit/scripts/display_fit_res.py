import os
import open3d as o3d

data_path = "../../../data/0005"

origin_pcd = []
fit_pcd = []
pcd_scene = o3d.io.read_point_cloud(os.path.join(data_path, f"points.ply"))
numPointsPcd = len(pcd_scene.points)
voxelRation = numPointsPcd // 100000 if numPointsPcd > 100000 else 1
pcd_scene = pcd_scene.uniform_down_sample(voxelRation)
pcd_scene.paint_uniform_color([0,1,0])
o3d.visualization.draw_geometries([pcd_scene])

for i in range(10):
    pcd_ori = o3d.io.read_point_cloud(os.path.join(data_path, f"{i}.ply"))
    numPcd_ori = len(pcd_ori.points)
    voxelRationOri = numPcd_ori // 10000 if numPcd_ori > 10000 else 1
    pcd_ori = pcd_ori.uniform_down_sample(voxelRationOri)
    pcd_ori.paint_uniform_color([0,1,0])
    pcd_fit = o3d.io.read_point_cloud(os.path.join(data_path, f"fit_{i}.ply"))
    pcd_fit.paint_uniform_color([0,0,1])
    origin_pcd.append(pcd_ori)
    fit_pcd.append(pcd_fit)
del fit_pcd[6]
o3d.visualization.draw_geometries([*origin_pcd, *fit_pcd])
