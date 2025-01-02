import os
import open3d as o3d

base_path = "/root/ros_ws/src/data/saved/scatter_pll"
# base_path = "/root/ros_ws/src/data/sim/scatter_pll"
pcd_path = os.path.join(base_path, "pcd")
fit_pcd_path = os.path.join(base_path, "fit")


pcd_files = sorted(os.listdir(pcd_path))
fit_files = sorted(os.listdir(fit_pcd_path))
scene = []

for pf in pcd_files:
    pcd = o3d.io.read_point_cloud(os.path.join(pcd_path, pf))
    pcd.paint_uniform_color([1, 0, 0])
    scene.append(pcd)
for ff in fit_files:
    pcd = o3d.io.read_point_cloud(os.path.join(fit_pcd_path, ff))
    pcd.paint_uniform_color([0, 0, 1])
    scene.append(pcd)
o3d.visualization.draw_geometries(scene, point_show_normal=False)