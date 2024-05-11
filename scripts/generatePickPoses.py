import os
import open3d as o3d
import numpy as np
from spatialmath import SE3, SO3

def gen_cube_side_pick_poses(size, num_each_side):
    pick_poses = []
    half_size = np.asarray(size) / 2
    for idx in range(3):
        idx1, idx2 = [i for i in range(3) if i != idx]
        print(f"index: {idx1}, {idx2}")
        tL = np.array([[half_size[0], half_size[1], half_size[2]]]*4*num_each_side)
        combinations = [(i, j) for i in range(2) for j in range(2)]
        for idx_t in range(4):
            step = 2*half_size[idx] / (num_each_side+1)
            for idx_n in range(num_each_side):
                tL[idx_n+idx_t*num_each_side, idx] = half_size[idx] - (idx_n+1)*step
                tL[idx_n+idx_t*num_each_side, idx1] *= (-1)**combinations[idx_t][0]
                tL[idx_n+idx_t*num_each_side, idx2] *= (-1)**combinations[idx_t][1]
                xL = np.array([[0,0,0]]*4)
                zL = np.array([[0,0,0]]*2)
                xL[0, idx1] = 1
                xL[1, idx1] = -1
                xL[2, idx2] = 1
                xL[3, idx2] = -1
                zL[0, idx2] = (-1)**(combinations[idx_t][1]+1)
                zL[1, idx1] = (-1)**(combinations[idx_t][0]+1)
                pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[0], z=zL[0]), tL[idx_n+idx_t*num_each_side]))
                pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[1], z=zL[0]), tL[idx_n+idx_t*num_each_side]))
                pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[2], z=zL[1]), tL[idx_n+idx_t*num_each_side]))
                pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[3], z=zL[1]), tL[idx_n+idx_t*num_each_side]))
    return pick_poses

def gen_cube_center_pick_poses(center=[0,0,0]):
    pick_poses = []
    for idx_z in range(2,3):
        idx1, idx2 = [i for i in range(3) if i != idx_z]
        xL = np.array([[0,0,0]]*4)
        zL = np.array([[0,0,0]]*2)
        zL[0, idx_z] = 1
        zL[1, idx_z] = -1
        xL[0, idx1] = 1
        xL[1, idx1] = -1
        xL[2, idx2] = 1
        xL[3, idx2] = -1
        pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[0], z=zL[0]), center))
        pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[1], z=zL[0]), center))
        pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[2], z=zL[0]), center))
        pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[3], z=zL[0]), center))
        pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[0], z=zL[1]), center))
        pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[1], z=zL[1]), center))
        pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[2], z=zL[1]), center))
        pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[3], z=zL[1]), center))
    return pick_poses

model_path = "../../data/outputs"
cube_size = [1.2, 1.0, 0.6]
# poses = gen_cube_side_pick_poses(cube_size, 2)
poses = gen_cube_center_pick_poses()
cube = o3d.geometry.TriangleMesh.create_box(*cube_size)
cube.translate(np.asarray(cube_size) / (-2))
cube_obb = cube.get_oriented_bounding_box()
cube_obb.color = [1,0,0]
scene = []
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
scene.append(cube_obb)
# scene.append(origin)
for pose in poses:
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    coord.transform(pose)
    scene.append(coord)
o3d.visualization.draw_geometries(scene)