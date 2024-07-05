import os
import open3d as o3d
import numpy as np
from math import pi, cos, sin, atan
from spatialmath import SE3, SO3

def gen_cube_side_pick_poses(size, num_each_side):
    pick_poses = []
    half_size = np.asarray(size) / 2
    for idx in range(3):
        idx1, idx2 = [i for i in range(3) if i != idx]
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

def gen_cone_side_pick_poses(height, top_r, bottom_r, num_each_side):
    pick_poses = []
    step = 0 if num_each_side == 1 else 2 * pi / num_each_side
    alpha = atan(abs(top_r - bottom_r) / height)
    for idx in range(num_each_side):
        t = idx * step
        top_x = top_r * cos(t)
        top_y = top_r * sin(t)
        top_z = height / 2
        bottom_x = bottom_r * cos(t)
        bottom_y = bottom_r * sin(t)
        bottom_z = -1 * height / 2
        top_xL = np.array([[top_x, top_y, 0]])
        top_zL = np.array([[0, 0, -1]])
        bottom_xL = np.array([[bottom_x, bottom_y, 0]])
        bottom_zL = np.array([[0, 0, 1]])
        if top_r > bottom_r:
            top_T = SE3.Rt(SO3.TwoVectors(x=top_xL, z=top_zL)*SO3.Ry(-alpha), [top_x, top_y, top_z])
            bottom_T = SE3.Rt(SO3.TwoVectors(x=bottom_xL, z=bottom_zL)*SO3.Ry(alpha), [bottom_x, bottom_y, bottom_z])
        else:
            top_T = SE3.Rt(SO3.TwoVectors(x=top_xL, z=top_zL)*SO3.Ry(alpha), [top_x, top_y, top_z])
            bottom_T = SE3.Rt(SO3.TwoVectors(x=bottom_xL, z=bottom_zL)*SO3.Ry(-alpha), [bottom_x, bottom_y, bottom_z])
        pick_poses.append(top_T)
        pick_poses.append(bottom_T)
    return pick_poses

def gen_cone_center_pick_poses(height, num_each_position, center=[0,0,0]):
    pick_poses = []
    step = 0 if num_each_position == 1 else 2 * pi / num_each_position
    print(step)
    for idx in range(num_each_position):
        top_x = 0
        top_y = 0
        top_z = height / 2
        bottom_x = 0
        bottom_y = 0
        bottom_z = -1 * height / 2
        center_xyz = center
        t = step * idx
        top_T = SE3.Rt(SO3.Rz(t)*SO3.Rx(pi), [top_x, top_y, top_z])
        bottom_T = SE3.Rt(SO3.Rz(-t), [bottom_x, bottom_y, bottom_z])
        center_T1 = SE3.Rt(SO3.Ry(pi/2)*SO3.Rx(t), center)
        center_T2 = SE3.Rt(SO3.Ry(-pi/2)*SO3.Rx(t), center)
        pick_poses.append(top_T)
        pick_poses.append(bottom_T)
        pick_poses.append(center_T1)
        pick_poses.append(center_T2)
    return pick_poses

def gen_ellipsoid_side_pick_poses(num, a, b, c, aabbInEllipsoid, T):
    pick_poses = []
    step = 0 if num == 1 else 2 * pi / num
    ABC = np.array([a, b, c])
    aabbExtent = aabbInEllipsoid.get_extent()
    aabbCenter = aabbInEllipsoid.get_center()
    diff = np.abs(aabbExtent - 2*ABC)
    mainIdx = np.argmax(diff)
    idx1, idx2 = [i for i in range(3) if i != mainIdx]
    XYZ = np.array([0.0, 0.0, 0.0])
    top = aabbCenter[mainIdx] + aabbExtent[mainIdx] / 2
    bottom = aabbCenter[mainIdx] - aabbExtent[mainIdx] / 2
    XYZ[mainIdx] = top if np.abs(top) < np.abs(bottom) else bottom
    z_dir = np.array([0.0, 0.0, 0.0])
    z_dir[mainIdx] = aabbCenter[mainIdx]
    z_dir = z_dir / np.linalg.norm(z_dir)
    for idx in range(num):
        t = step * idx
        xyz = [0, 0, 0]
        xyz[idx1] = XYZ[idx1] + ABC[idx1]*np.cos(t)
        xyz[idx2] = XYZ[idx2] + ABC[idx2]*np.sin(t)
        xyz[mainIdx] = XYZ[mainIdx]
        x_dir = np.array([0.0, 0.0, 0.0])
        x_dir[idx1] = np.cos(t)
        x_dir[idx2] = np.sin(t)
        x_dir[mainIdx] = 0
        T_pick1 = SE3.Rt(SO3.TwoVectors(x=x_dir, z=z_dir), xyz)
        T_pick2 = SE3.Rt(SO3.TwoVectors(x=-1*x_dir, z=z_dir), xyz)
        pick_poses.append(T*T_pick1)
        pick_poses.append(T*T_pick2)
    return pick_poses

def gen_ellipsoid_center_pick_poses(num_each_direction, center=[0,0,0]):
    pick_poses = []
    step = 0 if num_each_direction == 1 else 2 * pi / num_each_direction
    for idx in range(num_each_direction):
        t = step * idx
        T1 = SE3.Rt(SO3.Rz(t), center)
        T2 = SE3.Rt(SO3.Rx(pi)*SO3.Rz(t), center)
        T3 = SE3.Rt(SO3.Ry(pi/2)*SO3.Rz(t), center)
        T4 = SE3.Rt(SO3.Ry(pi/2)*SO3.Rx(pi)*SO3.Rz(t), center)
        T5 = SE3.Rt(SO3.Rx(pi/2)*SO3.Rz(t), center)
        T6 = SE3.Rt(SO3.Rx(pi/2)*SO3.Rx(pi)*SO3.Rz(t), center)
        pick_poses.append(T1)
        pick_poses.append(T2)
        pick_poses.append(T3)
        pick_poses.append(T4)
        pick_poses.append(T5)
        pick_poses.append(T6)
    return pick_poses

def test_cube_poses():
    cube_size = [1.2, 1.0, 0.6]
    poses1 = gen_cube_side_pick_poses(cube_size, 2)
    poses2 = gen_cube_center_pick_poses()
    poses = poses1 + poses2
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

def test_cone_poses():
    height, top_r, bottom_r = [1.2, 1.0, 1.6]
    poses1 = gen_cone_side_pick_poses(height, top_r, bottom_r, 100)
    poses2 = gen_cone_center_pick_poses(height, 10)
    poses = poses1 + poses2
    scene = []
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    # scene.append(origin)
    for pose in poses:
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        coord.transform(pose)
        scene.append(coord)
    o3d.visualization.draw_geometries(scene)

def test_ellipsoid_poses():
    # poses = gen_ellipsoid_center_pick_poses(10,[0.5,0.5,0.5])
    T = SE3([0.1,0.2,0.3])
    aabb = o3d.geometry.AxisAlignedBoundingBox(np.array([0.05, 0.15, 0.1]), np.array([0.15, 0.25, 0.]))
    poses = gen_ellipsoid_side_pick_poses(10, 0.05, 0.05, 0.1, aabb, T)
    scene = []
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    scene.append(origin)
    for pose in poses:
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        coord.transform(pose)
        scene.append(coord)
    o3d.visualization.draw_geometries(scene)

if __name__ == "__main__":
    model_path = "../../../data/outputs"
    # test_cube_poses()
    # test_cone_poses()
    test_ellipsoid_poses()