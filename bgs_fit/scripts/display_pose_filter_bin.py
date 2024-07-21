from generatePickPoses import *
from utils import *

class Gripper():
    def __init__(self, gripperSize=[0.2, 0.5, 0.5, 0.05], flangeSize = [0.25, 0.05, 0.05, 0.45]):
        # gripperSize: xDim, yDim, zDim, thickness
        # flangeSize: radius_1, height_1, radius_2, height_2
        xDim, yDim, zDim, thickness = gripperSize
        radius_1, height_1, radius_2, height_2 = flangeSize
        self.toolMesh = []
        cylinder_1 = o3d.geometry.TriangleMesh.create_cylinder(radius=radius_1, height=height_1)
        cylinder_1.transform(SE3(0, 0, height_1/2))
        cylinder_1.paint_uniform_color(np.array([52,60,72]) / 255)
        cylinder_2 = o3d.geometry.TriangleMesh.create_cylinder(radius=radius_2, height=height_2)
        cylinder_2.transform(SE3(0, 0, height_1+height_2/2))
        cylinder_2.paint_uniform_color(np.array([67, 101, 90]) / 255)
        cube_1 = o3d.geometry.TriangleMesh.create_box(width=xDim, height=yDim, depth=thickness)
        cube_1.transform(SE3(-xDim/2, -yDim/2, height_1+height_2))
        cube_1.paint_uniform_color(np.array([67, 101, 90]) / 255)
        cube_2 = o3d.geometry.TriangleMesh.create_box(width=xDim, height=thickness, depth=zDim - thickness)
        cube_2.transform(SE3(-xDim/2, -thickness/2 - yDim/2 + thickness/2, height_1+height_2+thickness))
        cube_2.paint_uniform_color(np.array([108,107,115]) / 255)
        cube_3 = o3d.geometry.TriangleMesh.create_box(width=xDim, height=thickness, depth=zDim - thickness)
        cube_3.transform(SE3(-xDim/2, -thickness/2 + yDim/2 - thickness/2, height_1+height_2+thickness))
        cube_3.paint_uniform_color(np.array([108,107,115]) / 255)
        # cube_4 = o3d.geometry.TriangleMesh.create_box(width=xDim, height=thickness, depth=zDim - thickness)
        # cube_4.transform(SE3(-xDim/2, -thickness/2, 0.02+0.5+thickness))
        self.toolMesh.append(cylinder_1)
        self.toolMesh.append(cylinder_2)
        self.toolMesh.append(cube_1)
        self.toolMesh.append(cube_2)
        self.toolMesh.append(cube_3)
        self.tcp_in_flange = SE3(0, 0, height_1 + height_2 + zDim - xDim / 2)
        # toolMesh.append(cube_4) # for test figner position
    def transform(self, T):
        for mesh in self.toolMesh:
            mesh.transform(T)
    def get_meshes(self):
        return self.toolMesh
    def get_tcp_in_flange(self):
        return self.tcp_in_flange


def generateCubePcd(a, b, c, numPts):
    points = generate_cube_points(size=(a, b, c), delta=0.0, points_density=0, total_points=numPts)
    pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc_normalize(points)[0])
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def generateConePcd(r_bottom=10, r_top_ratio=0.5, height=20, total_points=10000):
    points = generate_cone_points(r_bottom, r_top_ratio, height, delta=0.0, points_density=0, total_points=total_points)
    pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc_normalize(points)[0])
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def generateEllipsoidPcd(a=1, b=1, c=1, total_points=10000):
    points = generate_ellipsoid_points(a, b, c, total_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

cube_size = [1, 1, 1]
cone_size = [0.5, 0.8, 1]
ellipsoid_size = [0.5, 0.5, 0.5]
pcds = []
posesMeshOri = []
poses = []
num_pcd = 1
for row in range(num_pcd):
    for col in range(num_pcd):
        block_ori = np.array([row*6, col*6, 0])
        for i in range(3):
            cube_pcd = generateConePcd(*cone_size, 5000)
            T_cube = SE3(block_ori + np.array([i*2 + 2, 2, 0]))
            cube_pcd.transform(T_cube)
            cone_pcd = generateConePcd(*cone_size, 5000)
            T_cone = SE3(block_ori + np.array([i*2 + 2, 4, 0]))
            cone_pcd.transform(T_cone)
            ellipsoid_pcd = generateConePcd(*cone_size, 5000)
            T_ellipsoid = SE3(block_ori + np.array([i*2 + 2, 6, 0]))
            ellipsoid_pcd.transform(T_ellipsoid)
            pcds.append(cube_pcd)
            pcds.append(cone_pcd)
            pcds.append(ellipsoid_pcd)

            # cube_poses_side = gen_cone_side_pick_poses(cone_size[2], cone_size[0]*cone_size[1], cone_size[0], 8)
            cube_poses_center = gen_cone_center_pick_poses(cone_size[2], 10)
            # cube_poses = cube_poses_side + cube_poses_center
            for pose in cube_poses_center:
                poses.append(T_cube*pose)

            # cone_poses_side = gen_cone_side_pick_poses(cone_size[2], cone_size[0]*cone_size[1], cone_size[0], 8)
            cone_poses_center = gen_cone_center_pick_poses(cone_size[2], 10)
            # cone_poses = cone_poses_side + cone_poses_center
            for pose in cone_poses_center:
                poses.append(T_cone*pose)

            # ellipsoid_poses_side = gen_cone_side_pick_poses(cone_size[2], cone_size[0]*cone_size[1], cone_size[0], 8)
            ellipsoid_poses_center = gen_cone_center_pick_poses(cone_size[2], 10)
            # ellipsoid_poses = ellipsoid_poses_side + ellipsoid_poses_center
            for pose in ellipsoid_poses_center:
                poses.append(T_ellipsoid*pose)

for pose in poses:
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    coord.transform(pose)
    posesMeshOri.append(coord)

binMesh = []
bin_color = np.array([67, 101, 90]) / 255
cube_x1 = o3d.geometry.TriangleMesh.create_box(width=num_pcd*6+0.1, height=0.1, depth=1)
cube_x1.transform(SE3([1, 1, -0.5]))
# cube_x1.paint_uniform_color(bin_color)
cube_x2 = o3d.geometry.TriangleMesh.create_box(width=num_pcd*6+0.1, height=0.1, depth=1)
cube_x2.transform(SE3([1, num_pcd*6+1, -0.5]))
# cube_x2.paint_uniform_color(bin_color)
cube_y1 = o3d.geometry.TriangleMesh.create_box(width=0.1, height=num_pcd*6+0.1, depth=1)
cube_y1.transform(SE3([1, 1, -0.5]))
# cube_y1.paint_uniform_color(bin_color)
cube_y2 = o3d.geometry.TriangleMesh.create_box(width=0.1, height=num_pcd*6+0.1, depth=1)
cube_y2.transform(SE3([num_pcd*6+1, 1, -0.5]))
# cube_y2.paint_uniform_color(bin_color)
binMesh.append(cube_x1)
binMesh.append(cube_x2)
binMesh.append(cube_y1)
binMesh.append(cube_y2)

coord_ori = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
o3d.visualization.draw_geometries([*pcds, *posesMeshOri, *binMesh, coord_ori])

# ref_pose = SE3.Rx(np.pi)
coord_ref = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
coord_ref.transform(SE3([4, 4, 0]))
posesFiltered = filter_pose_by_bin_side(poses, bin_size=[7,7,0], ignore_size=[2,2,0], bin_pose=SE3(4, 4 ,0), threshold=np.pi/3)
posesMeshAxisFiltered = []
for pose in posesFiltered:
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    coord.transform(pose)
    posesMeshAxisFiltered.append(coord)
cube = o3d.geometry.TriangleMesh.create_box(width=2, height=2, depth=2)
cube.transform(SE3(np.array([4,4,0]) - np.array([1, 1, 1])))
obb = cube.get_minimal_oriented_bounding_box()
obb.color= [1,0,0]
o3d.visualization.draw_geometries([*pcds, *posesMeshAxisFiltered, *binMesh, coord_ori, coord_ref, obb])



