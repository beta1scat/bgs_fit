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
        cube_4 = o3d.geometry.TriangleMesh.create_box(width=xDim, height=yDim-thickness*2, depth=xDim)
        cube_4.transform(SE3(-xDim/2, -yDim/2+thickness, height_1+height_2+zDim-xDim))
        OBB = cube_4.get_minimal_oriented_bounding_box()
        OBB.color = [0,1,0]
        # cube_5 = o3d.geometry.TriangleMesh.create_box(width=xDim, height=thickness, depth=zDim - thickness)
        # cube_5.transform(SE3(-xDim/2, -thickness/2, height_1+height_2+thickness))
        self.toolMesh.append(cylinder_1)
        self.toolMesh.append(cylinder_2)
        self.toolMesh.append(cube_1)
        self.toolMesh.append(cube_2)
        self.toolMesh.append(cube_3)
        # self.toolMesh.append(cube_4) # for test figner contact obb
        # self.toolMesh.append(cube_5) # for test figner position
        self.contactOBB = OBB

        self.tcp_in_flange = SE3(0, 0, height_1 + height_2 + zDim - xDim / 2)
    def transform(self, T):
        self.contactOBB.translate(T.t)
        self.contactOBB.rotate(T.R, T.t)
        for mesh in self.toolMesh:
            mesh.transform(T)
    def get_meshes(self):
        return self.toolMesh
    def get_contact_obb(self):
        return self.contactOBB
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
poses = []
posesMeshOri = []
cube_pcd = generateCubePcd(*cube_size, 50000)
T_cube = SE3([0,0,0])
cube_pcd.transform(T_cube)
cube_poses_side = gen_cube_side_pick_poses(cube_size, 2)
# cube_poses_center = gen_cube_center_pick_poses()
cube_poses = cube_poses_side
for pose in cube_poses:
    poses.append(T_cube*pose)

for pose in poses:
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    coord.transform(pose)
    posesMeshOri.append(coord)

coord_ori = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
o3d.visualization.draw_geometries([cube_pcd, *posesMeshOri, coord_ori])
posesFiltered = filter_pose_by_axis_diff(poses, axis=2, ref_axis=[0,0,-1], t=np.pi/3, sorted=False)

posesMeshAxisFiltered = []
for pose in posesFiltered:
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    coord.transform(pose)
    posesMeshAxisFiltered.append(coord)

gripper = Gripper()
sphere = o3d.geometry.TriangleMesh.create_sphere(0.3)
sphere.transform(SE3([0.2, 0.8, 0.5]))
sphere.paint_uniform_color([1,0,0])
T = posesFiltered[0]*gripper.get_tcp_in_flange().inv()
gripper.transform(T)
obb = gripper.get_contact_obb()
idxs = obb.get_point_indices_within_bounding_box(cube_pcd.points)
inliersPcd = cube_pcd.select_by_index(idxs)
inliersPcd.paint_uniform_color([1,1,0])
outliersPcd = cube_pcd.select_by_index(idxs, invert=True)
o3d.visualization.draw_geometries([cube_pcd, *posesMeshAxisFiltered, coord_ori, *gripper.get_meshes(), sphere])
o3d.visualization.draw_geometries([inliersPcd, outliersPcd, *posesMeshAxisFiltered, coord_ori, *gripper.get_meshes(), gripper.get_contact_obb()])
gripper.transform(T.inv())



