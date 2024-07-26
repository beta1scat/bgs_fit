import os
import cv2
import json
import matplotlib
matplotlib.use('TkAgg') # Fix Qt conflict when use cv2 and matplotlib
import matplotlib.pyplot as plt
from pycocotools import mask as coco_mask
from shape_fitter import *
from pick_poses_generator import *

def generate_cone_points(r_bottom=10, r_top_ratio=0.5, height=20, delta=0.1, points_density=1, total_points=10000):
    assert r_bottom > 0, "cone r_top should > 0"
    assert height > 0, "cone height should > 0"
    assert points_density >= 0, "number of points density should >= 0"
    assert total_points > 0, "number of points should > 0"
    r_top = r_bottom * r_top_ratio
    half_height = height / 2
    points = []
    area1 = np.pi * r_top * r_top
    area2 = np.pi * r_bottom * r_bottom
    area3 = np.pi * (r_top + r_bottom) * np.sqrt((r_bottom - r_top)**2 + height**2)
    total_area = area1 + area2 + area3
    l = np.sqrt((r_bottom - r_top)**2 + height**2)
    num_points_l = 0
    if points_density != 0:
        num_points_l = int(np.pi * (r_top + r_bottom) * l * points_density)
    else:
        num_points_l = int(total_points * (area3 / total_area))
    for _ in range(num_points_l):
        ratio = np.random.uniform(-1, 1)
        ratio_0_1 = (ratio + 1) / 2
        z = ratio * half_height
        r = ratio_0_1 * (r_top - r_bottom) + r_bottom
        phi = 2 * np.pi * np.random.rand()
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        points.append([x + np.random.uniform(-1, 1) * delta, y + np.random.uniform(-1, 1) * delta, z + np.random.uniform(-1, 1) * delta])
    return np.array(points)

def generate_ellipsoid_points(a=10, b=10, c=10, total_points=10000):
    # 生成椭球体的点云
    # 椭球的参数方程：x = a*sin(t)*cos(p), y = b*sin(t)*sin(p), z = c*cos(t)
    # 椭球体面积估算方式：S = 4π(abc)^(2/3)
    theta = np.pi * np.random.rand(total_points)
    phi = 2 * np.pi * np.random.rand(total_points)
    x = a * np.sin(theta) * np.cos(phi)
    y = b * np.sin(theta) * np.sin(phi)
    z = c * np.cos(theta)
    points = np.column_stack((x, y, z))
    return points

colors = plt.get_cmap("tab20")

data_path = "../../../data/0005"
model_path = "../../../data/models"

fx = 2327.564263511396
fy = 2327.564263511396
cx = 720.5
cy = 540.5
tx = -162.92949844579772
ty = 0
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(1440, 1080, fx, fy, cx, cy)
camera_extrinsic = np.eye(4)

depth_image = cv2.imread(os.path.join(data_path, "depth.tiff"), cv2.IMREAD_UNCHANGED)

shape_fitter = ShapeFitter(os.path.join(model_path, "best_model_ssg.pth"), 3, True)
pose_gen = PickPosesGenerator()
with open(os.path.join(data_path, "mask.json")) as json_file:
    masks = json.load(json_file)

for mask in masks:
    print(mask)
    m = coco_mask.decode(mask['segmentation']).astype(bool)
    segmented_depth_image = np.copy(depth_image)
    segmented_depth_image[~m] = 0
    pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(segmented_depth_image), camera_intrinsic)
    pcd.estimate_normals()
    camera = [0,0, -800]
    pcd.orient_normals_towards_camera_location(camera)
    pcd_fps = pcd.farthest_point_down_sample(5000)
    cl, ind = pcd_fps.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
    pcd_fit = pcd_fps.select_by_index(ind)
    pcd_clas = pcd_fit.farthest_point_down_sample(2000)
    pcd_clas.points = o3d.utility.Vector3dVector(pc_normalize(pcd_clas.points)[0])
    # pcd_clas.translate([0,1,0])
    o3d.visualization.draw_geometries([pcd_clas], point_show_normal=True)
    print(f"number of fit points: {len(pcd_fit.points)}")
    print(f"number of classification points: {len(pcd_clas.points)}")
    type = shape_fitter.get_pcd_category(pcd_clas)
    print(type)
    if type == 0:
        a, b, c, T_cube = shape_fitter.fit_pcd_by_cuboid(pcd_fit)
        coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        coord_frame.transform(T_cube)
        obb = pcd_fit.get_minimal_oriented_bounding_box()
        cube = o3d.geometry.TriangleMesh.create_box(width=a*2, height=b*2, depth=c*2)
        cube.translate(-1*np.array([a,b,c]))
        cube.transform(T_cube)
        fit_pcd = cube.sample_points_poisson_disk(5000)
        fit_pcd.paint_uniform_color([0, 0, 1])
        poses = pose_gen.generate_cuboid_pick_poses(np.array([a, b, c]) * 2, 2, T_cube)
        poses_coords = []
        for pose in poses:
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])
            coord.transform(pose)
            poses_coords.append(coord)
        o3d.visualization.draw_geometries([fit_pcd, obb, pcd_fit, coord_frame, coord_frame_origin, *poses_coords])
        # o3d.io.write_point_cloud(os.path.join(data_path, f"fit_{idx}.ply"), fit_pcd)
    elif type == 1:
        r1, r2, height, T_cone = shape_fitter.fit_pcd_by_frustum(pcd_fit)
        fit_cone_points = generate_cone_points(r_bottom=r2, r_top_ratio=r1/r2, height=height, delta=0.0, points_density=0, total_points=5000)
        fit_cone_pcd = o3d.geometry.PointCloud()
        fit_cone_pcd.points = o3d.utility.Vector3dVector(fit_cone_points)
        fit_cone_pcd.paint_uniform_color([0, 0, 1])
        fit_cone_pcd.transform(T_cone)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        coord_frame.transform(T_cone)
        coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        poses = pose_gen.generate_frustum_pick_poses(r1, r2, height, 8, 8, T_cone)
        poses_coords = []
        for pose in poses:
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])
            coord.transform(pose)
            poses_coords.append(coord)
        o3d.visualization.draw_geometries([fit_cone_pcd, pcd_fit, coord_frame, coord_frame_origin, *poses_coords], point_show_normal=False)
        # o3d.io.write_point_cloud(os.path.join(data_path, f"fit_{idx}.ply"), fit_cone_pcd)
    elif type == 2:
        a, b, c, T_ellip = shape_fitter.fit_pcd_by_ellipsoid(pcd_fit)
        fit_ellipsoid_points = generate_ellipsoid_points(a, b, c, total_points=5000)
        fit_ellipsoid_pcd = o3d.geometry.PointCloud()
        fit_ellipsoid_pcd.points = o3d.utility.Vector3dVector(fit_ellipsoid_points)
        fit_ellipsoid_pcd.paint_uniform_color([0, 0, 1])
        fit_ellipsoid_pcd.transform(T_ellip)
        coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        coord_frame.transform(T_ellip)
        poses_coords = []
        poses = pose_gen.generate_ellipsoid_pick_poses(8, T_ellip)
        for pose in poses:
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])
            coord.transform(pose)
            poses_coords.append(coord)
        o3d.visualization.draw_geometries([fit_ellipsoid_pcd, pcd_fit, coord_frame, coord_frame_origin, *poses_coords], point_show_normal=False)
        # o3d.io.write_point_cloud(os.path.join(data_path, f"fit_{idx}.ply"), fit_ellipsoid_pcd)
    else:
        print(f"类型错误")

