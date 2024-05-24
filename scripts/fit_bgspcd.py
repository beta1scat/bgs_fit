import os
import open3d as o3d
import numpy as np
from spatialmath import SO3, SE3
# from utils import *
def fit_cube_obb(point_cloud):
    obbOri = point_cloud.get_minimal_oriented_bounding_box()
    TobbOri = SE3.Rt(obbOri.R, obbOri.center)
    point_cloud.transform(TobbOri.inv())
    coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    aabb = point_cloud.get_axis_aligned_bounding_box()
    a = aabb.max_bound[0]
    b = aabb.max_bound[1]
    c = aabb.max_bound[2]
    return a, b, c, TobbOri