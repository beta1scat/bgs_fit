import numpy as np
import open3d as o3d
from math import pi, cos, sin, atan
from spatialmath import SE3, SO3

class PickPosesGenerator():
    def generate_cuboid_pick_poses(self, size, num_s, T):
        return self.gen_cuboid_side_pick_poses(size, T, num_s) + self.gen_cuboid_center_pick_poses(size, T)

    def gen_cuboid_side_pick_poses(self, size, T, num_each_side):
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
                    pick_poses.append(T * SE3.Rt(SO3.TwoVectors(x=xL[0], z=zL[0]), tL[idx_n+idx_t*num_each_side]) * SE3.Rz(np.pi/2))
                    pick_poses.append(T * SE3.Rt(SO3.TwoVectors(x=xL[1], z=zL[0]), tL[idx_n+idx_t*num_each_side]) * SE3.Rz(np.pi/2))
                    pick_poses.append(T * SE3.Rt(SO3.TwoVectors(x=xL[2], z=zL[1]), tL[idx_n+idx_t*num_each_side]) * SE3.Rz(np.pi/2))
                    pick_poses.append(T * SE3.Rt(SO3.TwoVectors(x=xL[3], z=zL[1]), tL[idx_n+idx_t*num_each_side]) * SE3.Rz(np.pi/2))
        return pick_poses

    def gen_cuboid_center_pick_poses(self, size, T):
        pick_poses = []
        center = np.zeros(3)
        half_size = np.array(size) / 2
        for idx_z in range(0,3):
            idx1, idx2 = [i for i in range(3) if i != idx_z]
            xL = np.array([[0.0,0.0,0.0]]*4)
            zL = np.array([[0.0,0.0,0.0]]*2)
            center_top = np.array([0.0,0.0,0.0])
            center_bottom = np.array([0.0,0.0,0.0])
            center_top[idx_z] = 1 * half_size[idx_z]
            center_bottom[idx_z] = -1 * half_size[idx_z]
            zL[0, idx_z] = 1
            zL[1, idx_z] = -1
            xL[0, idx1] = 1
            xL[1, idx1] = -1
            xL[2, idx2] = 1
            xL[3, idx2] = -1
            pick_poses.append(T * SE3.Rt(SO3.TwoVectors(x=xL[0], z=zL[0]), center_bottom))
            pick_poses.append(T * SE3.Rt(SO3.TwoVectors(x=xL[1], z=zL[0]), center_bottom))
            pick_poses.append(T * SE3.Rt(SO3.TwoVectors(x=xL[2], z=zL[0]), center_bottom))
            pick_poses.append(T * SE3.Rt(SO3.TwoVectors(x=xL[3], z=zL[0]), center_bottom))
            pick_poses.append(T * SE3.Rt(SO3.TwoVectors(x=xL[0], z=zL[1]), center_top))
            pick_poses.append(T * SE3.Rt(SO3.TwoVectors(x=xL[1], z=zL[1]), center_top))
            pick_poses.append(T * SE3.Rt(SO3.TwoVectors(x=xL[2], z=zL[1]), center_top))
            pick_poses.append(T * SE3.Rt(SO3.TwoVectors(x=xL[3], z=zL[1]), center_top))
            pick_poses.append(T * SE3.Rt(SO3.TwoVectors(x=xL[0], z=zL[0]), center))
            pick_poses.append(T * SE3.Rt(SO3.TwoVectors(x=xL[1], z=zL[0]), center))
            pick_poses.append(T * SE3.Rt(SO3.TwoVectors(x=xL[2], z=zL[0]), center))
            pick_poses.append(T * SE3.Rt(SO3.TwoVectors(x=xL[3], z=zL[0]), center))
            pick_poses.append(T * SE3.Rt(SO3.TwoVectors(x=xL[0], z=zL[1]), center))
            pick_poses.append(T * SE3.Rt(SO3.TwoVectors(x=xL[1], z=zL[1]), center))
            pick_poses.append(T * SE3.Rt(SO3.TwoVectors(x=xL[2], z=zL[1]), center))
            pick_poses.append(T * SE3.Rt(SO3.TwoVectors(x=xL[3], z=zL[1]), center))
        return pick_poses

    def generate_frustum_pick_poses(self, top_r, bottom_r, height, num_s, num_c, T):
        return self.gen_frustum_side_pick_poses(top_r, bottom_r, height, num_s, T) + self.gen_frustum_center_pick_poses(height, num_c, T)

    def gen_frustum_side_pick_poses(self, top_r, bottom_r, height, num_each_side, T):
        pick_poses = []
        step = 0 if num_each_side == 1 else 2 * pi / num_each_side
        alpha = atan(abs(top_r - bottom_r) / height)
        top_z = height / 2
        bottom_z = -1 * height / 2
        top_zL = np.array([[0, 0, -1]])
        bottom_zL = np.array([[0, 0, 1]])
        for idx in range(num_each_side):
            t = idx * step
            top_x = top_r * cos(t)
            top_y = top_r * sin(t)
            bottom_x = bottom_r * cos(t)
            bottom_y = bottom_r * sin(t)
            top_xL = np.array([[top_x, top_y, 0]])
            bottom_xL = np.array([[bottom_x, bottom_y, 0]])
            if top_r > bottom_r:
                top_T = SE3.Rt(SO3.TwoVectors(x=top_xL, z=top_zL) * SO3.Ry(-alpha), [top_x, top_y, top_z])
                top_T2 = SE3.Rt(SO3.TwoVectors(x=top_xL, z=top_zL), [top_x, top_y, top_z])
                bottom_T = SE3.Rt(SO3.TwoVectors(x=bottom_xL, z=bottom_zL) * SO3.Ry(-alpha), [bottom_x, bottom_y, bottom_z])
                bottom_T2 = SE3.Rt(SO3.TwoVectors(x=bottom_xL, z=bottom_zL), [bottom_x, bottom_y, bottom_z])
            else:
                top_T = SE3.Rt(SO3.TwoVectors(x=top_xL, z=top_zL) * SO3.Ry(alpha), [top_x, top_y, top_z])
                top_T2 = SE3.Rt(SO3.TwoVectors(x=top_xL, z=top_zL), [top_x, top_y, top_z])
                bottom_T = SE3.Rt(SO3.TwoVectors(x=bottom_xL, z=bottom_zL) * SO3.Ry(-alpha), [bottom_x, bottom_y, bottom_z])
                bottom_T2 = SE3.Rt(SO3.TwoVectors(x=bottom_xL, z=bottom_zL), [bottom_x, bottom_y, bottom_z])
            pick_poses.append(T * top_T * SE3.Rz(np.pi/2))
            pick_poses.append(T * top_T2 * SE3.Rz(np.pi/2))
            pick_poses.append(T * bottom_T * SE3.Rz(np.pi/2))
            pick_poses.append(T * bottom_T2 * SE3.Rz(np.pi/2))
        return pick_poses

    def gen_frustum_center_pick_poses(self, height, num_each_position, T):
        pick_poses = []
        center = np.zeros(3)
        step = 0 if num_each_position == 1 else 2 * pi / num_each_position
        top_x = 0
        top_y = 0
        top_z = height / 2
        bottom_x = 0
        bottom_y = 0
        bottom_z = -1 * height / 2
        for idx in range(num_each_position):
            t = step * idx
            top_T = SE3.Rt(SO3.Rz(t) * SO3.Rx(pi), [top_x, top_y, top_z])
            bottom_T = SE3.Rt(SO3.Rz(-t), [bottom_x, bottom_y, bottom_z])
            top_T1 = SE3.Rt(SO3.Ry(pi/2) * SO3.Rx(t), [top_x, top_y, top_z])
            top_T2 = SE3.Rt(SO3.Ry(-pi/2) * SO3.Rx(t), [top_x, top_y, top_z])
            bottom_T1 = SE3.Rt(SO3.Ry(pi/2) * SO3.Rx(t), [bottom_x, bottom_y, bottom_z])
            bottom_T2 = SE3.Rt(SO3.Ry(-pi/2) * SO3.Rx(t), [bottom_x, bottom_y, bottom_z])
            center_T1 = SE3.Rt(SO3.Ry(pi/2) * SO3.Rx(t), center)
            center_T2 = SE3.Rt(SO3.Ry(-pi/2) * SO3.Rx(t), center)
            pick_poses.append(T * top_T)
            pick_poses.append(T * bottom_T)
            pick_poses.append(T * top_T1)
            pick_poses.append(T * top_T2)
            pick_poses.append(T * bottom_T1)
            pick_poses.append(T * bottom_T2)
            pick_poses.append(T * center_T1)
            pick_poses.append(T * center_T2)
        return pick_poses

    def generate_ellipsoid_pick_poses(self, num_each_direction, T):
        center = np.zeros(3)
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
            pick_poses.append(T * T1)
            pick_poses.append(T * T2)
            pick_poses.append(T * T3)
            pick_poses.append(T * T4)
            pick_poses.append(T * T5)
            pick_poses.append(T * T6)
        return pick_poses

    def filter_pose_by_gripper_range(self, pcd, poses, fingerRange, t = 10):
        """
            Check pick poses for 2-finger gripper
        Given:
            pcd - point cloud in open3d
            fingerRange - [w, h, d] contact range in x, y, z axis at pick pose coordinate
            poses - SE3 poses to check
            t - threshold of points in finger contact OBB
        Return:
            fileteredPoses - poses after filter
        """
        pts = np.asarray(pcd)
        filteredPoses = []
        for pose in poses:
            fingerObb = o3d.geometry.OrientedBoundingBox(pose.t, pose.R, np.array(fingerRange))
            idxs = fingerObb.get_point_indices_within_bounding_box(pcd.points)
            if len(idxs) > t:
                filteredPoses.append(pose)
        return filteredPoses

    def filter_pose_by_axis_diff(self, poses, axis=2, ref_axis=[0,0,1], t=np.pi/2, sorted=False):
        """
            根据位姿的某一个轴与参考轴的夹角进行排序，并剔除掉超过阈值的位姿。
        参数:
            poses (list of np.ndarray/SE3): 一组位姿矩阵，每个矩阵为4x4的变换矩阵或使用 spatialmath 中的 SE3 对象表示。
            axis (int): 指定的轴索引（0, 1, 2分别代表x, y, z轴）。
            ref_axis (list): 参考轴，默认为[0, 0, 1]，即z轴。
            t (float): 夹角阈值，超过该阈值的位姿将被剔除。
            sorted (bool): 是否对结果进行排序，默认为False。
        返回:
            list of np.ndarray: 剔除掉超过阈值后的位姿列表，并根据需要进行排序。
        """
        def angle_between_axes(self, pose_axis, ref_axis):
            # 计算两个轴之间的夹角
            pose_axis = np.array(pose_axis)
            ref_axis = np.array(ref_axis)
            cos_angle = np.dot(pose_axis, ref_axis) / (np.linalg.norm(pose_axis) * np.linalg.norm(ref_axis))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免数值误差导致的cos值超出[-1, 1]范围
            angle = np.arccos(cos_angle)
            return angle
        valid_poses = []
        angles = []
        for pose in poses:
            # 获取位姿矩阵的指定轴
            if isinstance(pose, SE3):
                pose_axis = pose.A[:3, axis]
            else:
                pose_axis = pose[:3, axis]
            angle = angle_between_axes(pose_axis, ref_axis)
            if angle <= t:
                valid_poses.append(pose)
                angles.append(angle)
        if sorted:
            # 按照角度进行排序
            sorted_indices = np.argsort(angles)
            valid_poses = [valid_poses[i] for i in sorted_indices]
        return valid_poses

    def sort_pose_by_rot_diff(self, poses, ref_pose):
        """
            根据旋转矩阵的差异对位姿列表进行排序。
        参数:
            poses (list of SE3): 一组位姿，使用 spatialmath 中的 SE3 对象表示。
            ref_pose (SE3): 参考位姿，使用 spatialmath 中的 SE3 对象表示。
        返回:
            list of SE3: 根据旋转矩阵差异排序后的位姿列表。
        """
        def rotation_difference(pose1, pose2):
            # 计算两个位姿的旋转矩阵差异
            return pose1.angdist(pose2, metric=6)
        # 计算每个位姿与参考位姿的旋转矩阵差异
        differences = [rotation_difference(pose, ref_pose) for pose in poses]
        # 根据差异对位姿进行排序
        sorted_indices = np.argsort(differences)
        sorted_poses = [poses[i] for i in sorted_indices]
        return sorted_poses

    def filter_pose_by_bin_side(self, poses, bin_size, ignore_size, bin_pose, threshold, axis=0):
        """
            根据距离料框壁的距离，选择对应的料框壁法向量，与给定的抓取位姿轴进行夹角过滤。
        参数:
            poses (list of SE3): 一组位姿，使用 spatialmath 中的 SE3 对象表示。
            bin_size: 在x, y平面内的尺寸。
            ignore_size: 在x, y平面内的忽略的尺寸。
            bin_pose: 料框的位姿，与抓取位姿相对于同一参考系。
            threshold: 角度阈值。
            axis: 抓取位姿的轴。
        返回:
            list of SE3: 根据旋转矩阵差异排序后的位姿列表。
        """
        half_size = np.asarray(bin_size) / 2
        half_ignore_size = np.asarray(ignore_size) / 2
        pose_filtered = []
        for pose in poses:
            pose_in_bin = bin_pose.inv() * pose
            if abs(pose_in_bin.t[0]) < half_ignore_size[0] and abs(pose_in_bin.t[1]) < half_ignore_size[1]:
                pose_filtered.append(pose)
                continue
            if half_size[0] - abs(pose_in_bin.t[0]) < half_size[1] - abs(pose_in_bin.t[1]):
                if pose_in_bin.t[0] > 0:
                    angle_between_X = np.arccos(np.dot(bin_pose.n, pose_in_bin[:3, axis]))
                else:
                    angle_between_X = np.arccos(np.dot(-1 * bin_pose.n, pose_in_bin[:3, axis]))
            else:
                if pose_in_bin.t[1] > 0:
                    angle_between_X = np.arccos(np.dot(bin_pose.o, pose_in_bin[:3, axis]))
                else:
                    angle_between_X = np.arccos(np.dot(-1 * bin_pose.o, pose_in_bin[:3, axis]))
            if angle_between_X < threshold:
                pose_filtered.append(pose)
        return pose_filtered