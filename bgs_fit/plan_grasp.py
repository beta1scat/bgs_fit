import os
import json
import time
import rclpy
import numpy as np
from rclpy.action import ActionClient
from rclpy.node import Node

from ros2_data.action import MoveXYZW
from tf_msgs.srv import StringBool
from spatialmath import SO3, SE3
from spatialmath import UnitQuaternion as UQ
from .scripts.utils import *

# Unit is meter
tool_coordinate = SE3.Trans([0, 0, 0.2])
cam_in_base = SE3.Trans([0.35, -0.3, 1]) * UQ([0.0, 0.707, -0.707, 0.0]).SE3()

class PlanGrasp(Node):

    def __init__(self):
        super().__init__('plan_grasp')
        self._action_client = ActionClient(self, MoveXYZW, "/MoveXYZW")

        self.srv = self.create_service(StringBool, 'plan_grasp', self.plan_grasp)
        self.get_logger().info('MoveXYZW action client initialization completed!')

    def send_goal(self, goal):
        goal_msg = MoveXYZW.Goal()
        goal_msg.positionx = goal[0]
        goal_msg.positiony = goal[1]
        goal_msg.positionz = goal[2]
        goal_msg.yaw = goal[3]
        goal_msg.pitch = goal[4]
        goal_msg.roll = goal[5]
        goal_msg.speed = 1.0
        self._action_client.wait_for_server()
        self.get_logger().info('Send goal!')
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def plan_grasp(self, request, response):
        self.get_logger().info('Vision results path: %s' % (request.str))
        # 读取图片 & 深度图
        base_path = request.str
        if not os.path.exists(base_path):
            response.b = False
            return response
        poses_path = os.path.join(base_path, "poses.txt")
        with open(poses_path) as f:
            poses_twist = json.load(f)
        poses = []
        for t in poses_twist:
            poses.append(SE3.Exp(t))
        self.get_logger().info(f"Number of pick points: {len(poses)}")
        poses_filtered = filter_pose_by_axis_diff(poses, 2, [0, 0, -1], np.pi/4)
# [INFO] [1719914215.326643135] [robot_pose_print]:   + position:
# [INFO] [1719914215.326661860] [robot_pose_print]:     - x = 0.302324
# [INFO] [1719914215.326678692] [robot_pose_print]:     - y = 0.000931
# [INFO] [1719914215.326699882] [robot_pose_print]:     - z = 0.357930
# [INFO] [1719914215.326719749] [robot_pose_print]:   + orientation:
# [INFO] [1719914215.326736310] [robot_pose_print]:     - x = 0.707081
# [INFO] [1719914215.326756899] [robot_pose_print]:     - y = 0.707132
# [INFO] [1719914215.326775113] [robot_pose_print]:     - z = 0.000176
# [INFO] [1719914215.326793849] [robot_pose_print]:     - w = 0.000241
        T_ref = SE3.Trans([0.302324, 0.000931, 0.357930])*UQ([0.000241, 0.707081, 0.707132, 0.000176]).SE3()
        poses_sorted = sort_pose_by_rot_diff(poses_filtered, T_ref)
        self.get_logger().info(f"Number of pick points after filtered: {len(poses_sorted)}")
        for pose in poses_sorted:
            poseInBase = cam_in_base*pose*SE3.Rz(np.pi/2)
            flangInBase = poseInBase*(tool_coordinate.inv())
            xyzPos = flangInBase.t
            rpyDeg = flangInBase.rpy(unit='deg')
            goal = [*xyzPos, *rpyDeg]
            self.get_action_result = False
            self.send_goal(goal)
            # while not self.get_action_result:
            #     self.get_logger().info("等待规划结果")
            #     time.sleep(1)
            # if self.plan_success:
            #     break
        response.b = True
        return response

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return
        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        # self.get_action_result = True
        result = future.result().result
        self.get_logger().info('Result: {0}'.format(result.result))
        if result == "MoveXYZW:SUCCESS":
            self.get_logger().info("规划成功")
            # self.plan_success = True
        # rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)

    plan_grasp = PlanGrasp()
    # plan_grasp.send_goal()
    rclpy.spin(plan_grasp)


if __name__ == '__main__':
    main()