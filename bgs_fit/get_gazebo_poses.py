#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
import json
from gazebo_msgs.srv import GetModelList, GetEntityState

class GazeboModelState(Node):
    def __init__(self):
        super().__init__('get_gazebo_model_states')
        self.declare_parameter('num', 0)
        num = self.get_parameter('num').get_parameter_value().integer_value
        base_path = '/root/ros_ws/src/data'
        out_dir = os.path.join(base_path, str(num).zfill(4))
        os.makedirs(out_dir, exist_ok=True)
        self.file_name = os.path.join(out_dir, "model_poses.json")
        self.get_model_list_client = self.create_client(GetModelList, '/get_model_list')
        self.get_entity_state_client = self.create_client(GetEntityState, '/ros2_grasp/get_entity_state')

        while not self.get_model_list_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /get_model_list not available, waiting again...')

        while not self.get_entity_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /ros2_grasp/get_entity_state not available, waiting again...')

        self.get_model_list()

    def get_model_list(self):
        req = GetModelList.Request()
        future = self.get_model_list_client.call_async(req)
        future.add_done_callback(self.model_list_callback)

    def model_list_callback(self, future):
        try:
            response = future.result()
            self.model_names = response.model_names
            self.get_logger().info(f"Found models: {self.model_names}")
            self.models_info = {}
            self.get_model_states()
        except Exception as e:
            self.get_logger().error(f'Service call failed {e}')

    def get_model_states(self):
        for model_name in self.model_names:
            req = GetEntityState.Request()
            req.name = model_name
            future = self.get_entity_state_client.call_async(req)
            future.add_done_callback(lambda future, model_name=model_name: self.entity_state_callback(future, model_name))

    def entity_state_callback(self, future, model_name):
        try:
            response = future.result()
            self.models_info[model_name] = {
                'position': {
                    'x': response.state.pose.position.x,
                    'y': response.state.pose.position.y,
                    'z': response.state.pose.position.z
                },
                'orientation': {
                    'x': response.state.pose.orientation.x,
                    'y': response.state.pose.orientation.y,
                    'z': response.state.pose.orientation.z,
                    'w': response.state.pose.orientation.w
                }
            }
            if len(self.models_info) == len(self.model_names):
                self.save_to_json()
        except Exception as e:
            self.get_logger().error(f'Service call failed {e}')

    def save_to_json(self):
        with open(self.file_name, 'w') as json_file:
            json.dump(self.models_info, json_file, indent=4)
        self.get_logger().info(f'Model states saved to {self.file_name}')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = GazeboModelState()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
