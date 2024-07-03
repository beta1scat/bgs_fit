import os
import cv2
import rclpy
import json

from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import PointCloud2
from cv_bridge import CvBridge

from sensor_msgs_py import point_cloud2 as pc2
from gazebo_msgs.srv import GetModelList, GetEntityState

imageIsDone = False
depthImageIsDone = False
cameraInfoIsDone = False
pcdIsDone = False
posesIsDone = False

def save_pointcloud_to_ply(pointcloud, file_path):
    with open(file_path, 'w') as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write("element vertex {}\n".format(pointcloud.shape[0]))
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")
        for point in pointcloud:
            file.write("{} {} {}\n".format(point[0], point[1], point[2]))

def save_camera_info(camera_info_msg, file_path):
    # 将 CameraInfo 消息转换为字典
    camera_info_dict = {
        'header': {
            'stamp': {'secs': camera_info_msg.header.stamp.sec, 'nsecs': camera_info_msg.header.stamp.nanosec},
            'frame_id': camera_info_msg.header.frame_id
        },
        'height': camera_info_msg.height,
        'width': camera_info_msg.width,
        'distortion_model': camera_info_msg.distortion_model,
        'D': list(camera_info_msg.d),
        'K': list(camera_info_msg.k),
        'R': list(camera_info_msg.r),
        'P': list(camera_info_msg.p),
        'binning_x': camera_info_msg.binning_x,
        'binning_y': camera_info_msg.binning_y,
        'roi': {
            'x_offset': camera_info_msg.roi.x_offset,
            'y_offset': camera_info_msg.roi.y_offset,
            'height': camera_info_msg.roi.height,
            'width': camera_info_msg.roi.width,
            'do_rectify': camera_info_msg.roi.do_rectify
        }
    }

    # 将字典保存到 JSON 文件中
    with open(file_path, 'w') as file:
        json.dump(camera_info_dict, file, indent=4)
class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('depth_image_subscriber')
        self.declare_parameter('base_path', '/root/ros_ws/src/data/0000')
        self.savePath = self.get_parameter('base_path').get_parameter_value().string_value
        os.makedirs(self.savePath, exist_ok=True)
        self.cv_bridge = CvBridge()
        self.imageSubscription = self.create_subscription(
            Image,
            '/RGBD_camera/image_raw',
            self.imageListener_callback,
            1)
        self.imageSubscription  # prevent unused variable warning
        self.depthSubscription = self.create_subscription(
            Image,
            '/RGBD_camera/depth/image_raw',
            self.depthListener_callback,
            1)
        self.depthSubscription  # prevent unused variable warning
        self.cameraInfoSubscription = self.create_subscription(
            CameraInfo,
            '/RGBD_camera/depth/camera_info',
            self.cameraInfoListener_callback,
            1)
        self.cameraInfoSubscription  # prevent unused variable warning
        self.pointsSubscription = self.create_subscription(
            PointCloud2,
            '/RGBD_camera/points',
            self.pointsListener_callback,
            1)
        self.pointsSubscription  # prevent unused variable warning

        # Get model poses
        self.get_model_list_client = self.create_client(GetModelList, '/get_model_list')
        self.get_entity_state_client = self.create_client(GetEntityState, '/ros2_grasp/get_entity_state')

        while not self.get_model_list_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /get_model_list not available, waiting again...')

        while not self.get_entity_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /ros2_grasp/get_entity_state not available, waiting again...')
        self.file_name = os.path.join(self.savePath, "model_poses.json")
        self.get_model_list()

    def imageListener_callback(self, msg):
        global imageIsDone
        if imageIsDone:
            self.get_logger().info("Image has already saved")
            return
        try:
            # Convert ROS Image message to OpenCV image
            self.get_logger().info('I heard height: "%s"' % msg.height)
            self.get_logger().info('I heard width: "%s"' % msg.width)
            self.get_logger().info('I heard encoding: "%s"' % msg.encoding)
            self.get_logger().info('I heard is_bigendian: "%s"' % msg.is_bigendian)
            self.get_logger().info('I heard step: "%s"' % msg.step)
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error("Error converting image: %s" % str(e))
            return
        # Save image locally
        image_file_path = os.path.join(self.savePath, 'image.png')
        cv2.imwrite(image_file_path, cv_image)
        self.get_logger().info("Image saved to %s" % image_file_path)
        imageIsDone = True

    def depthListener_callback(self, msg):
        global depthImageIsDone
        if depthImageIsDone:
            self.get_logger().info("Depth image has already saved")
            return
        try:
            # Convert ROS Image message to OpenCV image
            # self.get_logger().info('I heard data: "%s"' % msg.data)
            self.get_logger().info('I heard height: "%s"' % msg.height)
            self.get_logger().info('I heard width: "%s"' % msg.width)
            self.get_logger().info('I heard encoding: "%s"' % msg.encoding)
            self.get_logger().info('I heard is_bigendian: "%s"' % msg.is_bigendian)
            self.get_logger().info('I heard step: "%s"' % msg.step)
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error("Error converting image: %s" % str(e))
            return
        # Save depth image locally
        depth_image_file_path = os.path.join(self.savePath, 'depth.tiff')  # Change this to your desired path
        cv2.imwrite(depth_image_file_path, cv_image)
        self.get_logger().info("Depth image saved to %s" % depth_image_file_path)
        depthImageIsDone = True

    def cameraInfoListener_callback(self, msg):
        global cameraInfoIsDone
        if cameraInfoIsDone:
            self.get_logger().info("Camera info has already saved")
            return
        self.get_logger().info('I heard header: "%s"' % msg.header)
        self.get_logger().info('I heard p: "%s"' % msg.p)
        info_file_path = os.path.join(self.savePath, 'camera_info.json')  # Change this to your desired path
        save_camera_info(msg, info_file_path)
        self.get_logger().info("Camera info saved to %s" % info_file_path)
        cameraInfoIsDone = True

    def pointsListener_callback(self, msg):
        global pcdIsDone
        if pcdIsDone:
            self.get_logger().info("Points has already saved")
            return
        self.get_logger().info('I heard header: "%s"' % msg.header)
        pc = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))
        points_file_path = os.path.join(self.savePath, 'points.ply')  # Change this to your desired path
        save_pointcloud_to_ply(pc, points_file_path)
        self.get_logger().info("Points saved to %s" % points_file_path)
        pcdIsDone = True

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
        global posesIsDone
        posesIsDone = True

def main(args=None):
    rclpy.init(args=args)
    # rate = node.create_timer(1)
    minimal_subscriber = MinimalSubscriber()
    while not (imageIsDone and depthImageIsDone and cameraInfoIsDone and pcdIsDone and posesIsDone):
        rclpy.spin_once(minimal_subscriber, timeout_sec=0)
        # rate.sleep()
    
    # rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()