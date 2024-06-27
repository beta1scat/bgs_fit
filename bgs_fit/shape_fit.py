
import rclpy
from rclpy.node import Node
from tf_msgs.srv import StringBool
from .scripts.utils import *

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(StringBool, 'shape_fit', self.shape_fit)

    def shape_fit(self, request, response):
        response.b = True
        self.get_logger().info('Incoming request\n str: %s' % (request.str))
        ellioseModel = EllipseLeastSquaresModel()
        return response

def main():
    rclpy.init()

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()