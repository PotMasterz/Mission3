import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
from datetime import datetime
from pathlib import Path
import cv2


class CaptureNode(Node):

    def __init__(self):
        super().__init__('capture_node')

        self._output_dir = Path.home() / 'mission3_images'
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._bridge = CvBridge()
        self._latest_frame = None

        self.create_subscription(Image, '/camera/color/image_raw', self._image_callback, 10)
        self.create_service(Trigger, '/capture/save', self._save_callback)
        self._path_pub = self.create_publisher(String, '/capture/last_path', 10)

        self.get_logger().info('CaptureNode ready')

    def _image_callback(self, msg):
        self._latest_frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _save_callback(self, request, response):
        if self._latest_frame is None:
            response.success = False
            response.message = 'No frame received yet'
            return response

        filename = datetime.now().strftime('capture_%Y%m%d_%H%M%S.jpg')
        full_path = self._output_dir / filename

        success = cv2.imwrite(str(full_path), self._latest_frame)
        if not success:
            response.success = False
            response.message = 'Failed to write image'
            return response

        response.success = True
        response.message = str(full_path)

        path_msg = String()
        path_msg.data = str(full_path)
        self._path_pub.publish(path_msg)

        self.get_logger().info(f'Saved: {full_path}')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = CaptureNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
