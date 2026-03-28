import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from mission3_interfaces.srv import NavigateTo
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Quaternion
from ament_index_python.packages import get_package_share_directory
import math
import time
import yaml
import os


class PointNode(Node):

    def __init__(self):
        super().__init__('point_node')

        config_path = os.path.join(
            get_package_share_directory('mission3'), 'config', 'waypoints.yaml'
        )
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self._waypoints = {}
        for wp in config['waypoints']:
            self._waypoints[wp['id']] = wp

        settings = config.get('settings', {})
        self._nav_timeout = settings.get('nav_timeout_seconds', 30)
        self._max_retries = settings.get('max_retries_nav', 3)

        self._nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.create_service(NavigateTo, '/point/navigate', self._navigate_callback)
        self._status_pub = self.create_publisher(String, '/point/status', 10)

        self.get_logger().info(
            f'PointNode ready — loaded {len(self._waypoints)} waypoints'
        )

    def _yaw_to_quaternion(self, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        return Quaternion(x=0.0, y=0.0, z=sy, w=cy)

    def _publish_status(self, msg):
        status_msg = String()
        status_msg.data = msg
        self._status_pub.publish(status_msg)
        self.get_logger().info(msg)

    def _navigate_callback(self, request, response):
        waypoint_id = request.waypoint_id
        self.get_logger().info(f'Navigate request to {waypoint_id}')

        if waypoint_id not in self._waypoints:
            response.success = False
            response.message = f'Waypoint {waypoint_id} not found'
            self._publish_status(f'FAILED: {response.message}')
            return response

        wp = self._waypoints[waypoint_id]

        for attempt in range(self._max_retries):
            self._publish_status(
                f'NAVIGATING to {waypoint_id} (attempt {attempt + 1}/{self._max_retries})'
            )

            success = self._navigate_to_pose(wp)
            if success:
                response.success = True
                response.message = f'Successfully navigated to {waypoint_id}'
                self._publish_status(f'ARRIVED at {waypoint_id}')
                return response

            if attempt < self._max_retries - 1:
                self.get_logger().warn(f'Navigation failed, retrying...')
                time.sleep(1)

        response.success = False
        response.message = (
            f'Failed to navigate to {waypoint_id} after {self._max_retries} attempts'
        )
        self._publish_status(f'FAILED: {response.message}')
        return response

    def _navigate_to_pose(self, waypoint):
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Nav2 action server not available')
            return False

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(waypoint['x'])
        goal.pose.pose.position.y = float(waypoint['y'])
        goal.pose.pose.orientation = self._yaw_to_quaternion(float(waypoint['yaw']))

        try:
            send_future = self._nav_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)

            goal_handle = send_future.result()
            if goal_handle is None or not goal_handle.accepted:
                self.get_logger().error('Goal rejected by Nav2')
                return False

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(
                self, result_future, timeout_sec=self._nav_timeout
            )

            result = result_future.result()
            if result is None:
                self.get_logger().error('Navigation timed out')
                return False

            if result.result.error_code == 0:
                return True
            else:
                self.get_logger().error(
                    f'Nav2 error {result.result.error_code}: {result.result.error_msg}'
                )
                return False

        except Exception as e:
            self.get_logger().error(f'Navigation error: {e}')
            return False


def main(args=None):
    rclpy.init(args=args)
    node = PointNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
