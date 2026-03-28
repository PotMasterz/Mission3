import time
from std_srvs.srv import Trigger
from mission3_interfaces.srv import VerifyCompliance
import rclpy


class ComplianceChecker:
    """Helper class to wait and verify if a person complied with a rule."""

    def __init__(self, node, capture_client, verify_client):
        self._node = node
        self._capture_client = capture_client
        self._verify_client = verify_client

    def check(self, before_path, violation_json, wait_seconds=17):
        """
        Wait, recapture, and verify compliance.

        Returns:
            (bool complied, str after_path)
        """
        self._node.get_logger().info(f'Waiting {wait_seconds}s for compliance...')
        time.sleep(wait_seconds)

        # Capture after image
        self._node.get_logger().info('Capturing after image...')
        capture_req = Trigger.Request()
        capture_future = self._capture_client.call_async(capture_req)
        rclpy.spin_until_future_complete(self._node, capture_future, timeout_sec=10.0)

        capture_result = capture_future.result()
        if capture_result is None or not capture_result.success:
            self._node.get_logger().error('Failed to capture after image')
            return False, ''

        after_path = capture_result.message

        # Verify compliance
        self._node.get_logger().info(f'Verifying compliance: before={before_path}, after={after_path}')
        verify_req = VerifyCompliance.Request()
        verify_req.before_path = before_path
        verify_req.after_path = after_path
        verify_req.violation_json = violation_json

        verify_future = self._verify_client.call_async(verify_req)
        rclpy.spin_until_future_complete(self._node, verify_future, timeout_sec=30.0)

        verify_result = verify_future.result()
        if verify_result is None:
            self._node.get_logger().error('Verify service call failed')
            return False, after_path

        self._node.get_logger().info(
            f'Compliance result: complied={verify_result.complied}, reason={verify_result.reason}'
        )
        return verify_result.complied, after_path
