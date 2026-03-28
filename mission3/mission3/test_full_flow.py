import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from mission3_interfaces.srv import AnalyzeRoom
import json
import sys


class TestFullFlow(Node):

    def __init__(self):
        super().__init__('test_full_flow')

        self._capture_client = self.create_client(Trigger, '/capture/save')
        self._analyze_client = self.create_client(AnalyzeRoom, '/bridge/analyze')
        self._speak_pub = self.create_publisher(String, '/speak', 10)

        self.get_logger().info('Waiting for capture and bridge services...')
        self._capture_client.wait_for_service(timeout_sec=30.0)
        self._analyze_client.wait_for_service(timeout_sec=30.0)
        self.get_logger().info('Services ready. Running test...')

        self._run_test()

    def _run_test(self):
        # Step 1: Capture
        self.get_logger().info('--- Step 1: Capturing image ---')
        cap_req = Trigger.Request()
        cap_future = self._capture_client.call_async(cap_req)
        rclpy.spin_until_future_complete(self, cap_future, timeout_sec=10.0)
        cap_result = cap_future.result()

        if cap_result is None or not cap_result.success:
            self.get_logger().error(f'Capture failed: {cap_result.message if cap_result else "timeout"}')
            return

        image_path = cap_result.message
        self.get_logger().info(f'Captured: {image_path}')

        # Step 2: Analyze
        self.get_logger().info('--- Step 2: Analyzing with GPT ---')
        analyze_req = AnalyzeRoom.Request()
        analyze_req.image_path = image_path
        analyze_req.room_label = 'living_room'
        analyze_req.is_forbidden = False

        # Check if a room label was passed as argument
        if len(sys.argv) > 1:
            analyze_req.room_label = sys.argv[1]
        if len(sys.argv) > 2 and sys.argv[2].lower() == 'true':
            analyze_req.is_forbidden = True

        analyze_future = self._analyze_client.call_async(analyze_req)
        rclpy.spin_until_future_complete(self, analyze_future, timeout_sec=60.0)
        analyze_result = analyze_future.result()

        if analyze_result is None or not analyze_result.success:
            self.get_logger().error('Analysis failed')
            return

        self.get_logger().info(f'GPT response: {analyze_result.violations_json}')

        # Step 3: Parse and speak
        self.get_logger().info('--- Step 3: Announcing violations ---')
        try:
            data = json.loads(analyze_result.violations_json)
            violations = data.get('violations', [])
        except json.JSONDecodeError:
            self.get_logger().error('Failed to parse violations JSON')
            return

        if not violations:
            msg = String()
            msg.data = 'No violations detected. Everything looks good.'
            self._speak_pub.publish(msg)
            self.get_logger().info('No violations found.')
            return

        for v in violations:
            person = v.get('person_description', 'someone')
            rule = v.get('rule_name', 'unknown')
            rule_num = v.get('rule_number', 0)
            instruction = v.get('instruction_to_speak', '')

            self.get_logger().info(
                f'VIOLATION: {person} breaking rule {rule_num} ({rule})'
            )

            # Speak the instruction through robot
            if instruction:
                msg = String()
                msg.data = instruction
                self._speak_pub.publish(msg)
                self.get_logger().info(f'SPEAKING: {instruction}')

        self.get_logger().info(f'Done. Found {len(violations)} violation(s).')


def main(args=None):
    rclpy.init(args=args)
    node = TestFullFlow()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
