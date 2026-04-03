import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from mission3_interfaces.srv import AnalyzeRoom
from datetime import datetime
from pathlib import Path
import json
import sys
import cv2


# Colors for each rule (BGR)
RULE_COLORS = {
    1: (0, 0, 255),    # Red - shoes
    2: (0, 165, 255),  # Orange - forbidden room
    3: (0, 255, 255),  # Yellow - littering
    4: (255, 0, 0),    # Blue - no drink
}

RULE_LABELS = {
    1: 'no_shoes',
    2: 'forbidden_room',
    3: 'littering',
    4: 'no_drink',
}


class TestFullFlow(Node):

    def __init__(self):
        super().__init__('test_full_flow')

        self._output_dir = Path.home() / 'mission3_images'
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._capture_client = self.create_client(Trigger, '/capture/save')
        self._analyze_client = self.create_client(AnalyzeRoom, '/bridge/analyze')
        self._speak_pub = self.create_publisher(String, '/speak', 10)

        self.get_logger().info('Waiting for capture and bridge services...')
        self._capture_client.wait_for_service(timeout_sec=30.0)
        self._analyze_client.wait_for_service(timeout_sec=30.0)
        self.get_logger().info('Services ready. Running test...')

        self._run_test()

    def _draw_detections(self, image_path, violations):
        """Draw bounding boxes on image and save detection + crop images."""
        img = cv2.imread(image_path)
        if img is None:
            self.get_logger().error(f'Failed to load image for detection: {image_path}')
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        crop_count = 0

        for v in violations:
            bbox = v.get('bbox')
            if not bbox:
                continue

            x_min = max(0, int(bbox.get('x_min', 0)))
            y_min = max(0, int(bbox.get('y_min', 0)))
            x_max = min(img.shape[1], int(bbox.get('x_max', 0)))
            y_max = min(img.shape[0], int(bbox.get('y_max', 0)))

            if x_max <= x_min or y_max <= y_min:
                self.get_logger().warn(f'Invalid bbox: {bbox}')
                continue

            rule_num = v.get('rule_number', 0)
            person = v.get('person_description', 'unknown')
            color = RULE_COLORS.get(rule_num, (255, 255, 255))
            label = f'Rule {rule_num}: {RULE_LABELS.get(rule_num, "unknown")}'

            # Draw rectangle
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                img,
                (x_min, y_min - label_size[1] - 10),
                (x_min + label_size[0] + 4, y_min),
                color, -1
            )
            cv2.putText(
                img, label,
                (x_min + 2, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

            # Crop and save
            crop = cv2.imread(image_path)
            cropped = crop[y_min:y_max, x_min:x_max]
            crop_path = self._output_dir / f'crop_{timestamp}_rule{rule_num}_{crop_count}.jpg'
            cv2.imwrite(str(crop_path), cropped)
            self.get_logger().info(f'Saved crop: {crop_path}')
            crop_count += 1

        # Save full annotated image
        detection_path = self._output_dir / f'detection_{timestamp}.jpg'
        cv2.imwrite(str(detection_path), img)
        self.get_logger().info(f'Saved detection image: {detection_path}')

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

        # Step 3: Parse violations
        self.get_logger().info('--- Step 3: Processing violations ---')
        try:
            data = json.loads(analyze_result.violations_json)
            violations = data.get('violations', [])
        except json.JSONDecodeError:
            self.get_logger().error('Failed to parse violations JSON')
            return

        if not violations:
            msg = String()
            msg.data = 'ไม่พบการละเมิดกฎใดๆ ทุกอย่างเรียบร้อยดีครับ'
            self._speak_pub.publish(msg)
            self.get_logger().info('No violations found.')
            return

        # Step 4: Draw bounding boxes and save detection images
        self.get_logger().info('--- Step 4: Drawing detections ---')
        self._draw_detections(image_path, violations)

        # Step 5: Speak violations (only instructions, NOT coordinates)
        self.get_logger().info('--- Step 5: Announcing violations ---')
        for v in violations:
            person = v.get('person_description', 'someone')
            rule = v.get('rule_name', 'unknown')
            rule_num = v.get('rule_number', 0)
            instruction = v.get('instruction_to_speak', '')

            self.get_logger().info(
                f'VIOLATION: {person} breaking rule {rule_num} ({rule})'
            )

            # Speak the instruction through robot (no coordinates)
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
