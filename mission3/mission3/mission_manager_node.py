import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from mission3_interfaces.srv import AnalyzeRoom, VerifyCompliance, NavigateTo
from ament_index_python.packages import get_package_share_directory
from mission3.compliance_helper import ComplianceChecker
import json
import yaml
import os
import cv2
from datetime import datetime
from pathlib import Path


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


# States
INIT = 'INIT'
NAVIGATE_TO_WAYPOINT = 'NAVIGATE_TO_WAYPOINT'
CAPTURE_AND_ANALYZE = 'CAPTURE_AND_ANALYZE'
HANDLE_VIOLATIONS = 'HANDLE_VIOLATIONS'
WAIT_AND_VERIFY = 'WAIT_AND_VERIFY'
FORBIDDEN_ROOM_ESCORT = 'FORBIDDEN_ROOM_ESCORT'
NEXT_WAYPOINT = 'NEXT_WAYPOINT'
MISSION_COMPLETE = 'MISSION_COMPLETE'
RETURN_TO_START = 'RETURN_TO_START'
ERROR_RECOVERY = 'ERROR_RECOVERY'
DONE = 'DONE'


class MissionManagerNode(Node):

    def __init__(self):
        super().__init__('mission_manager_node')

        # Load config
        config_path = os.path.join(
            get_package_share_directory('mission3'), 'config', 'waypoints.yaml'
        )
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self._waypoints = {}
        for wp in config['waypoints']:
            self._waypoints[wp['id']] = wp

        settings = config.get('settings', {})
        self._compliance_wait = settings.get('compliance_wait_seconds', 17)
        self._max_retries_compliance = settings.get('max_retries_compliance', 3)

        # Detection images directory
        self._output_dir = Path.home() / 'mission3_images'
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Patrol order — build from config (skip wp_start)
        self._patrol_list = [
            wp['id'] for wp in config['waypoints'] if wp['id'] != 'wp_start'
        ]
        self._current_wp_index = 0

        # Service clients
        self._capture_client = self.create_client(Trigger, '/capture/save')
        self._analyze_client = self.create_client(AnalyzeRoom, '/bridge/analyze')
        self._verify_client = self.create_client(VerifyCompliance, '/bridge/verify')
        self._navigate_client = self.create_client(NavigateTo, '/point/navigate')

        # Publishers
        self._speak_pub = self.create_publisher(String, '/speak', 10)
        self._state_pub = self.create_publisher(String, '/mission_manager/state', 10)
        self._log_pub = self.create_publisher(String, '/mission_manager/violations_log', 10)

        # Compliance checker
        self._compliance = ComplianceChecker(
            self, self._capture_client, self._verify_client
        )

        # State tracking
        self._state = INIT
        self._handled_persons = set()
        self._error_count = 0
        self._violations_this_loop = 0
        self._current_violations = []
        self._current_violation_index = 0
        self._current_before_path = ''
        self._escort_attempts = 0

        # Timer drives the state machine
        self._timer = self.create_timer(1.0, self._tick)

        self.get_logger().info('MissionManagerNode ready')

    # --- Helpers ---

    def _publish_state(self):
        msg = String()
        msg.data = self._state
        self._state_pub.publish(msg)

    def _speak(self, text):
        msg = String()
        msg.data = text
        self._speak_pub.publish(msg)
        self.get_logger().info(f'SPEAK: {text}')

    def _log_event(self, event):
        msg = String()
        msg.data = event
        self._log_pub.publish(msg)
        self.get_logger().info(f'LOG: {event}')

    def _save_detection_image(self, image_path, violations):
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
                continue

            rule_num = v.get('rule_number', 0)
            color = RULE_COLORS.get(rule_num, (255, 255, 255))
            label = f'Rule {rule_num}: {RULE_LABELS.get(rule_num, "unknown")}'

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(img, (x_min, y_min - label_size[1] - 10),
                          (x_min + label_size[0] + 4, y_min), color, -1)
            cv2.putText(img, label, (x_min + 2, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Crop and save
            cropped = cv2.imread(image_path)[y_min:y_max, x_min:x_max]
            crop_path = self._output_dir / f'crop_{timestamp}_rule{rule_num}_{crop_count}.jpg'
            cv2.imwrite(str(crop_path), cropped)
            self.get_logger().info(f'Saved crop: {crop_path}')
            crop_count += 1

        detection_path = self._output_dir / f'detection_{timestamp}.jpg'
        cv2.imwrite(str(detection_path), img)
        self.get_logger().info(f'Saved detection image: {detection_path}')

    def _current_waypoint_id(self):
        return self._patrol_list[self._current_wp_index]

    def _current_waypoint(self):
        return self._waypoints[self._current_waypoint_id()]

    def _call_navigate(self, waypoint_id):
        """Call /point/navigate and return success bool."""
        req = NavigateTo.Request()
        req.waypoint_id = waypoint_id
        future = self._navigate_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=120.0)
        result = future.result()
        if result is None:
            return False
        return result.success

    def _call_capture(self):
        """Call /capture/save and return (success, path)."""
        req = Trigger.Request()
        future = self._capture_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        result = future.result()
        if result is None or not result.success:
            return False, ''
        return True, result.message

    def _call_analyze(self, image_path, room_label, is_forbidden):
        """Call /bridge/analyze and return (success, violations_list)."""
        req = AnalyzeRoom.Request()
        req.image_path = image_path
        req.room_label = room_label
        req.is_forbidden = is_forbidden
        future = self._analyze_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=60.0)
        result = future.result()
        if result is None or not result.success:
            return False, []
        try:
            parsed = json.loads(result.violations_json)
            return True, parsed.get('violations', [])
        except json.JSONDecodeError:
            return False, []

    def _find_nearest_non_forbidden(self):
        """Find nearest non-forbidden waypoint for escort."""
        for wp_id in self._patrol_list:
            wp = self._waypoints[wp_id]
            if not wp.get('forbidden', False):
                return wp_id
        return 'wp_start'

    # --- State Machine ---

    def _tick(self):
        self._publish_state()

        if self._state == INIT:
            self._do_init()
        elif self._state == NAVIGATE_TO_WAYPOINT:
            self._do_navigate()
        elif self._state == CAPTURE_AND_ANALYZE:
            self._do_capture_and_analyze()
        elif self._state == HANDLE_VIOLATIONS:
            self._do_handle_violations()
        elif self._state == WAIT_AND_VERIFY:
            self._do_wait_and_verify()
        elif self._state == FORBIDDEN_ROOM_ESCORT:
            self._do_forbidden_escort()
        elif self._state == NEXT_WAYPOINT:
            self._do_next_waypoint()
        elif self._state == MISSION_COMPLETE:
            self._do_mission_complete()
        elif self._state == RETURN_TO_START:
            self._do_return_to_start()
        elif self._state == ERROR_RECOVERY:
            self._do_error_recovery()
        elif self._state == DONE:
            pass  # Mission finished

    def _do_init(self):
        self.get_logger().info('Waiting for services...')
        all_ready = True
        for client in [self._capture_client, self._analyze_client,
                       self._verify_client, self._navigate_client]:
            if not client.wait_for_service(timeout_sec=1.0):
                all_ready = False
                break

        if all_ready:
            self.get_logger().info('All services ready. Starting mission!')
            self._speak('เริ่มภารกิจแล้วครับ กำลังตรวจตราห้องต่างๆ')
            self._state = NAVIGATE_TO_WAYPOINT
        else:
            self.get_logger().info('Waiting for services to come online...')

    def _do_navigate(self):
        wp_id = self._current_waypoint_id()
        wp = self._current_waypoint()
        self.get_logger().info(f'Navigating to {wp_id} ({wp["label"]})')
        self._speak(f'กำลังเคลื่อนที่ไปยัง {wp["label"]} ครับ')

        success = self._call_navigate(wp_id)
        if success:
            self.get_logger().info(f'Arrived at {wp_id}')
            self._state = CAPTURE_AND_ANALYZE
        else:
            self.get_logger().error(f'Failed to navigate to {wp_id}')
            self._state = ERROR_RECOVERY

    def _do_capture_and_analyze(self):
        wp = self._current_waypoint()
        self.get_logger().info(f'Capturing and analyzing {wp["label"]}')

        # Capture
        success, image_path = self._call_capture()
        if not success:
            self.get_logger().error('Capture failed')
            self._state = ERROR_RECOVERY
            return

        self._current_before_path = image_path

        # Analyze
        success, violations = self._call_analyze(
            image_path, wp['label'], wp.get('forbidden', False)
        )
        if not success:
            self.get_logger().error('Analyze failed')
            self._state = ERROR_RECOVERY
            return

        # Save detection image with bounding boxes
        if violations:
            self._save_detection_image(image_path, violations)

        # Filter out already-handled persons
        new_violations = []
        for v in violations:
            person = v.get('person_description', 'unknown')
            if person not in self._handled_persons:
                new_violations.append(v)
            else:
                self.get_logger().info(f'Skipping already-handled: {person}')

        if new_violations:
            self._current_violations = new_violations
            self._current_violation_index = 0
            self._state = HANDLE_VIOLATIONS
        else:
            self.get_logger().info('No new violations found')
            self._state = NEXT_WAYPOINT

    def _do_handle_violations(self):
        if self._current_violation_index >= len(self._current_violations):
            self._state = NEXT_WAYPOINT
            return

        v = self._current_violations[self._current_violation_index]
        person = v.get('person_description', 'unknown')
        rule_num = v.get('rule_number', 0)
        rule_name = v.get('rule_name', 'unknown')
        instruction = v.get('instruction_to_speak', '')
        action = v.get('action_required', '')

        # Log identification
        self._log_event(f'IDENTIFIED: rule {rule_num} ({rule_name}) by {person}')
        self._violations_this_loop += 1

        # Speak instruction
        self._speak(instruction)
        self._log_event(f'INSTRUCTED: {person} → {instruction}')

        # Track as handled
        self._handled_persons.add(person)

        # Route based on action
        if action == 'leave_room':
            self._escort_attempts = 0
            self._state = FORBIDDEN_ROOM_ESCORT
        else:
            # remove_shoes, pick_trash, get_drink all use wait-and-verify
            self._state = WAIT_AND_VERIFY

    def _do_wait_and_verify(self):
        v = self._current_violations[self._current_violation_index]
        person = v.get('person_description', 'unknown')
        rule_num = v.get('rule_number', 0)
        violation_json = json.dumps(v)

        complied, after_path = self._compliance.check(
            self._current_before_path, violation_json, self._compliance_wait
        )

        if complied:
            self._log_event(
                f'COMPLIED: {person} | rule {rule_num} | '
                f'before={self._current_before_path} after={after_path}'
            )
            self._speak('ขอบคุณที่ให้ความร่วมมือครับ')
        else:
            self._log_event(f'FAILED_COMPLIANCE: {person} | rule {rule_num}')
            self._speak('ดูเหมือนว่ายังไม่ได้แก้ไขปัญหาครับ')

        # Move to next violation
        self._current_violation_index += 1
        if self._current_violation_index < len(self._current_violations):
            self._state = HANDLE_VIOLATIONS
        else:
            self._state = NEXT_WAYPOINT

    def _do_forbidden_escort(self):
        self._escort_attempts += 1
        self.get_logger().info(
            f'Forbidden room escort attempt {self._escort_attempts}/3'
        )

        # Navigate out to nearest non-forbidden waypoint
        safe_wp = self._find_nearest_non_forbidden()
        self._speak('กรุณาตามผมออกจากห้องนี้ด้วยครับ ห้องนี้เป็นห้องห้ามเข้า')
        self._call_navigate(safe_wp)

        # Navigate back to forbidden room to recheck
        forbidden_wp_id = self._current_waypoint_id()
        self._call_navigate(forbidden_wp_id)

        # Recapture and re-analyze
        success, image_path = self._call_capture()
        if success:
            wp = self._current_waypoint()
            success, violations = self._call_analyze(
                image_path, wp['label'], wp.get('forbidden', False)
            )
            if success and len(violations) == 0:
                v = self._current_violations[self._current_violation_index]
                person = v.get('person_description', 'unknown')
                rule_num = v.get('rule_number', 0)
                self._log_event(
                    f'COMPLIED: {person} | rule {rule_num} | '
                    f'before={self._current_before_path} after={image_path}'
                )
                self._speak('ห้องนี้ไม่มีคนแล้วครับ ขอบคุณครับ')
                self._current_violation_index += 1
                if self._current_violation_index < len(self._current_violations):
                    self._state = HANDLE_VIOLATIONS
                else:
                    self._state = NEXT_WAYPOINT
                return

        if self._escort_attempts >= 3:
            v = self._current_violations[self._current_violation_index]
            person = v.get('person_description', 'unknown')
            rule_num = v.get('rule_number', 0)
            self._log_event(f'FAILED_COMPLIANCE: {person} | rule {rule_num}')
            self._current_violation_index += 1
            if self._current_violation_index < len(self._current_violations):
                self._state = HANDLE_VIOLATIONS
            else:
                self._state = NEXT_WAYPOINT
        # else: stay in FORBIDDEN_ROOM_ESCORT for next attempt

    def _do_next_waypoint(self):
        self._current_wp_index += 1

        if self._current_wp_index >= len(self._patrol_list):
            # Completed a full loop
            if self._violations_this_loop == 0:
                self.get_logger().info('Full loop with zero violations — mission complete!')
                self._state = MISSION_COMPLETE
            else:
                self.get_logger().info(
                    f'Loop done with {self._violations_this_loop} violations. Starting new loop.'
                )
                self._violations_this_loop = 0
                self._current_wp_index = 0
                self._state = NAVIGATE_TO_WAYPOINT
        else:
            self._state = NAVIGATE_TO_WAYPOINT

    def _do_mission_complete(self):
        self._speak('ภารกิจเสร็จสิ้นครับ กำลังกลับไปจุดเริ่มต้น')
        self.get_logger().info('Mission complete!')
        self._state = RETURN_TO_START

    def _do_return_to_start(self):
        success = self._call_navigate('wp_start')
        if success:
            self._speak('กลับถึงจุดเริ่มต้นแล้วครับ ภารกิจเสร็จสมบูรณ์')
            self.get_logger().info('Returned to start. Done.')
        else:
            self._speak('ไม่สามารถกลับจุดเริ่มต้นได้ แต่ภารกิจเสร็จสิ้นแล้วครับ')
            self.get_logger().error('Failed to navigate to wp_start')
        self._state = DONE

    def _do_error_recovery(self):
        self._error_count += 1
        self.get_logger().warn(f'Error recovery — count: {self._error_count}')

        if self._error_count > 3:
            self.get_logger().warn('Too many errors. Skipping to next waypoint.')
            self._error_count = 0

        self._state = NEXT_WAYPOINT


def main(args=None):
    rclpy.init(args=args)
    node = MissionManagerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
