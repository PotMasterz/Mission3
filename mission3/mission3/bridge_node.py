import rclpy
from rclpy.node import Node
from mission3_interfaces.srv import AnalyzeRoom, VerifyCompliance
from ament_index_python.packages import get_package_share_directory
import openai
import base64
import json
import time
import yaml
import os


class BridgeNode(Node):

    def __init__(self):
        super().__init__('bridge_node')

        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            self.get_logger().error('OPENAI_API_KEY environment variable not set!')
            raise RuntimeError('OPENAI_API_KEY not set')

        self._client = openai.OpenAI(api_key=api_key)

        config_path = os.path.join(
            get_package_share_directory('mission3'), 'config', 'prompts.yaml'
        )
        with open(config_path, 'r') as f:
            prompts = yaml.safe_load(f)
        self._analyze_prompt = prompts['analyze']['system']
        self._verify_prompt = prompts['verify']['system']

        self.create_service(AnalyzeRoom, '/bridge/analyze', self._analyze_callback)
        self.create_service(VerifyCompliance, '/bridge/verify', self._verify_callback)

        self.get_logger().info('BridgeNode ready')

    def _encode_image(self, path):
        with open(path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        return f'data:image/jpeg;base64,{b64}'

    def _call_gpt(self, messages, retries=3):
        for attempt in range(retries):
            try:
                result = self._client.chat.completions.create(
                    model='gpt-4o',
                    messages=messages,
                    max_tokens=1024,
                )
                return result.choices[0].message.content
            except Exception as e:
                self.get_logger().warn(f'GPT call failed (attempt {attempt + 1}/{retries}): {e}')
                if attempt < retries - 1:
                    time.sleep(2)
        return None

    def _analyze_callback(self, request, response):
        self.get_logger().info(f'Analyze request: room={request.room_label}, forbidden={request.is_forbidden}')

        try:
            image_url = self._encode_image(request.image_path)
        except Exception as e:
            self.get_logger().error(f'Failed to read image: {e}')
            response.success = False
            response.violations_json = '{"violations": []}'
            return response

        user_text = f'Room: {request.room_label}, Forbidden: {request.is_forbidden}'
        if request.is_forbidden:
            user_text += '\nThis is a FORBIDDEN room. Any person visible is violating rule 2 (no entering forbidden room).'

        messages = [
            {'role': 'system', 'content': self._analyze_prompt},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': user_text},
                {'type': 'image_url', 'image_url': {'url': image_url}},
            ]},
        ]

        result = self._call_gpt(messages)
        if result is None:
            self.get_logger().error('GPT analyze failed after all retries')
            response.success = False
            response.violations_json = '{"violations": []}'
            return response

        # Strip markdown code fences if GPT wraps the JSON
        result = result.strip()
        if result.startswith('```'):
            result = result.split('\n', 1)[1] if '\n' in result else result[3:]
            if result.endswith('```'):
                result = result[:-3].strip()

        try:
            json.loads(result)  # validate JSON
        except json.JSONDecodeError:
            self.get_logger().error(f'GPT returned invalid JSON: {result}')
            response.success = False
            response.violations_json = '{"violations": []}'
            return response

        response.success = True
        response.violations_json = result
        self.get_logger().info(f'Analyze result: {result}')
        return response

    def _verify_callback(self, request, response):
        self.get_logger().info('Verify compliance request')

        try:
            before_url = self._encode_image(request.before_path)
            after_url = self._encode_image(request.after_path)
        except Exception as e:
            self.get_logger().error(f'Failed to read images: {e}')
            response.complied = False
            response.reason = f'Failed to read images: {e}'
            return response

        messages = [
            {'role': 'system', 'content': self._verify_prompt},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': f'Violation context: {request.violation_json}'},
                {'type': 'text', 'text': 'BEFORE image:'},
                {'type': 'image_url', 'image_url': {'url': before_url}},
                {'type': 'text', 'text': 'AFTER image:'},
                {'type': 'image_url', 'image_url': {'url': after_url}},
            ]},
        ]

        result = self._call_gpt(messages)
        if result is None:
            self.get_logger().error('GPT verify failed after all retries')
            response.complied = False
            response.reason = 'GPT verification failed'
            return response

        result = result.strip()
        if result.startswith('```'):
            result = result.split('\n', 1)[1] if '\n' in result else result[3:]
            if result.endswith('```'):
                result = result[:-3].strip()

        try:
            parsed = json.loads(result)
            response.complied = parsed.get('complied', False)
            response.reason = parsed.get('reason', '')
        except json.JSONDecodeError:
            self.get_logger().error(f'GPT returned invalid JSON: {result}')
            response.complied = False
            response.reason = 'Invalid GPT response'

        self.get_logger().info(f'Verify result: complied={response.complied}, reason={response.reason}')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = BridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
