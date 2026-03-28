from setuptools import setup

package_name = 'mission3'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', [
            'config/waypoints.yaml',
            'config/prompts.yaml',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='team',
    maintainer_email='team@robocup.org',
    description='RoboCup@Home Mission 3 - Stickler for the Rules',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'capture_node = mission3.capture_node:main',
            'bridge_node = mission3.bridge_node:main',
        ],
    },
)
