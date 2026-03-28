import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Launch arguments
    map_arg = DeclareLaunchArgument(
        'map',
        default_value=os.path.expanduser('~/map.yaml'),
        description='Path to map yaml file'
    )

    # TurtleBot3 Navigation2
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('turtlebot3_navigation2'),
                'launch',
                'navigation2.launch.py'
            )
        ),
        launch_arguments={
            'map': LaunchConfiguration('map'),
            'use_sim_time': 'false',
        }.items(),
    )

    # Mission3 nodes
    capture_node = Node(
        package='mission3',
        executable='capture_node',
        name='capture_node',
        output='screen',
    )

    bridge_node = Node(
        package='mission3',
        executable='bridge_node',
        name='bridge_node',
        output='screen',
        additional_env={'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY', '')},
    )

    point_node = Node(
        package='mission3',
        executable='point_node',
        name='point_node',
        output='screen',
    )

    mission_manager_node = Node(
        package='mission3',
        executable='mission_manager_node',
        name='mission_manager_node',
        output='screen',
    )

    return LaunchDescription([
        map_arg,
        nav2_launch,
        capture_node,
        bridge_node,
        point_node,
        mission_manager_node,
    ])
