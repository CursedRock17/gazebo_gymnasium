"""
Auto-generated launch for ant. Same toggleable add-ons as cartpole.launch.py:
    verbose:=true | use_foxglove:=true | foxglove_port:=N | record_rosbag:=true
"""

import os
from datetime import datetime

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.actions import AppendEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PythonExpression

from launch_ros.actions import Node


# See cartpole.launch.py for the rationale on these env vars.
GZ_PLUGIN_PATHS = (
    '/usr/lib/x86_64-linux-gnu/gz-sim-8/plugins:'
    '/usr/local/lib/gz-sim-8/plugins'
)


def generate_launch_description():
    resources = get_package_share_directory('gazebo_gymnasium_resources')
    bringup_share = get_package_share_directory('gazebo_gymnasium_bringup')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world_path = os.path.join(resources, 'worlds', 'ant.sdf')

    verbose_arg = DeclareLaunchArgument('verbose', default_value='true')
    use_foxglove_arg = DeclareLaunchArgument('use_foxglove', default_value='false')
    foxglove_port_arg = DeclareLaunchArgument('foxglove_port', default_value='8765')
    record_rosbag_arg = DeclareLaunchArgument('record_rosbag', default_value='false')
    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description='Run gz sim server-only (no GUI). Big training speedup.',
    )

    pin_python_home = SetEnvironmentVariable('PYTHONHOME', '/usr')
    set_verbose_env = SetEnvironmentVariable('GAZEBO_GYM_VERBOSE',
                                              LaunchConfiguration('verbose'))
    prepend_gz_plugin_path = AppendEnvironmentVariable(
        'GZ_SIM_SYSTEM_PLUGIN_PATH', GZ_PLUGIN_PATHS,
        prepend=True,
    )
    prepend_pythonpath = AppendEnvironmentVariable(
        'PYTHONPATH', os.path.join(resources, 'plugins'), prepend=True,
    )

    # `-s` when headless=true so gz sim runs server-only.
    gz_args = PythonExpression([
        "'-r ' + '", world_path, "' + (' -s' if '",
        LaunchConfiguration('headless'), "' == 'true' else '')"
    ])
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': gz_args}.items(),
    )

    # ros_gz_bridge driven by config/ant_bridge.yaml. Topic list
    # lives outside the launch so each env's bridge surface can be
    # reviewed (and have per-topic QoS overrides set) in one place.
    bridge_config = os.path.join(
        bringup_share, 'config', 'ant_bridge.yaml',
    )
    ros_gz_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        parameters=[{'config_file': bridge_config}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_foxglove')),
    )
    foxglove_bridge_node = Node(
        package='foxglove_bridge', executable='foxglove_bridge',
        parameters=[{'port': LaunchConfiguration('foxglove_port')}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_foxglove')),
    )

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    bag_path = os.path.join('rosbags', f'ant_{ts}')
    record_rosbag_node = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-a', '-s', 'mcap', '-o', bag_path],
        output='screen',
        condition=IfCondition(LaunchConfiguration('record_rosbag')),
    )

    # Static map -> world transform so Foxglove's 3D panel has a root
    # frame to anchor to (pose/info publishes link transforms with
    # parent_frame_id="world").
    world_static_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        arguments=['--frame-id', 'map', '--child-frame-id', 'world'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_foxglove')),
    )

    return LaunchDescription([
        verbose_arg, headless_arg, use_foxglove_arg, foxglove_port_arg,
        record_rosbag_arg,
        pin_python_home, set_verbose_env,
        prepend_gz_plugin_path, prepend_pythonpath,
        gz_sim, ros_gz_bridge, foxglove_bridge_node,
        world_static_tf, record_rosbag_node,
    ])
