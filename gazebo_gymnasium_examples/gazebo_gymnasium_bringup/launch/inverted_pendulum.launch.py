# Copyright 2026 Lucas Wendland
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
InvertedPendulum launch — gz sim + the sync-gate plugin.

Same toggleable add-ons as cartpole.launch.py, plus:
    headless:=true           gz sim server-only (no GUI). Massive speedup
                             during training — flip this on once you've
                             confirmed the sim looks right.
"""

import os
from datetime import datetime

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    AppendEnvironmentVariable,
    SetEnvironmentVariable,
)
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
    gazebo_gymnasium_resources = get_package_share_directory('gazebo_gymnasium_resources')
    bringup_share = get_package_share_directory('gazebo_gymnasium_bringup')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world_path = os.path.join(gazebo_gymnasium_resources, 'worlds', 'inverted_pendulum.sdf')

    # Args
    verbose_arg = DeclareLaunchArgument(
        'verbose', default_value='true',
        description='Per-step debug logging in the env (true/false).'
    )
    use_foxglove_arg = DeclareLaunchArgument(
        'use_foxglove', default_value='false',
        description='Launch foxglove_bridge + ros_gz_bridge for live viewing.'
    )
    foxglove_port_arg = DeclareLaunchArgument(
        'foxglove_port', default_value='8765',
    )
    record_rosbag_arg = DeclareLaunchArgument(
        'record_rosbag', default_value='false',
    )
    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description=(
            'Run gz sim server-only (no GUI window). Typically 5-10x faster '
            'than GUI mode since the GL renderer no longer throttles physics. '
            'Use during training; flip back to false for visual debugging.'
        ),
    )

    # Env vars
    pin_python_home = SetEnvironmentVariable('PYTHONHOME', '/usr')
    set_verbose_env = SetEnvironmentVariable(
        'GAZEBO_GYM_VERBOSE', LaunchConfiguration('verbose')
    )
    prepend_gz_plugin_path = AppendEnvironmentVariable(
        'GZ_SIM_SYSTEM_PLUGIN_PATH', GZ_PLUGIN_PATHS,
        prepend=True,
    )
    plugins_dir = os.path.join(gazebo_gymnasium_resources, 'plugins')
    prepend_pythonpath = AppendEnvironmentVariable(
        'PYTHONPATH', plugins_dir,
        prepend=True,
    )

    # gz sim. When headless:=true, append `-s` to the gz CLI args so the
    # server runs without spawning a GUI window — biggest training-speed
    # lever in the project.
    gz_args = PythonExpression([
        "'-r ' + '", world_path, "' + (' -s' if '",
        LaunchConfiguration('headless'), "' == 'true' else '')"
    ])
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': gz_args}.items(),
    )

    # Static map -> world transform. The pose/info bridge publishes link
    # transforms with parent="world", so Foxglove's 3D panel needs a "world"
    # frame in TF to anchor everything. The identity transform from "map"
    # to "world" gives Foxglove a root display_frame to point at.
    world_static_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        arguments=['--frame-id', 'map', '--child-frame-id', 'world'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_foxglove')),
    )

    # ros_gz_bridge driven by config/inverted_pendulum_bridge.yaml. See
    # cartpole.launch.py for the rationale behind YAML-driven configs.
    bridge_config = os.path.join(
        bringup_share, 'config', 'inverted_pendulum_bridge.yaml',
    )
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'config_file': bridge_config}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_foxglove')),
    )
    foxglove_bridge_node = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        parameters=[{'port': LaunchConfiguration('foxglove_port')}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_foxglove')),
    )

    # rosbag — distinct name prefix so it doesn't collide with cartpole bags.
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    bag_path = os.path.join('rosbags', f'inv_pendulum_{ts}')
    record_rosbag_node = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-a', '-s', 'mcap', '-o', bag_path],
        output='screen',
        condition=IfCondition(LaunchConfiguration('record_rosbag')),
    )

    return LaunchDescription([
        verbose_arg,
        use_foxglove_arg,
        foxglove_port_arg,
        record_rosbag_arg,
        headless_arg,
        pin_python_home,
        set_verbose_env,
        prepend_gz_plugin_path,
        prepend_pythonpath,
        gz_sim,
        ros_gz_bridge,
        foxglove_bridge_node,
        world_static_tf,
        record_rosbag_node,
    ])
