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
CartPole launch — gz sim + the sync-gate plugin.

Toggleable add-ons (all opt-in, default off so the simple flow stays simple):
    use_foxglove:=true       Launch foxglove_bridge + ros_gz_bridge so you can
                             watch the run live in Foxglove Studio
                             (ws://localhost:8765).
    record_rosbag:=true      Launch `ros2 bag record -a -s mcap -o rosbags/cartpole_<ts>`
                             to capture every topic in MCAP format for offline
                             replay (also Foxglove-loadable).
    verbose:=true            Per-step debug prints in the env (default true).

The training script is intentionally separate — start the simulator with this
launch, then run `python training_scripts/train_cartpole_sb3.py` (or
training_scripts/train_cartpole_custom.py) in another terminal. Use
`cartpole_train.launch.py` for the all-in-one flow.
"""

import os
from datetime import datetime

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    AppendEnvironmentVariable,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PythonExpression

from launch_ros.actions import Node


# Where gz-sim 8 (Harmonic) looks for system plugin .so files. The
# python-system-loader plugin used by our worlds lives here when
# gz-harmonic is apt-installed. We prepend (not set) so a user with a
# custom plugin dir in their shell doesn't lose it.
GZ_PLUGIN_PATHS = (
    '/usr/lib/x86_64-linux-gnu/gz-sim-8/plugins:'
    '/usr/local/lib/gz-sim-8/plugins'
)


def generate_launch_description():
    gazebo_gymnasium_resources = get_package_share_directory('gazebo_gymnasium_resources')
    bringup_share = get_package_share_directory('gazebo_gymnasium_bringup')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world_path = os.path.join(gazebo_gymnasium_resources, 'worlds', 'cartpole.sdf')

    # --- Launch arguments ---

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
        description='Port foxglove_bridge listens on (default 8765).'
    )
    record_rosbag_arg = DeclareLaunchArgument(
        'record_rosbag', default_value='false',
        description='Record all topics to a timestamped MCAP rosbag.'
    )
    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description=(
            'Run gz sim server-only (no GUI). 5-10x training speedup; '
            'flip back to false for visual debugging.'
        ),
    )

    # --- Env vars consumed by the env class on the agent side ---

    # libgz-sim8 embeds Python 3.12. Pin PYTHONHOME so conda/pyenv on PATH
    # don't redirect away from /usr/lib/python3/dist-packages.
    pin_python_home = SetEnvironmentVariable('PYTHONHOME', '/usr')
    set_verbose_env = SetEnvironmentVariable(
        'GAZEBO_GYM_VERBOSE', LaunchConfiguration('verbose')
    )
    # Add gz-sim 8's standard plugin dirs to GZ_SIM_SYSTEM_PLUGIN_PATH so
    # python-system-loader is findable regardless of how the user's shell
    # is set up. Prepend so any custom user paths still take priority.
    prepend_gz_plugin_path = AppendEnvironmentVariable(
        'GZ_SIM_SYSTEM_PLUGIN_PATH', GZ_PLUGIN_PATHS,
        prepend=True,
    )
    # Also add the resources package's `plugins/` dir to PYTHONPATH so
    # PythonSystemLoader's `PyImport_ImportModule(<module_name>)` can
    # find our sync-gate modules (cartpole_learner, etc.).
    plugins_dir = os.path.join(gazebo_gymnasium_resources, 'plugins')
    prepend_pythonpath = AppendEnvironmentVariable(
        'PYTHONPATH', plugins_dir,
        prepend=True,
    )

    # --- gz sim ---

    gz_args = PythonExpression([
        "'-r ' + '", world_path, "' + (' -s' if '",
        LaunchConfiguration('headless'), "' == 'true' else '')"
    ])
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': gz_args}.items(),
    )

    # --- Foxglove block ---
    #
    # Bridge gz topics to ROS, then start foxglove_bridge so Foxglove Studio
    # can subscribe. Conditionally launched.
    #
    # Topic list lives in config/cartpole_bridge.yaml — keeping the
    # mappings in YAML lets us review the topic surface in isolation and
    # set per-topic QoS overrides (matters for image streams). Note:
    # /env/metrics is published by the env class directly as a
    # gazebo_gymnasium_msgs/EnvMetrics ROS topic, not bridged from gz,
    # so it doesn't appear in the YAML.
    cartpole_bridge_config = os.path.join(
        bringup_share, 'config', 'cartpole_bridge.yaml',
    )
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'config_file': cartpole_bridge_config}],
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

    # --- rosbag block ---
    #
    # `ros2 bag record -a -s mcap -o rosbags/cartpole_<timestamp>` —
    # the timestamp keeps successive runs from colliding. ros2 bag refuses to
    # overwrite an existing path. ExecuteProcess (not ComposableNode) because
    # `ros2 bag record` is a CLI tool, not a node we can compose.
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    bag_path = os.path.join('rosbags', f'cartpole_{ts}')
    record_rosbag_node = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-a', '-s', 'mcap', '-o', bag_path],
        output='screen',
        condition=IfCondition(LaunchConfiguration('record_rosbag')),
    )

    # Static map -> world transform. Pose/info bridge publishes link TFs
    # parented to "world"; this anchors Foxglove's 3D panel root.
    world_static_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        arguments=['--frame-id', 'map', '--child-frame-id', 'world'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_foxglove')),
    )

    return LaunchDescription([
        # Args
        verbose_arg,
        headless_arg,
        use_foxglove_arg,
        foxglove_port_arg,
        record_rosbag_arg,
        # Env vars
        pin_python_home,
        set_verbose_env,
        prepend_gz_plugin_path,
        prepend_pythonpath,
        # Processes
        gz_sim,
        ros_gz_bridge,
        foxglove_bridge_node,
        world_static_tf,
        record_rosbag_node,
    ])
