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

"""Sim-only launch for the line-following rover.

Pair with `train_line_follower_ppo.py` in another terminal (or use
`line_follower_train.launch.py` to do both in one shot).

Toggleable add-ons match the cartpole convention:
    verbose:=true | use_foxglove:=true | foxglove_port:=N | record_rosbag:=true
"""

import os
from datetime import datetime

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import IncludeLaunchDescription
from launch.actions import AppendEnvironmentVariable
from launch.actions import SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    resources = get_package_share_directory("gazebo_gymnasium_resources")
    bringup_share = get_package_share_directory("gazebo_gymnasium_bringup")
    ros_gz_sim = get_package_share_directory("ros_gz_sim")

    world_path = os.path.join(resources, "worlds", "line_follower.sdf")
    bridge_config = os.path.join(
        bringup_share, "config", "line_follower_bridge.yaml",
    )
    rover_urdf = os.path.join(resources, "models", "rover", "rover.urdf")

    verbose_arg = DeclareLaunchArgument("verbose", default_value="false")
    use_foxglove_arg = DeclareLaunchArgument("use_foxglove", default_value="false")
    foxglove_port_arg = DeclareLaunchArgument("foxglove_port", default_value="8765")
    record_rosbag_arg = DeclareLaunchArgument("record_rosbag", default_value="false")
    headless_arg = DeclareLaunchArgument(
        "headless", default_value="false",
        description="Run gz sim server-only (no GUI). Big training speedup.",
    )

    pin_python_home = SetEnvironmentVariable("PYTHONHOME", "/usr")
    set_verbose_env = SetEnvironmentVariable(
        "GAZEBO_GYM_VERBOSE", LaunchConfiguration("verbose"),
    )
    # See cartpole.launch.py for the rationale on these.
    prepend_gz_plugin_path = AppendEnvironmentVariable(
        "GZ_SIM_SYSTEM_PLUGIN_PATH",
        "/usr/lib/x86_64-linux-gnu/gz-sim-8/plugins:"
        "/usr/local/lib/gz-sim-8/plugins",
        prepend=True,
    )
    prepend_pythonpath = AppendEnvironmentVariable(
        "PYTHONPATH", os.path.join(resources, "plugins"), prepend=True,
    )

    gz_args = PythonExpression([
        "'-r ' + '", world_path, "' + (' -s' if '",
        LaunchConfiguration("headless"), "' == 'true' else '')"
    ])
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": gz_args}.items(),
    )

    # ros_gz_bridge driven by config/line_follower_bridge.yaml. The YAML
    # includes the camera image with BEST_EFFORT QoS so Foxglove can
    # subscribe (default RELIABLE would silently drop every frame).
    ros_gz_bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge",
        parameters=[{"config_file": bridge_config}],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_foxglove")),
    )

    foxglove_bridge_node = Node(
        package="foxglove_bridge", executable="foxglove_bridge",
        parameters=[{"port": LaunchConfiguration("foxglove_port")}],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_foxglove")),
    )

    # robot_state_publisher: serves the rover URDF as /robot_description
    # AND publishes link transforms on /tf using the joint states bridged
    # in from gz. Without it, Foxglove's 3D panel sees no internal-link
    # transforms; just the world->rover model pose.
    with open(rover_urdf) as f:
        robot_description = f.read()
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": True,
        }],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_foxglove")),
    )

    # Static transform: world (gz's root) -> map (Foxglove default frame).
    # Identity transform — just ensures the "world" frame exists in TF so
    # Foxglove panels with display_frame=world have something to anchor to.
    world_static_tf = Node(
        package="tf2_ros", executable="static_transform_publisher",
        arguments=["--frame-id", "map", "--child-frame-id", "world"],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_foxglove")),
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_path = os.path.join("rosbags", f"line_follower_{ts}")
    record_rosbag_node = ExecuteProcess(
        cmd=["ros2", "bag", "record", "-a", "-s", "mcap", "-o", bag_path],
        output="screen",
        condition=IfCondition(LaunchConfiguration("record_rosbag")),
    )

    return LaunchDescription([
        verbose_arg, headless_arg, use_foxglove_arg, foxglove_port_arg,
        record_rosbag_arg,
        pin_python_home, set_verbose_env,
        prepend_gz_plugin_path, prepend_pythonpath,
        gz_sim,
        ros_gz_bridge, foxglove_bridge_node,
        robot_state_publisher_node, world_static_tf,
        record_rosbag_node,
    ])
