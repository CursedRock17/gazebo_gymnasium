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

"""Multi-agent CartPole launch — N cartpoles spawned into one gz sim.

Starts an empty world named `cartpole_multi`, then dynamically spawns
N cartpoles via the ros_gz_sim Create service. Sequential spawning
sidesteps the world-load race that capped static `<include>`-based
worlds at 4 cartpoles regardless of how many were declared.

Arguments:
    n_agents:=4      # any positive integer; no pre-generated SDFs needed.
    headless:=true   # gz sim server-only — big training speedup.
    verbose:=false   # per-step debug prints from the vec env.
"""

import os
from datetime import datetime

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import IncludeLaunchDescription
from launch.actions import SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    resources = get_package_share_directory("gazebo_gymnasium_resources")
    bringup = get_package_share_directory("gazebo_gymnasium_bringup")
    ros_gz_sim = get_package_share_directory("ros_gz_sim")

    n_agents_arg = DeclareLaunchArgument(
        "n_agents", default_value="4",
        description="Number of cartpoles to spawn. Any positive integer.",
    )
    headless_arg = DeclareLaunchArgument(
        "headless", default_value="true",
        description="Run gz sim server-only — big training speedup.",
    )
    verbose_arg = DeclareLaunchArgument("verbose", default_value="false")
    record_rosbag_arg = DeclareLaunchArgument(
        "record_rosbag", default_value="false")

    pin_python_home = SetEnvironmentVariable("PYTHONHOME", "/usr")
    set_verbose_env = SetEnvironmentVariable(
        "GAZEBO_GYM_VERBOSE", LaunchConfiguration("verbose"),
    )
    prepend_gz_plugin_path = AppendEnvironmentVariable(
        "GZ_SIM_SYSTEM_PLUGIN_PATH",
        "/usr/lib/x86_64-linux-gnu/gz-sim-8/plugins:"
        "/usr/local/lib/gz-sim-8/plugins",
        prepend=True,
    )
    prepend_pythonpath = AppendEnvironmentVariable(
        "PYTHONPATH", os.path.join(resources, "plugins"), prepend=True,
    )

    world_path = os.path.join(resources, "worlds", "cartpole_multi.sdf")
    gz_args = PythonExpression([
        "'-r " + world_path + "' + (' -s' if '",
        LaunchConfiguration("headless"),
        "' == 'true' else '')",
    ])
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": gz_args}.items(),
    )

    spawner_script = os.path.join(bringup, "scripts",
                                  "spawn_multi_cartpoles.py")
    spawner = ExecuteProcess(
        cmd=["python3", spawner_script,
             "--n-agents", LaunchConfiguration("n_agents")],
        output="screen",
    )

    world_static_tf = Node(
        package="tf2_ros", executable="static_transform_publisher",
        arguments=["--frame-id", "map", "--child-frame-id", "world"],
        output="screen",
        condition=IfCondition(LaunchConfiguration("verbose")),
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_path = os.path.join("rosbags", f"cartpole_multi_{ts}")
    record_rosbag_node = ExecuteProcess(
        cmd=["ros2", "bag", "record", "-a", "-s", "mcap", "-o", bag_path],
        output="screen",
        condition=IfCondition(LaunchConfiguration("record_rosbag")),
    )

    return LaunchDescription([
        n_agents_arg, headless_arg, verbose_arg, record_rosbag_arg,
        pin_python_home, set_verbose_env,
        prepend_gz_plugin_path, prepend_pythonpath,
        gz_sim, spawner, world_static_tf, record_rosbag_node,
    ])
