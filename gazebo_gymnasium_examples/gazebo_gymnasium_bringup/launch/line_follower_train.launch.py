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

"""All-in-one launch: gz sim + line-follower PPO trainer + Foxglove + rosbag.

Bundles every process needed for an end-to-end training run into one
launch. Bridge mappings live in config/line_follower_bridge.yaml (and
include the BEST_EFFORT QoS override the camera image needs).

Arguments (all optional):
    verbose:=false       — per-step debug logging in the env
    foxglove:=true       — enable the Foxglove + bridge stack
    foxglove_port:=8765  — port foxglove_bridge listens on
    timesteps:=10000     — total_timesteps passed to the trainer via
                           the GAZEBO_GYM_TIMESTEPS env var
    ros2bag:=false       — also record every topic to a timestamped MCAP
    namespace:=/         — prefix applied to ROS nodes (does NOT remap
                           the gz topics — those are bridged at fixed
                           names by the YAML config)

Example:
    ros2 launch gazebo_gymnasium_bringup line_follower_train.launch.py \\
        timesteps:=50000 ros2bag:=true
"""

import os
import sys
from datetime import datetime
from pathlib import Path

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
from launch_ros.actions import Node


def _find_project_root() -> str:
    """Walk up from the installed launch file until we find training_scripts/.

    From here we're at `<root>/install/<pkg>/share/<pkg>/launch`. Walk up
    until `training_scripts/train_line_follower_ppo.py` exists, then return
    that root path.
    """
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "training_scripts" / "train_line_follower_ppo.py").exists():
            return str(ancestor)
    return str(here.parents[6]) if len(here.parents) > 6 else str(here.parent)


def generate_launch_description():
    project_root = _find_project_root()
    resources = get_package_share_directory("gazebo_gymnasium_resources")
    bringup_share = get_package_share_directory("gazebo_gymnasium_bringup")
    ros_gz_sim = get_package_share_directory("ros_gz_sim")
    world_path = os.path.join(resources, "worlds", "line_follower.sdf")
    bridge_config = os.path.join(
        bringup_share, "config", "line_follower_bridge.yaml",
    )
    rover_urdf = os.path.join(resources, "models", "rover", "rover.urdf")

    # --- Arguments ---
    verbose_arg = DeclareLaunchArgument("verbose", default_value="false")
    foxglove_arg = DeclareLaunchArgument("foxglove", default_value="true")
    foxglove_port_arg = DeclareLaunchArgument("foxglove_port", default_value="8765")
    timesteps_arg = DeclareLaunchArgument("timesteps", default_value="10000")
    ros2bag_arg = DeclareLaunchArgument("ros2bag", default_value="false")
    namespace_arg = DeclareLaunchArgument("namespace", default_value="/")

    # --- Env vars consumed by the env class + trainer ---
    pin_python_home = SetEnvironmentVariable("PYTHONHOME", "/usr")
    set_verbose_env = SetEnvironmentVariable(
        "GAZEBO_GYM_VERBOSE", LaunchConfiguration("verbose"),
    )
    set_timesteps_env = SetEnvironmentVariable(
        "GAZEBO_GYM_TIMESTEPS", LaunchConfiguration("timesteps"),
    )
    # See cartpole.launch.py for the rationale.
    prepend_gz_plugin_path = AppendEnvironmentVariable(
        "GZ_SIM_SYSTEM_PLUGIN_PATH",
        "/usr/lib/x86_64-linux-gnu/gz-sim-8/plugins:"
        "/usr/local/lib/gz-sim-8/plugins",
        prepend=True,
    )
    prepend_pythonpath = AppendEnvironmentVariable(
        "PYTHONPATH", os.path.join(resources, "plugins"), prepend=True,
    )

    # --- gz sim ---
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": "-r " + world_path}.items(),
    )

    # --- ros_gz_bridge ---
    # Single Node, YAML-driven. Earlier revisions of this file ran the
    # bridge as a composable node inside a ComposableNodeContainer, but
    # ros_gz_bridge's composable variant has a different (less complete)
    # config-file code path than the standalone parameter_bridge — easier
    # to use the standalone form and let the namespacing happen via the
    # foxglove_bridge namespace argument.
    ros_gz_bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge",
        parameters=[{"config_file": bridge_config}],
        output="screen",
        condition=IfCondition(LaunchConfiguration("foxglove")),
    )

    foxglove_bridge = Node(
        package="foxglove_bridge", executable="foxglove_bridge",
        namespace=LaunchConfiguration("namespace"),
        parameters=[{"port": LaunchConfiguration("foxglove_port")}],
        output="screen",
        condition=IfCondition(LaunchConfiguration("foxglove")),
    )

    # --- robot_state_publisher: rover URDF -> /robot_description + /tf
    # for internal links. Without this, Foxglove's 3D panel sees the
    # world->rover model transform but no internal joints, so the wheels
    # look unattached to the chassis. -->
    with open(rover_urdf) as f:
        robot_description = f.read()
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": True,
        }],
        output="screen",
        condition=IfCondition(LaunchConfiguration("foxglove")),
    )

    # --- Static map -> world transform: gives Foxglove a root TF frame
    # so panels with display_frame=map have something to anchor to. The
    # rover ends up at map -> world -> rover (via pose/info bridge) ->
    # base_link (via robot_state_publisher). -->
    world_static_tf = Node(
        package="tf2_ros", executable="static_transform_publisher",
        arguments=["--frame-id", "map", "--child-frame-id", "world"],
        output="screen",
        condition=IfCondition(LaunchConfiguration("foxglove")),
    )

    # --- Trainer subprocess ---
    train_script = os.path.join(project_root, "training_scripts", "train_line_follower_ppo.py")
    trainer = ExecuteProcess(
        cmd=[sys.executable, train_script],
        output="screen",
    )

    # --- Rosbag (opt-in) ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_path = os.path.join("rosbags", f"line_follower_{ts}")
    record_rosbag = ExecuteProcess(
        cmd=["ros2", "bag", "record", "-a", "-s", "mcap", "-o", bag_path],
        output="screen",
        condition=IfCondition(LaunchConfiguration("ros2bag")),
    )

    return LaunchDescription([
        verbose_arg, foxglove_arg, foxglove_port_arg, timesteps_arg,
        ros2bag_arg, namespace_arg,
        pin_python_home, set_verbose_env, set_timesteps_env,
        prepend_gz_plugin_path, prepend_pythonpath,
        gz_sim,
        ros_gz_bridge, foxglove_bridge,
        robot_state_publisher, world_static_tf,
        trainer,
        record_rosbag,
    ])
