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
"""Batched-harness CartPole launch — N bare cartpoles, one in-sim harness plugin.

Alternative to cartpole_multi.launch.py that uses the MultiAgentHarness backend:
the harness plugin owns actuation + sensing via the ECM and exposes O(1)
transport (/rl/actions, /rl/reset, /rl/observations). Reset is in place (no
respawn), so no delete/create race.

Train against it with the harness client:
    python training_scripts/train.py --agent cartpole --n_agents 16 --backend harness

Arguments: n_agents:=4  headless:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import IncludeLaunchDescription
from launch.actions import SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PythonExpression


def generate_launch_description():
    resources = get_package_share_directory("gazebo_gymnasium_resources")
    bringup = get_package_share_directory("gazebo_gymnasium_bringup")
    ros_gz_sim = get_package_share_directory("ros_gz_sim")

    n_agents_arg = DeclareLaunchArgument("n_agents", default_value="4")
    headless_arg = DeclareLaunchArgument("headless", default_value="true")

    # The harness plugin reads N + agent from these (single fixed world SDF).
    set_n = SetEnvironmentVariable("GAZEBO_GYM_N_AGENTS", LaunchConfiguration("n_agents"))
    set_agent = SetEnvironmentVariable("GAZEBO_GYM_AGENT", "cartpole")
    # Make the harness plugin module importable by the gz server.
    prepend_pythonpath = AppendEnvironmentVariable(
        "PYTHONPATH", os.path.join(resources, "plugins"), prepend=True
    )
    prepend_gz_plugin_path = AppendEnvironmentVariable(
        "GZ_SIM_SYSTEM_PLUGIN_PATH",
        "/usr/lib/x86_64-linux-gnu/gz-sim-8/plugins:/usr/local/lib/gz-sim-8/plugins",
        prepend=True,
    )

    world_path = os.path.join(resources, "worlds", "cartpole_harness.sdf")
    gz_args = PythonExpression(
        [
            "'-r " + world_path + "' + (' -s' if '",
            LaunchConfiguration("headless"),
            "' == 'true' else '')",
        ]
    )
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": gz_args}.items(),
    )

    spawner = ExecuteProcess(
        cmd=[
            "python3",
            os.path.join(bringup, "scripts", "spawn_multi_cartpoles.py"),
            "--n-agents",
            LaunchConfiguration("n_agents"),
            # must match <world name=...> in cartpole_harness.sdf, otherwise
            # `create -world ...` targets a world that isn't running.
            "--world",
            "cartpole_harness",
            # matches the cartpole AgentSpec.spawn_z — force actuation needs
            # the cart clear of the ground plane (contact friction pins it).
            "--spawn-z",
            "0.6",
            "--model-uri",
            "package://gazebo_gymnasium_resources/models/cartpole_bare",
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            n_agents_arg,
            headless_arg,
            set_n,
            set_agent,
            prepend_pythonpath,
            prepend_gz_plugin_path,
            gz_sim,
            spawner,
        ]
    )
