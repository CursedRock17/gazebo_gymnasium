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
CartPole one-command launch — gz sim AND the training script together.

Picks up everything cartpole.launch.py supports (verbose, use_foxglove,
record_rosbag, foxglove_port) and adds:

    trainer:=sb3|custom      Which training script to run as a child process.
                             Defaults to sb3.
    timesteps:=N             total_timesteps for the trainer (passed via env
                             var; both train_cartpole_*.py read GAZEBO_GYM_TIMESTEPS).
    delay_train:=SECONDS     Sleep this long before starting the trainer so
                             gz sim has time to advertise services. Default 5.

Usage:
    ros2 launch gazebo_gymnasium_bringup cartpole_train.launch.py
    ros2 launch gazebo_gymnasium_bringup cartpole_train.launch.py trainer:=custom
    ros2 launch gazebo_gymnasium_bringup cartpole_train.launch.py use_foxglove:=true record_rosbag:=true
"""

import os
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _find_project_root() -> str:
    """The training scripts live in <root>/training_scripts/. From this
    launch file we're at <root>/install/<pkg>/share/<pkg>/launch — walk up
    until we find the marker, fall back to a sensible relative guess."""
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / 'training_scripts' / 'train_cartpole_sb3.py').exists():
            return str(ancestor)
    # Source-tree fallback (when the launch file hasn't been installed).
    return str(here.parents[6]) if len(here.parents) > 6 else str(here.parent)


def generate_launch_description():
    project_root = _find_project_root()

    bringup_share = get_package_share_directory('gazebo_gymnasium_bringup')
    sim_launch = os.path.join(bringup_share, 'launch', 'cartpole.launch.py')

    # --- Args ---

    trainer_arg = DeclareLaunchArgument(
        'trainer', default_value='sb3', choices=['sb3', 'custom'],
        description='Which training script to run (sb3 or custom).'
    )
    timesteps_arg = DeclareLaunchArgument(
        'timesteps', default_value='200000',
        description='total_timesteps passed to the trainer via GAZEBO_GYM_TIMESTEPS.'
    )
    delay_train_arg = DeclareLaunchArgument(
        'delay_train', default_value='5.0',
        description='Seconds to wait before starting the trainer so gz sim is ready.'
    )
    # Pass-through args mirrored from cartpole.launch.py so users can set them
    # here too. Defaults match the underlying launch.
    verbose_arg = DeclareLaunchArgument('verbose', default_value='true')
    use_foxglove_arg = DeclareLaunchArgument('use_foxglove', default_value='false')
    foxglove_port_arg = DeclareLaunchArgument('foxglove_port', default_value='8765')
    record_rosbag_arg = DeclareLaunchArgument('record_rosbag', default_value='false')

    # --- Env vars for the trainer subprocess ---
    set_timesteps_env = SetEnvironmentVariable(
        'GAZEBO_GYM_TIMESTEPS', LaunchConfiguration('timesteps')
    )

    # --- Include the sim launch ---
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch),
        launch_arguments={
            'verbose': LaunchConfiguration('verbose'),
            'use_foxglove': LaunchConfiguration('use_foxglove'),
            'foxglove_port': LaunchConfiguration('foxglove_port'),
            'record_rosbag': LaunchConfiguration('record_rosbag'),
        }.items(),
    )

    # --- Trainer subprocess ---
    # Use the system python explicitly so we don't accidentally inherit a
    # pyenv/conda interpreter that doesn't have the gz bindings on sys.path.
    sb3_script = os.path.join(project_root, 'training_scripts', 'train_cartpole_sb3.py')
    custom_script = os.path.join(project_root, 'training_scripts', 'train_cartpole_custom.py')

    sb3_trainer = ExecuteProcess(
        cmd=[sys.executable, sb3_script],
        output='screen',
        condition=LaunchConfigurationEquals('trainer', 'sb3'),
    )
    custom_trainer = ExecuteProcess(
        cmd=[sys.executable, custom_script],
        output='screen',
        condition=LaunchConfigurationEquals('trainer', 'custom'),
    )

    # Delay the trainer so gz sim has time to advertise /world/<name>/control
    # before the env's WorldController tries to call it. The WorldController
    # already retries internally, but a small delay keeps the logs cleaner.
    delayed_sb3 = TimerAction(
        period=LaunchConfiguration('delay_train'),
        actions=[sb3_trainer],
    )
    delayed_custom = TimerAction(
        period=LaunchConfiguration('delay_train'),
        actions=[custom_trainer],
    )

    return LaunchDescription([
        trainer_arg,
        timesteps_arg,
        delay_train_arg,
        verbose_arg,
        use_foxglove_arg,
        foxglove_port_arg,
        record_rosbag_arg,
        set_timesteps_env,
        sim,
        delayed_sb3,
        delayed_custom,
    ])
