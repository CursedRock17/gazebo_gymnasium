"""Full InvertedPendulum training stack.

Usage:
    ros2 launch gazebo_gymnasium_examples inverted_pendulum_full.launch.py
    ros2 launch gazebo_gymnasium_examples inverted_pendulum_full.launch.py headless:=true
    ros2 launch gazebo_gymnasium_examples inverted_pendulum_full.launch.py timesteps:=500000
"""
import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    timesteps_arg = DeclareLaunchArgument(
        "timesteps", default_value="200000",
        description="Total PPO training timesteps")

    headless_arg = DeclareLaunchArgument(
        "headless", default_value="false",
        description="Run Gazebo without GUI for faster training")

    timesteps = LaunchConfiguration("timesteps")
    headless = LaunchConfiguration("headless")

    set_gz_partition = SetEnvironmentVariable("GZ_PARTITION", "0")

    examples_share = get_package_share_directory("gazebo_gymnasium_examples")
    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")

    world_path = PathJoinSubstitution([
        examples_share, "inverted_pendulum", "worlds", "inverted_pendulum.sdf"
    ])

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": PythonExpression([
                "'-s ' + '", world_path, "'",
                " if '", headless, "' == 'true' else '", world_path, "'",
            ]),
        }.items(),
    )

    bridge = TimerAction(
        period=3.0,
        actions=[Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="inverted_pendulum_bridge",
            arguments=[
                "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
                "/world/inverted_pendulum/model/inverted_pendulum/joint_state"
                "@sensor_msgs/msg/JointState[gz.msgs.Model",
            ],
            output="screen",
        )],
    )

    script_dir = os.path.join(
        get_package_prefix("gazebo_gymnasium_examples"),
        "lib", "gazebo_gymnasium_examples", "inverted_pendulum",
    )
    train_script = os.path.join(script_dir, "train_ppo.py")

    train = TimerAction(
        period=6.0,
        actions=[ExecuteProcess(
            cmd=["python3", train_script, "--timesteps", timesteps],
            additional_env={"GZ_PARTITION": "0"},
            cwd="/tmp",
            output="screen",
        )],
    )

    return LaunchDescription([
        set_gz_partition,
        timesteps_arg,
        headless_arg,
        gz_sim,
        bridge,
        train,
    ])
