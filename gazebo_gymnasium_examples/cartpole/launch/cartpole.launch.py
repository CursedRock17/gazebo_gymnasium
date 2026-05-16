"""Launch Gazebo with the CartPole world (paused, ready for external RL control)."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    examples_share = get_package_share_directory('gazebo_gymnasium_examples')
    ros_gz_sim_share = get_package_share_directory('ros_gz_sim')

    world_path = PathJoinSubstitution([
        examples_share, 'cartpole', 'worlds', 'cartpole.sdf'
    ])

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, 'launch', 'gz_sim.launch.py')
        ),
        # No -r flag: Gazebo starts paused by default, giving the external
        # training script deterministic control over simulation stepping.
        launch_arguments={'gz_args': world_path}.items(),
    )

    return LaunchDescription([gz_sim])
