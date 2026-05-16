"""Launch Gazebo and the SB3 PPO training script together."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
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
        launch_arguments={'gz_args': ['-r -p ', world_path]}.items(),
    )

    train_script = os.path.join(
        get_package_share_directory('gazebo_gymnasium_examples'),
        '..', '..', 'lib', 'gazebo_gymnasium_examples', 'cartpole', 'train_sb3.py'
    )

    # Delay the training script to give Gazebo time to fully start
    train = TimerAction(
        period=3.0,
        actions=[ExecuteProcess(cmd=['python3', train_script], output='screen')]
    )

    return LaunchDescription([gz_sim, train])
