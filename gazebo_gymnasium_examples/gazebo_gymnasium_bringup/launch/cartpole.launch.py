import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution


# Must create a default launch file for users to access diff drive example
def generate_launch_description():
    # Grab Internal Libraries
    gazebo_gymnasium_resources = get_package_share_directory(
      'gazebo_gymnasium_resources')
    # Grab External Libraries
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # Setup to launch the simulator and Gazebo world
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': PathJoinSubstitution([
            gazebo_gymnasium_resources,
            'worlds',
            'cartpole.sdf'
        ])}.items(),
    )

    return LaunchDescription([
        gz_sim,
    ])
