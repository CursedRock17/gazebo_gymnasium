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

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution

from launch_ros.actions import Node


# Must create a default launch file for users to access diff drive example
def generate_launch_description():
    # Grab Internal Libraries
    gazebo_gymnasium_resources = get_package_share_directory(
      'gazebo_gymnasium_resources')
    # Create Easier Package-internal paths
    gazebo_gymnasium_bringup = get_package_share_directory(
      'gazebo_gymnasium_bringup')
    # Grab External Libraries
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # Load the diff drive bot SDF file from resources
    diff_drive_sdf = os.path.join(
        gazebo_gymnasium_resources, 'models', 'diff_drive', 'model.sdf')
    with open(diff_drive_sdf, 'r') as infp:
        robot_desc = infp.read()

    # Setup to launch the simulator and Gazebo world
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': PathJoinSubstitution([
            gazebo_gymnasium_resources,
            'worlds',
            'diff_drive.sdf'
        ])}.items(),
    )

    # Takes the description and joint angles as inputs
    # Then publishes the 3D poses of the robot links
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[
            {'use_sim_time': True},
            {'robot_description': robot_desc},
        ]
    )

    # Visualize in RViz
    rviz = Node(
       package='rviz2',
       executable='rviz2',
       arguments=['-d', os.path.join(
            gazebo_gymnasium_bringup, 'config', 'diff_drive.rviz')],
    )

    # ROS2 -> Gazebo bridge to allow constant communication - converts from
    # ROS msg types to Gazebo msg types
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': os.path.join(
                gazebo_gymnasium_bringup,
                'config', 'gazebo_gymnasium_diff_drive_bringup.yaml'),
            'qos_overrides./tf_static.publisher.durability': 'transient_local'
        }],
        output='screen'
    )

    teleop = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        prefix=["gnome-terminal", " --"],
        remappings=[('cmd_vel', '/diff_drive/cmd_vel')],
    )

    return LaunchDescription([
        gz_sim,
        #ros_gz_bridge,
        #robot_state_publisher,
        #rviz,
        #teleop
    ])
