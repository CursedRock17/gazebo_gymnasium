"""Launch Gazebo with the CartPole world (paused, ready for external RL control).

ROS 2 topic bridge
------------------
A ros_gz_bridge node is included so you can inspect and interact with the
simulation using standard ROS 2 tools:

  # Simulation clock (lets ros2 topic hz and stamp-aware tools work correctly)
  ros2 topic echo /clock

  # Cart command — publish a target position (meters) from ROS 2:
  ros2 topic pub /model/cartpole/joint/slider_to_cart/0/cmd_pos std_msgs/msg/Float64 "data: 0.3"

  # Joint states from Gazebo (gz.msgs.Model bridged to sensor_msgs/JointState):
  ros2 topic echo /world/cartpole/model/cartpole/joint_state

Note: the training script drives Gazebo directly via gz.transport — the bridge is
for observation and debugging only, not required for RL training.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node


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

    # Bridge gz topics → ROS 2 for debugging and introspection.
    # The training script uses gz.transport directly; this bridge is optional.
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # Simulation clock — required for ros2 topic hz and time-aware tools
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            # Cart command: ROS 2 → gz (lets you manually command the cart)
            '/model/cartpole/joint/slider_to_cart/0/cmd_pos'
            '@std_msgs/msg/Float64]gz.msgs.Double',
            # Joint states: gz → ROS 2
            '/world/cartpole/model/cartpole/joint_state'
            '@sensor_msgs/msg/JointState[gz.msgs.Model',
        ],
        output='screen',
    )

    return LaunchDescription([gz_sim, bridge])
