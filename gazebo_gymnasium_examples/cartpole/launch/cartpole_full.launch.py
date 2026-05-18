"""Full CartPole training stack.

Launches in sequence:
  1. Gazebo with the CartPole world (GZ_PARTITION=0 forced for all processes)
  2. ros_gz_bridge (clock, joint states)
  3. foxglove_bridge — optional, default OFF (foxglove:=true to enable)
  4. rosbag2 recorder — optional, default OFF (rosbag:=true to enable)
  5. SB3 training script (PPO or A2C, discrete or continuous, configurable)

Usage
-----
    # Minimal training (no bag, no Foxglove — fastest)
    ros2 launch gazebo_gymnasium_examples cartpole_full.launch.py

    # Headless (no GUI) for faster training
    ros2 launch gazebo_gymnasium_examples cartpole_full.launch.py headless:=true

    # Enable live Foxglove Studio view (ws://localhost:8765)
    ros2 launch gazebo_gymnasium_examples cartpole_full.launch.py foxglove:=true

    # Enable bag recording (MCAP format)
    ros2 launch gazebo_gymnasium_examples cartpole_full.launch.py rosbag:=true

    # Both enabled
    ros2 launch gazebo_gymnasium_examples cartpole_full.launch.py foxglove:=true rosbag:=true

    # A2C with continuous action space
    ros2 launch gazebo_gymnasium_examples cartpole_full.launch.py algorithm:=a2c continuous:=true

Foxglove Studio (requires foxglove:=true)
-----------------------------------------
    1. Open Foxglove Studio
    2. Connect → "Foxglove WebSocket" → ws://localhost:8765
    3. Add panels: 3D (TF), Plot (joint_state), RawMessages

    Or open a recorded bag directly in Foxglove Studio (MCAP format).

Bag recording (requires rosbag:=true)
--------------------------------------
Each run writes to a timestamped subdirectory (./cartpole_bag_YYYYMMDD_HHMMSS/).
Override with bag_output:=<path>.
"""
import datetime
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
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    # -----------------------------------------------------------------------
    # Launch arguments
    # -----------------------------------------------------------------------
    # Timestamped default so consecutive runs never collide on the bag directory
    _ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    algorithm_arg = DeclareLaunchArgument(
        'algorithm', default_value='ppo',
        description="RL algorithm: 'ppo' (default) or 'a2c'")

    continuous_arg = DeclareLaunchArgument(
        'continuous', default_value='false',
        description="Use continuous (Box) action space instead of Discrete")

    timesteps_arg = DeclareLaunchArgument(
        'timesteps', default_value='100000',
        description="Total training timesteps (default: 100000)")

    bag_output_arg = DeclareLaunchArgument(
        'bag_output', default_value=f'./cartpole_bag_{_ts}',
        description=f"rosbag2 output directory (default: ./cartpole_bag_{_ts})")

    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description="Run Gazebo without GUI for faster training (default: false).")

    foxglove_arg = DeclareLaunchArgument(
        'foxglove', default_value='false',
        description="Launch foxglove_bridge WebSocket server on port 8765 (default: false).")

    rosbag_arg = DeclareLaunchArgument(
        'rosbag', default_value='false',
        description="Record /clock, joint_state, and /tf to an MCAP bag (default: false).")

    algorithm = LaunchConfiguration('algorithm')
    continuous = LaunchConfiguration('continuous')
    timesteps = LaunchConfiguration('timesteps')
    bag_output = LaunchConfiguration('bag_output')
    headless = LaunchConfiguration('headless')
    foxglove_enabled = LaunchConfiguration('foxglove')
    rosbag_enabled = LaunchConfiguration('rosbag')

    # -----------------------------------------------------------------------
    # Environment — GZ_PARTITION=0 must match on Gazebo and Python sides
    # -----------------------------------------------------------------------
    set_gz_partition = SetEnvironmentVariable('GZ_PARTITION', '0')

    # -----------------------------------------------------------------------
    # 1. Gazebo
    # -----------------------------------------------------------------------
    examples_share = get_package_share_directory('gazebo_gymnasium_examples')
    ros_gz_sim_share = get_package_share_directory('ros_gz_sim')

    world_path = PathJoinSubstitution([
        examples_share, 'cartpole', 'worlds', 'cartpole.sdf'
    ])

    # headless=true passes -s (server-only, no GUI) to gz_sim.
    # Without the render loop Gazebo is no longer throttled by the display
    # refresh, typically giving 5-10× higher simulation fps.
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': PythonExpression([
                "'-s ' + '", world_path, "'",
                " if '", headless, "' == 'true' else '", world_path, "'",
            ]),
        }.items(),
    )

    # -----------------------------------------------------------------------
    # 2. ros_gz_bridge — expose Gazebo topics to ROS 2 for Foxglove and bag
    #
    # Note: the Gazebo cmd_pos topic '/model/cartpole/joint/slider_to_cart/0/cmd_pos'
    # contains a '0' segment which is illegal in ROS 2 topic names.  ROS 2 rejects
    # it even in remap rules, so we cannot bridge it here.  The joint_state topic
    # already captures the cart position/velocity, which is sufficient for playback.
    # -----------------------------------------------------------------------
    bridge = TimerAction(
        period=3.0,
        actions=[Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='cartpole_bridge',
            arguments=[
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                '/world/cartpole/model/cartpole/joint_state'
                '@sensor_msgs/msg/JointState[gz.msgs.Model',
            ],
            output='screen',
        )],
    )

    # -----------------------------------------------------------------------
    # 3. foxglove_bridge — optional, enabled with foxglove:=true
    # -----------------------------------------------------------------------
    foxglove = TimerAction(
        period=3.5,
        actions=[Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            parameters=[{
                'port': 8765,
                'address': '0.0.0.0',
                'tls': False,
                'use_sim_time': True,
            }],
            output='screen',
            condition=IfCondition(foxglove_enabled),
        )],
    )

    # -----------------------------------------------------------------------
    # 4. rosbag2 recorder — optional, enabled with rosbag:=true
    #
    # Topics recorded:
    #   /clock                                    simulation time
    #   /world/cartpole/model/cartpole/joint_state cart + pole positions/velocities
    #   /tf                                        TF tree published by cartpole_env
    # -----------------------------------------------------------------------
    recorder = TimerAction(
        period=4.0,
        actions=[ComposableNodeContainer(
            name='rosbag_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='rosbag2_transport',
                    plugin='rosbag2_transport::Recorder',
                    name='cartpole_bag_recorder',
                    parameters=[{
                        'record.topics': [
                            '/clock',
                            '/world/cartpole/model/cartpole/joint_state',
                            '/tf',
                        ],
                        'record.all': False,
                        'storage.uri': bag_output,
                        'storage.storage_id': 'mcap',
                        'use_sim_time': True,
                    }],
                ),
            ],
            output='screen',
            condition=IfCondition(rosbag_enabled),
        )],
    )

    # -----------------------------------------------------------------------
    # 5. Training script — PPO or A2C, selected via 'algorithm' launch arg
    #
    # cwd='/tmp' avoids a PyTorch CWD bug where torch/__init__.py incorrectly
    # resolves C extensions when the working directory is the colcon workspace.
    # -----------------------------------------------------------------------
    script_dir = os.path.join(
        get_package_prefix('gazebo_gymnasium_examples'),
        'lib', 'gazebo_gymnasium_examples', 'cartpole',
    )
    ppo_script = os.path.join(script_dir, 'train_sb3.py')
    a2c_script = os.path.join(script_dir, 'train_a2c.py')

    train_ppo = TimerAction(
        period=6.0,
        actions=[ExecuteProcess(
            cmd=['python3', ppo_script, '--timesteps', timesteps],
            additional_env={'GZ_PARTITION': '0'},
            cwd='/tmp',
            output='screen',
            condition=IfCondition(PythonExpression(["'", algorithm, "' == 'ppo'"])),
        )],
    )

    train_a2c_discrete = TimerAction(
        period=6.0,
        actions=[ExecuteProcess(
            cmd=['python3', a2c_script, '--timesteps', timesteps],
            additional_env={'GZ_PARTITION': '0'},
            cwd='/tmp',
            output='screen',
            condition=IfCondition(PythonExpression(
                ["'", algorithm, "' == 'a2c' and '", continuous, "' != 'true'"]
            )),
        )],
    )

    train_a2c_continuous = TimerAction(
        period=6.0,
        actions=[ExecuteProcess(
            cmd=['python3', a2c_script, '--timesteps', timesteps, '--continuous'],
            additional_env={'GZ_PARTITION': '0'},
            cwd='/tmp',
            output='screen',
            condition=IfCondition(PythonExpression(
                ["'", algorithm, "' == 'a2c' and '", continuous, "' == 'true'"]
            )),
        )],
    )

    # -----------------------------------------------------------------------
    # Assemble
    # -----------------------------------------------------------------------
    return LaunchDescription([
        set_gz_partition,
        algorithm_arg,
        continuous_arg,
        timesteps_arg,
        bag_output_arg,
        headless_arg,
        foxglove_arg,
        rosbag_arg,
        gz_sim,
        bridge,
        foxglove,
        recorder,
        train_ppo,
        train_a2c_discrete,
        train_a2c_continuous,
    ])
