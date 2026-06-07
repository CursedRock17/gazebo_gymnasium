#!/usr/bin/env python3
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

"""Generate world SDFs, launch files, and skeleton plugin stubs for every auto-converted model.

Skips half_cheetah (name-collision issue in the converted SDF — needs manual cleanup) and the
already-curated cartpole + inverted_pendulum.

The plugin stubs are placeholders that emit a "TODO: implement" message when loaded. They keep the
env *launchable* (so users can verify the model renders) while making it explicit that the
action/observation plumbing for each new env still needs hand work.

Run from project root: ./venv/bin/python scripts/generate_worlds_and_launches.py
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCES = PROJECT_ROOT / "gazebo_gymnasium_examples" / "gazebo_gymnasium_resources"
WORLDS = RESOURCES / "worlds"
MODELS = RESOURCES / "models"
PLUGINS = RESOURCES / "plugins"
LAUNCH_DIR = (PROJECT_ROOT
              / "gazebo_gymnasium_examples"
              / "gazebo_gymnasium_bringup"
              / "launch")

# Already manually curated — leave alone.
HAND_CURATED = {"cartpole", "inverted_pendulum", "diff_drive"}

# half_cheetah failed validation due to duplicate frame names (mjcf2urdf
# emitted both `bthigh` legs with the same name). Needs manual rename pass.
NEEDS_MANUAL_FIX = {"half_cheetah"}


WORLD_TEMPLATE = """<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="{env_name}">
    <physics name="10ms" type="ignored">
      <max_step_size>0.01</max_step_size>
      <real_time_update_rate>0</real_time_update_rate>
    </physics>
    <plugin filename="gz-sim-physics-system"           name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-contact-system"           name="gz::sim::systems::Contact"/>
    <plugin filename="gz-sim-sensors-system"           name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-user-commands-system"     name="gz::sim::systems::UserCommands"/>

    <plugin
      filename="gz-sim-python-system-loader-system"
      name="gz::sim::systems::PythonSystemLoader">
      <module_name>{env_name}_learner</module_name>
      <agent_name>{env_name}</agent_name>
      <frame_skip>5</frame_skip>
    </plugin>

    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.8 0.8 0.8 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <model name="{env_name}">
      <pose>0 0 {z_offset} 0 0 0</pose>
      <include merge="true">
        <uri>package://gazebo_gymnasium_resources/models/{env_name}</uri>
      </include>
    </model>
  </world>
</sdf>
"""


LAUNCH_TEMPLATE =\
                  '''"""
Auto-generated launch for {env_name}. Same toggleable add-ons as cartpole.launch.py:
    verbose:=true | use_foxglove:=true | foxglove_port:=N | record_rosbag:=true
"""

import os
from datetime import datetime

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.actions import AppendEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PythonExpression

from launch_ros.actions import Node


# See cartpole.launch.py for the rationale on these env vars.
GZ_PLUGIN_PATHS = (
    '/usr/lib/x86_64-linux-gnu/gz-sim-8/plugins:'
    '/usr/local/lib/gz-sim-8/plugins'
)


def generate_launch_description():
    resources = get_package_share_directory('gazebo_gymnasium_resources')
    bringup_share = get_package_share_directory('gazebo_gymnasium_bringup')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world_path = os.path.join(resources, 'worlds', '{env_name}.sdf')

    verbose_arg = DeclareLaunchArgument('verbose', default_value='true')
    use_foxglove_arg = DeclareLaunchArgument('use_foxglove', default_value='false')
    foxglove_port_arg = DeclareLaunchArgument('foxglove_port', default_value='8765')
    record_rosbag_arg = DeclareLaunchArgument('record_rosbag', default_value='false')
    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description='Run gz sim server-only (no GUI). Big training speedup.',
    )

    pin_python_home = SetEnvironmentVariable('PYTHONHOME', '/usr')
    set_verbose_env = SetEnvironmentVariable('GAZEBO_GYM_VERBOSE',
                                              LaunchConfiguration('verbose'))
    prepend_gz_plugin_path = AppendEnvironmentVariable(
        'GZ_SIM_SYSTEM_PLUGIN_PATH', GZ_PLUGIN_PATHS,
        prepend=True,
    )
    prepend_pythonpath = AppendEnvironmentVariable(
        'PYTHONPATH', os.path.join(resources, 'plugins'), prepend=True,
    )

    # `-s` when headless=true so gz sim runs server-only.
    gz_args = PythonExpression([
        "'-r ' + '", world_path, "' + (' -s' if '",
        LaunchConfiguration('headless'), "' == 'true' else '')"
    ])
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={{'gz_args': gz_args}}.items(),
    )

    # ros_gz_bridge driven by config/{env_name}_bridge.yaml. Topic list
    # lives outside the launch so each env's bridge surface can be
    # reviewed (and have per-topic QoS overrides set) in one place.
    bridge_config = os.path.join(
        bringup_share, 'config', '{env_name}_bridge.yaml',
    )
    ros_gz_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        parameters=[{{'config_file': bridge_config}}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_foxglove')),
    )
    foxglove_bridge_node = Node(
        package='foxglove_bridge', executable='foxglove_bridge',
        parameters=[{{'port': LaunchConfiguration('foxglove_port')}}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_foxglove')),
    )

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    bag_path = os.path.join('rosbags', f'{env_name}_{{ts}}')
    record_rosbag_node = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-a', '-s', 'mcap', '-o', bag_path],
        output='screen',
        condition=IfCondition(LaunchConfiguration('record_rosbag')),
    )

    # Static map -> world transform so Foxglove's 3D panel has a root
    # frame to anchor to (pose/info publishes link transforms with
    # parent_frame_id="world").
    world_static_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        arguments=['--frame-id', 'map', '--child-frame-id', 'world'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_foxglove')),
    )

    return LaunchDescription([
        verbose_arg, headless_arg, use_foxglove_arg, foxglove_port_arg,
        record_rosbag_arg,
        pin_python_home, set_verbose_env,
        prepend_gz_plugin_path, prepend_pythonpath,
        gz_sim, ros_gz_bridge, foxglove_bridge_node,
        world_static_tf, record_rosbag_node,
    ])
'''


PLUGIN_STUB_TEMPLATE =\
                       '''"""
{env_name}_learner — auto-generated SKELETON sync-gate plugin.

Status: NOT IMPLEMENTED. The model loads, but this plugin currently does
nothing with actions or state. To wire it up:

  1. Decide on action mapping (cmd_force / cmd_vel / cmd_pos) for each
     actuated joint. Browse the model SDF at
     gazebo_gymnasium_resources/models/{env_name}/{env_name}.sdf for joint
     names and the actuator plugin attached.
  2. Decide observation vector (look at canonical Gymnasium
     `gymnasium/envs/mujoco/{env_name}_v5.py` for the reward + obs layout).
  3. Copy the body of cartpole_learner.py or inverted_pendulum_learner.py
     and adapt for this env's joints/links.

The empty `pre_update` / `post_update` here lets you `ros2 launch` the env
and verify the model renders without errors before doing the wiring work.
"""

from gz.sim8 import World, world_entity
from gz.transport13 import Node, AdvertiseMessageOptions
from gz.msgs10.float_v_pb2 import Float_V


class {class_name}:
    def __init__(self):
        self.world_name = None
        self.agent_name = None
        self.frame_skip = 5
        self._action_node = None
        self._state_node = None
        self._state_pub = None
        self._has_warned = False

    def configure(self, entity, element, ecm, eventManager):
        self.world_name = World(world_entity(ecm)).name(ecm)
        self.agent_name = element.get_string("agent_name") or "{env_name}"
        if not self.agent_name.startswith("/"):
            self.agent_name = "/" + self.agent_name
        sdf_frame_skip = int(element.get_double("frame_skip") or 0)
        if sdf_frame_skip > 0:
            self.frame_skip = sdf_frame_skip

        self._action_node = Node()
        self._action_node.subscribe(Float_V, "/env/action", self._on_action)
        self._state_node = Node()
        self._state_pub = self._state_node.advertise(
            "/env/state", Float_V, AdvertiseMessageOptions())

        print(f"[{class_name}] LOADED but UNIMPLEMENTED for env={{self.agent_name!r}}")
        print(f"[{class_name}] Actions arriving on /env/action will be ignored.")
        print(f"[{class_name}] Implement pre_update/post_update + topic callbacks "
              f"following the cartpole_learner.py / inverted_pendulum_learner.py pattern.")

    def pre_update(self, info, ecm):
        pass

    def post_update(self, info, ecm):
        pass

    def reset(self, info, ecm):
        msg = Float_V()
        msg.data.extend([0.0, 0.0, 0.0, 0.0])
        if self._state_pub is not None:
            self._state_pub.publish(msg)

    def _on_action(self, msg):
        if not self._has_warned:
            print(f"[{class_name}] received action — but plugin is UNIMPLEMENTED.")
            self._has_warned = True


def get_system():
    return {class_name}()
'''


# Per-env hint of where to spawn the model so it isn't underground/falling.
# Locomotion envs sit on the ground; pendulum-like envs need clearance for
# the link to swing.
Z_OFFSETS = {
    "ant": 0.75,
    "half_cheetah": 0.5,
    "hopper": 1.25,
    "humanoid": 1.3,
    "humanoidstandup": 0.3,
    "inverted_double_pendulum": 0.6,
    "point": 0.0,
    "pusher": 0.0,
    "pusher_v5": 0.0,
    "reacher": 0.0,
    "swimmer": 0.05,
    "walker2d": 1.25,
    "walker2d_v5": 1.25,
}


def env_class_name(env_name: str) -> str:
    return "".join(part.capitalize() for part in env_name.split("_")) + "SyncGate"


def main():
    WORLDS.mkdir(parents=True, exist_ok=True)
    LAUNCH_DIR.mkdir(parents=True, exist_ok=True)
    PLUGINS.mkdir(parents=True, exist_ok=True)

    envs = []
    for model_dir in sorted(MODELS.iterdir()):
        if not model_dir.is_dir():
            continue
        env_name = model_dir.name
        if env_name in HAND_CURATED or env_name in NEEDS_MANUAL_FIX:
            continue
        if env_name == "mujoco_sources":
            continue
        if not (model_dir / f"{env_name}.sdf").exists():
            continue
        envs.append(env_name)

    print(f"Generating for {len(envs)} envs: {envs}")

    for env_name in envs:
        z = Z_OFFSETS.get(env_name, 0.0)

        world_path = WORLDS / f"{env_name}.sdf"
        world_path.write_text(WORLD_TEMPLATE.format(env_name=env_name, z_offset=z))

        launch_path = LAUNCH_DIR / f"{env_name}.launch.py"
        launch_path.write_text(LAUNCH_TEMPLATE.format(env_name=env_name))

        plugin_path = PLUGINS / f"{env_name}_learner.py"
        if not plugin_path.exists():
            plugin_path.write_text(PLUGIN_STUB_TEMPLATE.format(
                env_name=env_name, class_name=env_class_name(env_name)))

        print(f"  OK {env_name}: world + launch (+ stub plugin if missing)")

    print(f"\nDone. {len(envs)} envs scaffolded.")
    print("Curated (untouched): cartpole, inverted_pendulum.")
    print("Needs manual fix: half_cheetah (duplicate frame names in SDF).")


if __name__ == "__main__":
    main()
