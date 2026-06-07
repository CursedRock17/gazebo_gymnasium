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

"""Line-follower sync-gate plugin.

Bridges the external Gymnasium env (running in another process) to the
Onshape-exported `rover` model by:

  in:   /env/action  (Float_V, two values)
            data[0] = linear x velocity command (m/s)
            data[1] = angular z velocity command (rad/s)

  out:  /env/state   (Float_V, four values)
            data[0] = rover linear x velocity (from odometry)
            data[1] = rover angular z velocity (from odometry)
            data[2] = camera pitch relative to ground (rad)
            data[3] = line offset in pixels — distance of the line centroid
                      from the camera image's vertical centerline. Negative
                      = line is to the left, positive = right. If the line
                      is not visible the value is set to LINE_LOST_VALUE.

The plugin commands the rover by publishing `gz.msgs.Twist` on the model's
`/cmd_vel` topic (consumed by the DiffDrive system plugin baked into the
rover SDF). The sync barrier is the standard one: after each `/env/action`
arrives, the plugin counts `frame_skip` post-update ticks, then publishes
`/env/state` once.

Line detection is a numpy threshold over the camera image. The rover's
ground is light gray (~0.85) and the track is near-black (~0.02), so a
brightness threshold reliably separates them.
"""

import math

from gz.msgs10.float_v_pb2 import Float_V
from gz.msgs10.image_pb2 import Image
from gz.msgs10.odometry_pb2 import Odometry
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.twist_pb2 import Twist
from gz.sim8 import World
from gz.sim8 import world_entity
from gz.transport13 import AdvertiseMessageOptions
from gz.transport13 import Node
import numpy as np


# A pixel is treated as "line" if its mean RGB is below this 0-255 value.
# Ground plane brightness is ~217 (0.85 × 255); line is ~5 (0.02 × 255).
# 80 is a safe midpoint with margin on both sides.
LINE_BRIGHTNESS_THRESHOLD = 80

# Sentinel published when the line is not visible. Picked large enough
# (relative to half image width) that the env can detect it unambiguously.
LINE_LOST_VALUE = 1e3


class LineFollowerSyncGate:

    def __init__(self):
        self.world_name = None
        self.agent_name = None
        self.frame_skip = 3

        self._pending_action = None
        self._ticks_since_action = 0
        self._action_active = False

        # Sensor cache, updated by callbacks.
        self._linear_x = 0.0
        self._angular_z = 0.0
        self._camera_pitch = 0.5  # default matches the SDF camera_link pose
        self._line_offset = LINE_LOST_VALUE
        self._image_width = 320

        # Transport nodes — one per logical channel keeps the gz-transport
        # subscriber registrations simple.
        self._action_node = None
        self._state_node = None
        self._cmd_vel_node = None
        self._odom_node = None
        self._pose_node = None
        self._camera_node = None

        self._state_pub = None
        self._cmd_vel_pub = None

    # ------------------------------------------------------------------ #
    # Gazebo lifecycle hooks
    # ------------------------------------------------------------------ #

    def configure(self, entity, element, ecm, eventManager):
        self.world_name = World(world_entity(ecm)).name(ecm)
        self.agent_name = element.get_string("agent_name") or "rover"
        if not self.agent_name.startswith("/"):
            self.agent_name = "/" + self.agent_name

        sdf_frame_skip = int(element.get_double("frame_skip") or 0)
        if sdf_frame_skip > 0:
            self.frame_skip = sdf_frame_skip

        # cmd_vel publisher — drives the DiffDrive system plugin baked into
        # the rover model. The DiffDrive plugin subscribes to /cmd_vel by
        # default; we publish on /model/<agent>/cmd_vel which it also accepts.
        self._cmd_vel_node = Node()
        cmd_topic = "/model" + self.agent_name + "/cmd_vel"
        self._cmd_vel_pub = self._cmd_vel_node.advertise(
            cmd_topic, Twist, AdvertiseMessageOptions())

        # Odometry subscription (rover's measured velocities).
        self._odom_node = Node()
        odom_topic = "/model" + self.agent_name + "/odometry"
        self._odom_node.subscribe(Odometry, odom_topic, self._on_odom)

        # Pose subscription (for camera pitch).
        self._pose_node = Node()
        pose_topic = "/world/" + self.world_name + "/pose/info"
        self._pose_node.subscribe(Pose_V, pose_topic, self._on_pose)

        # Camera image subscription.
        self._camera_node = Node()
        # gz-sim publishes camera images on a topic derived from the sensor's
        # <topic> tag. In the rover SDF we set topic="camera"; the canonical
        # full topic name then is /world/<world>/model/<agent>/link/camera_link/sensor/camera/image
        camera_topic = (
            "/world/" + self.world_name
            + "/model" + self.agent_name
            + "/link/camera_link/sensor/camera/image"
        )
        self._camera_node.subscribe(Image, camera_topic, self._on_image)

        # /env/action, /env/state — the standard sync-gate channels.
        self._action_node = Node()
        self._action_node.subscribe(Float_V, "/env/action", self._on_action)

        self._state_node = Node()
        self._state_pub = self._state_node.advertise(
            "/env/state", Float_V, AdvertiseMessageOptions())

        print(f"[LineFollowerSyncGate] ready: world={self.world_name!r} "
              f"agent={self.agent_name!r} frame_skip={self.frame_skip}")

    def pre_update(self, info, ecm):
        if info.paused:
            return
        if self._pending_action is None:
            return
        # Publish cmd_vel every physics tick — DiffDrive expects a continuous
        # stream. Zero it during the grace period so the rover doesn't drift
        # while the agent is computing the next action.
        msg = Twist()
        if self._action_active:
            msg.linear.x = float(self._pending_action[0])
            msg.angular.z = float(self._pending_action[1])
        else:
            msg.linear.x = 0.0
            msg.angular.z = 0.0
        self._cmd_vel_pub.publish(msg)

    def post_update(self, info, ecm):
        if info.paused:
            return
        if self._pending_action is None or not self._action_active:
            return
        self._ticks_since_action += 1
        if self._ticks_since_action >= self.frame_skip:
            self._publish_state()
            self._action_active = False

    def reset(self, info, ecm):
        self._pending_action = None
        self._action_active = False
        self._ticks_since_action = 0
        self._linear_x = 0.0
        self._angular_z = 0.0
        self._line_offset = LINE_LOST_VALUE
        self._publish_state()

    # ------------------------------------------------------------------ #
    # Topic callbacks
    # ------------------------------------------------------------------ #

    def _on_action(self, msg):
        if len(msg.data) < 2:
            return
        self._pending_action = (float(msg.data[0]), float(msg.data[1]))
        self._ticks_since_action = 0
        self._action_active = True

    def _on_odom(self, msg):
        # gz.msgs.Odometry has twist (linear + angular) in the body frame.
        self._linear_x = float(msg.twist.linear.x)
        self._angular_z = float(msg.twist.angular.z)

    def _on_pose(self, pose_v_msg):
        # The camera link's orientation gives us its pitch relative to the
        # world. The camera_link's SDF pose has pitch=0.5 rad baked in; if
        # the rover tips forward (e.g. caster pops up under hard accel)
        # this number changes — which is exactly what we want to observe.
        for pose in pose_v_msg.pose:
            if pose.name == "camera_link":
                # Pitch from quaternion (w, x, y, z) — extract Y rotation.
                q = pose.orientation
                # Standard quat -> pitch formula.
                sinp = 2.0 * (q.w * q.y - q.z * q.x)
                if abs(sinp) >= 1:
                    self._camera_pitch = math.copysign(math.pi / 2, sinp)
                else:
                    self._camera_pitch = math.asin(sinp)

    def _on_image(self, img_msg):
        try:
            width = int(img_msg.width)
            height = int(img_msg.height)
            self._image_width = width
            if width == 0 or height == 0:
                return
            # gz publishes 3-byte RGB by default for our camera config.
            arr = np.frombuffer(img_msg.data, dtype=np.uint8)
            if arr.size != width * height * 3:
                return
            arr = arr.reshape(height, width, 3)
            # Look only at the bottom half of the image — that's the
            # ground close to the rover, less noisy than the horizon.
            roi = arr[height // 2:, :, :]
            gray = roi.mean(axis=2)
            mask = gray < LINE_BRIGHTNESS_THRESHOLD
            total = int(mask.sum())
            if total < 20:
                # Too few black pixels: line is out of view.
                self._line_offset = LINE_LOST_VALUE
                return
            # Weighted centroid along x (columns), then offset from
            # the image's horizontal center.
            cols = np.arange(width, dtype=np.float64)
            x_centroid = float((mask.sum(axis=0) * cols).sum() / total)
            self._line_offset = x_centroid - (width / 2.0)
        except Exception as exc:
            # An exception inside a gz transport callback would otherwise
            # silently kill subsequent callbacks. Surface it but keep
            # publishing the last good value.
            print(f"[LineFollowerSyncGate] _on_image error: {exc}")

    # ------------------------------------------------------------------ #
    # State publishing
    # ------------------------------------------------------------------ #

    def _publish_state(self):
        msg = Float_V()
        msg.data.extend([
            float(self._linear_x),
            float(self._angular_z),
            float(self._camera_pitch),
            float(self._line_offset),
        ])
        self._state_pub.publish(msg)


def get_system():
    """Expose the system so Gazebo's PythonSystemLoader can find it."""
    return LineFollowerSyncGate()
