#!/usr/bin/env python3
"""
CartPole environment backed by Gazebo Sim.

Matches the classic CartPole-v1 interface. Compatible with any Gymnasium-compatible
RL library — SB3, RLlib, CleanRL, or a hand-rolled agent.

SDF kinematic summary (cartpole.sdf)
-------------------------------------
  Model origin   : z = +0.10 m above the Gazebo world ground plane
  Cart slide axis: Y  (joint slider_to_cart, axis 0 1 0)
  Pole hinge     : at x = +0.12 m from the cart centre (joint cart_to_pole)
  Pole rotation  : about X  (axis 1 0 0)
  Pole geometry  : 1 m long box, centred at z = +0.47 m from the hinge
                   → tip is at z = 0.97 m from the hinge

ROS 2 / TF integration (optional)
----------------------------------
When ROS 2 is available, joint states are published to /tf every step for
visualisation in RViz2 (no URDF or /robot_description required).

  ros2 run rviz2 rviz2   # Fixed Frame = "world", add a TF display

Published TF tree:
    world                          (Gazebo world origin)
    └─ cartpole/base_link          (model origin, z=+0.10 m)
       └─ cartpole/cart            (translates along Y by cart_position)
          └─ cartpole/pole_hinge   (fixed at x=+0.12 m from cart centre)
             └─ cartpole/pole_tip  (tip of pole, rotates about X by pole_angle,
                                    translated 0.97 m along pole's local Z)

TF publishing is silently disabled if rclpy is unavailable or takes more than
_TF_INIT_TIMEOUT_S to initialise (e.g. slow DDS discovery).
"""
import math
import time
import concurrent.futures

import numpy as np
from gymnasium.spaces import Box, Discrete

from gz.transport13 import Node, AdvertiseMessageOptions
from gz.msgs10.double_pb2 import Double
from gz.msgs10.model_pb2 import Model

from gazebo_gymnasium import GazeboEnv

# Set to True via --debug flag in train_sb3.py to enable verbose tracing.
# Prints every callback, every step, and every termination event.
DEBUG = False


def _dbg(msg: str) -> None:
    if DEBUG:
        print(f"[DBG] {msg}", flush=True)


WORLD_NAME = "cartpole"
MODEL_NAME = "cartpole"

_CMD_TOPIC = f"/model/{MODEL_NAME}/joint/slider_to_cart/0/cmd_pos"
_JOINT_STATE_TOPIC = f"/world/{WORLD_NAME}/model/{MODEL_NAME}/joint_state"

# Target positions for the joint position controller.
# Values are in metres along the Y axis.  ±0.75 m gives 1.5× the old ±0.5 m
# amplitude, producing a faster cart movement while staying inside the ±1.2 m
# termination bound.
_POS_LEFT = -0.75
_POS_RIGHT = 0.75

# Episode termination thresholds.
# ±1.2 m cart position matches the tighter CartPole-v1 variant.
# ±12.5° pole angle (~0.218 rad) is the classic failure threshold.
# 500 steps = episode success (truncation).
_MAX_CART_POS = 1.2
_MAX_POLE_ANGLE = 12.5 * math.pi / 180   # ≈ 0.21817 rad
_MAX_STEPS = 500

# How long to wait for a fresh joint-state message after each physics step.
_OBS_TIMEOUT_S = 0.2

# If ROS 2 TF initialisation takes longer than this, skip TF publishing.
# DDS middleware discovery can block for 30+ seconds on some systems.
_TF_INIT_TIMEOUT_S = 3.0

# Pole geometry from the SDF: visual box centred at z=0.47 m, length 1 m.
# Tip = 0.47 + 0.5 = 0.97 m from the pole hinge along the pole's local Z axis.
_POLE_TIP_Z = 0.97

# Model origin is 0.10 m above the Gazebo world ground plane.
_MODEL_ORIGIN_Z = 0.10

# Pole hinge is offset 0.12 m along X from the cart centre.
_HINGE_OFFSET_X = 0.12


# ---------------------------------------------------------------------------
# ROS 2 TF helpers
# ---------------------------------------------------------------------------

def _init_tf_impl():
    """Create a ROS 2 node and TF broadcaster (blocking — called in a thread)."""
    import rclpy
    from tf2_ros import TransformBroadcaster
    try:
        rclpy.init()
    except RuntimeError:
        pass  # already initialised by the caller
    node = rclpy.create_node("cartpole_tf_broadcaster")
    return node, TransformBroadcaster(node)


def _try_init_tf():
    """Non-blocking ROS 2 init: returns (node, broadcaster) or (None, None)."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_init_tf_impl)
        try:
            return future.result(timeout=_TF_INIT_TIMEOUT_S)
        except Exception:
            return None, None


def _wall_clock_stamp():
    """Current wall-clock time as builtin_interfaces/Time.

    self._ros_node.get_clock() returns time 0 without rclpy.spin(), which
    causes tf2/RViz2 to silently discard transforms as 'too old'.
    """
    from builtin_interfaces.msg import Time
    t = time.time()
    return Time(sec=int(t), nanosec=int((t % 1) * 1_000_000_000))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CartPoleEnv(GazeboEnv):
    """
    Observation space (Box, 4):
        [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

    Action space (Discrete 2):
        0 — push cart left  (target Y = -0.5 m)
        1 — push cart right (target Y = +0.5 m)

    Reward: +1 for every step the pole remains upright.
    Terminated: pole angle > 12° or |cart_position| > 2.4 m.
    Truncated: episode exceeds _MAX_STEPS steps.
    """

    def __init__(self, steps_per_action: int = 5):
        # Obs bounds must accommodate values that can arise during the full
        # steps_per_action window (5 × 10 ms = 50 ms).  The pole can swing
        # past 12° in that window, so we use ±π, not ±0.20944.
        obs_space = Box(
            low=np.array([-4.8, -np.inf, -np.pi, -np.inf], dtype=np.float32),
            high=np.array([4.8,  np.inf,  np.pi,  np.inf], dtype=np.float32),
        )
        super().__init__(WORLD_NAME, obs_space, Discrete(2), steps_per_action)

        # Publisher: send target position to the cart's joint position controller
        self._cmd_node = Node()
        self._cmd_pub = self._cmd_node.advertise(_CMD_TOPIC, Double, AdvertiseMessageOptions())

        self._state_node = Node()
        self._state_node.subscribe(Model, _JOINT_STATE_TOPIC, self._on_joint_state)

        self._cart_position = 0.0
        self._cart_velocity = 0.0
        self._pole_angle = 0.0
        self._pole_ang_velocity = 0.0

        # Print GZ_PARTITION so the user can verify it matches in both terminals.
        # If unset, gz.transport defaults to hostname:username — both sides must
        # be on the same machine with the same user for the default to work.
        import os as _os
        gz_part = _os.environ.get("GZ_PARTITION", "<not set — using hostname:username default>")
        print(f"[CartPoleEnv] GZ_PARTITION={gz_part}", flush=True)

        # Optional ROS 2 TF publishing — disabled gracefully if unavailable
        self._ros_node, self._tf_broadcaster = _try_init_tf()
        if self._tf_broadcaster is not None:
            print("[CartPoleEnv] ROS 2 available — publishing /tf")
        else:
            print("[CartPoleEnv] ROS 2 unavailable or timed out — TF disabled")

        self._ping_world_control()

    # --- gz.transport callback ---

    def _on_joint_state(self, msg: Model):
        for joint in msg.joint:
            if joint.name == "slider_to_cart":
                pos = joint.axis1.position
                vel = joint.axis1.velocity
                if math.isnan(pos) or math.isnan(vel):
                    print(f"[WARN] NaN in slider_to_cart: pos={pos}  vel={vel}", flush=True)
                    return
                self._cart_position = pos
                self._cart_velocity = vel
            elif joint.name == "cart_to_pole":
                pos = joint.axis1.position
                vel = joint.axis1.velocity
                if math.isnan(pos) or math.isnan(vel):
                    print(f"[WARN] NaN in cart_to_pole: pos={pos}  vel={vel}", flush=True)
                    return
                self._pole_angle = pos
                self._pole_ang_velocity = vel
        _dbg(f"callback  cart={self._cart_position:+.4f}  pole={self._pole_angle:+.4f}  "
             f"cart_vel={self._cart_velocity:+.4f}  pole_vel={self._pole_ang_velocity:+.4f}")
        self._count_physics_step()

    # --- TF publishing ---

    def _publish_tf(self):
        """Publish the full TF tree to /tf (no-op when ROS 2 is unavailable).

        Axes match the SDF exactly:
          - Cart slides along Y (slider_to_cart axis: 0 1 0)
          - Pole rotates about X (cart_to_pole axis: 1 0 0)
          - Pole hinge is at x=+0.12 m from the cart centre
          - Pole tip is at z=+0.97 m along the pole's local Z axis
        """
        if self._tf_broadcaster is None:
            return

        from geometry_msgs.msg import TransformStamped

        stamp = _wall_clock_stamp()
        half = self._pole_angle / 2.0
        transforms = []

        # world → cartpole/base_link
        # The SDF model origin is 0.10 m above the Gazebo world ground plane.
        t_base = TransformStamped()
        t_base.header.stamp = stamp
        t_base.header.frame_id = "world"
        t_base.child_frame_id = "cartpole/base_link"
        t_base.transform.translation.z = _MODEL_ORIGIN_Z
        t_base.transform.rotation.w = 1.0
        transforms.append(t_base)

        # cartpole/base_link → cartpole/cart
        # Cart position is the prismatic joint displacement along Y.
        t_cart = TransformStamped()
        t_cart.header.stamp = stamp
        t_cart.header.frame_id = "cartpole/base_link"
        t_cart.child_frame_id = "cartpole/cart"
        t_cart.transform.translation.y = float(self._cart_position)
        t_cart.transform.rotation.w = 1.0
        transforms.append(t_cart)

        # cartpole/cart → cartpole/pole_hinge
        # The revolute joint (cart_to_pole) is at x=+0.12 m from the cart centre.
        t_hinge = TransformStamped()
        t_hinge.header.stamp = stamp
        t_hinge.header.frame_id = "cartpole/cart"
        t_hinge.child_frame_id = "cartpole/pole_hinge"
        t_hinge.transform.translation.x = _HINGE_OFFSET_X
        t_hinge.transform.rotation.w = 1.0
        transforms.append(t_hinge)

        # cartpole/pole_hinge → cartpole/pole_tip
        # Rotation: about X axis by pole_angle. Quaternion for R_x(θ):
        #   qx = sin(θ/2),  qy = qz = 0,  qw = cos(θ/2)
        # Translation: pole tip is 0.97 m along the pole's local Z axis,
        # expressed in the pole_hinge frame (BEFORE rotation is applied the
        # translation is in the child frame, so z=0.97 in pole-local space).
        t_tip = TransformStamped()
        t_tip.header.stamp = stamp
        t_tip.header.frame_id = "cartpole/pole_hinge"
        t_tip.child_frame_id = "cartpole/pole_tip"
        t_tip.transform.translation.z = _POLE_TIP_Z
        t_tip.transform.rotation.x = math.sin(half)
        t_tip.transform.rotation.w = math.cos(half)
        transforms.append(t_tip)

        self._tf_broadcaster.sendTransform(transforms)

    # --- GazeboEnv overrides ---

    def reset(self, seed=None, options=None):
        print(f"[reset] ep={self._current_episode + 1}  triggering world_control.reset()", flush=True)
        _dbg(f"reset() called, episode={self._current_episode}")
        # super().reset() calls WorldController.reset() which resets all entities
        # AND leaves Gazebo paused (pause=True is set in the service request).
        # This guarantees a deterministic, paused initial state before we return.
        obs, info = super().reset(seed=seed, options=options)

        # Publish zero target so the position controller aims for y=0 on the
        # first step after reset.  No settle step needed: Gazebo is paused and
        # the controller will process this command when physics resumes in step().
        zero = Double()
        zero.data = 0.0
        self._cmd_pub.publish(zero)

        # Brief sleep lets gz.transport flush the zero command into Gazebo's
        # receive buffer before the first step fires.
        time.sleep(0.05)

        self._publish_tf()
        obs = self.set_default_observation()
        _dbg("reset() done, returning zeros")
        return obs, info

    def apply_action(self, action: int):
        msg = Double()
        msg.data = _POS_LEFT if action == 0 else _POS_RIGHT
        self._cmd_pub.publish(msg)
        _dbg(f"apply_action({action}) → target={msg.data:+.3f}")

    def get_observation(self):
        _dbg(f"get_observation  step={self._current_step}  "
             f"cart={self._cart_position:+.4f}  pole={self._pole_angle:+.4f}  "
             f"cart_vel={self._cart_velocity:+.4f}  pole_vel={self._pole_ang_velocity:+.4f}")
        self._publish_tf()
        return np.array([
            self._cart_position,
            self._cart_velocity,
            self._pole_angle,
            self._pole_ang_velocity,
        ], dtype=np.float32)

    def get_reward(self, action) -> float:
        return 1.0

    def is_terminated(self) -> bool:
        pole_bad = abs(self._pole_angle) > _MAX_POLE_ANGLE
        cart_bad = abs(self._cart_position) > _MAX_CART_POS
        if pole_bad or cart_bad:
            # Always print so the user can see termination reason without --debug
            reason = []
            if pole_bad:
                reason.append(f"pole={self._pole_angle:+.4f} > ±{_MAX_POLE_ANGLE:.5f}")
            if cart_bad:
                reason.append(f"cart={self._cart_position:+.4f} > ±{_MAX_CART_POS}")
            print(f"[TERM] step={self._current_step}  " + "  ".join(reason), flush=True)
        return pole_bad or cart_bad

    def is_truncated(self) -> bool:
        return self._current_step >= _MAX_STEPS

    def set_default_observation(self):
        self._cart_position = 0.0
        self._cart_velocity = 0.0
        self._pole_angle = 0.0
        self._pole_ang_velocity = 0.0
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def get_info(self) -> dict:
        return {
            "gz_episode": self._current_episode,  # "episode" is reserved by SB3
            "gz_step": self._current_step,
            "TimeLimit.truncated": self.is_truncated(),
        }
