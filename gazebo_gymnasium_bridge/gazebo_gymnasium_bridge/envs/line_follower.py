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

"""GazeboLineFollowerEnv: vision-based line following for the diff_drive rover.

Observation (Box(4,) float32):
    [0] forward linear velocity x  (m/s, ∈ [MIN_X_VEL, MAX_X_VEL])
    [1] yaw rate z                 (rad/s, ∈ [MIN_Z_VEL, MAX_Z_VEL])
    [2] camera pitch to ground     (rad, ∈ [-π, π])
    [3] line offset from camera center (pixels,
        ∈ [-IMAGE_WIDTH/2, IMAGE_WIDTH/2] when visible; large sentinel
        otherwise — see LINE_LOST_PIXELS)

Action (Box(2,) float32):
    [0] commanded linear x velocity  (m/s, ∈ [MIN_X_VEL, MAX_X_VEL])
    [1] commanded angular z velocity (rad/s, ∈ [MIN_Z_VEL, MAX_Z_VEL])

Reward (weighted sum of four terms, plus a penalty when the line is lost):

    reward = (
        + W_ALIGNMENT * alignment        # negative absolute line offset (centering)
        + W_HEADING   * heading          # small bonus for low yaw rate when centered
        + W_MOTION    * motion           # forward velocity bonus, peaks at MIN_X_VEL
        + W_SMOOTHNESS * smoothness      # negative |Δaction| (action smoothness)
    ) + (LINE_LOST_PENALTY if line_lost else 0.0)

The hard caps on velocity (max forward 1.0 m/s, max yaw 1.25 rad/s) come
from the physical rover the sim is meant to model — beyond ~1 m/s forward
the front caster pops up and odometry drift becomes severe. The action
space's outer interval extends slightly past the physical max (so SAC
exploration noise still has a smooth boundary) but the policy is rewarded
for staying inside the safe regime.

Termination: line lost OR camera pitch outside ±0.6 rad (rover tipped).
Truncation: current_step >= max_episode_steps.
"""

import os
import threading
import time
from typing import Optional

import gymnasium as gym
from gymnasium.spaces import Box
from gz.msgs10.float_v_pb2 import Float_V
from gz.transport13 import AdvertiseMessageOptions
from gz.transport13 import Node
import numpy as np

from ..backend.metrics import MetricsPublisher
from ..backend.nodes import world_control


# Physical and observation envelope. MIN/MAX_X are used for both
# observation bounds AND action bounds — the agent commands what it
# perceives. MIN_X_VEL is 0.5 m/s (the user-specified minimum forward
# speed); MAX_X_VEL is 1.0 m/s (physical limit before the caster pops up).
MIN_X_VEL = 0.5
MAX_X_VEL = 1.0
# Yaw rate (z angular velocity) goes both ways.
MIN_Z_VEL = -1.25
MAX_Z_VEL = 1.25
# Camera image width; should match the SDF camera config.
IMAGE_WIDTH = 160
HALF_IMAGE_WIDTH = IMAGE_WIDTH / 2.0
# Sentinel from the plugin — must match LINE_LOST_VALUE in
# `line_follower_learner.py`.
LINE_LOST_PIXELS = 1e3

# Reward weights (per the user's spec: alignment 2.0, heading 1.0,
# motion 5.0, smoothness -0.1). LINE_LOST_PENALTY ends the episode early
# AND tips the cumulative reward strongly negative.
W_ALIGNMENT = 2.0
W_HEADING = 1.0
W_MOTION = 5.0
W_SMOOTHNESS = -0.1
LINE_LOST_PENALTY = -50.0

# Episode termination when the rover tips past this pitch (rad).
PITCH_TERMINATION_RAD = 0.6

DEFAULT_FRAME_SKIP = 3


class GazeboLineFollowerEnv(gym.Env):
    """Vision-based line-following Gymnasium env for the diff_drive rover."""

    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(self, world_name: str = "line_follower",
                 max_episode_steps: int = 500,
                 frame_skip: int = DEFAULT_FRAME_SKIP,
                 reset_timeout: float = 5.0,
                 step_timeout: float = 5.0,
                 render_mode: Optional[str] = None):
        super().__init__()

        self.observation_space = Box(
            low=np.array(
                [MIN_X_VEL, MIN_Z_VEL, -np.pi, -HALF_IMAGE_WIDTH],
                dtype=np.float32,
            ),
            high=np.array(
                [MAX_X_VEL, MAX_Z_VEL, np.pi, HALF_IMAGE_WIDTH],
                dtype=np.float32,
            ),
            shape=(4,),
            dtype=np.float32,
        )
        self.action_space = Box(
            low=np.array([MIN_X_VEL, MIN_Z_VEL], dtype=np.float32),
            high=np.array([MAX_X_VEL, MAX_Z_VEL], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        self.render_mode = render_mode

        self.world_name = world_name
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.reset_timeout = reset_timeout
        self.step_timeout = step_timeout

        self.current_step = 0
        self.current_episode = 0
        self.episode_reward = 0.0

        # For smoothness term: keep the previous action so we can take
        # |Δaction| each step.
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._line_lost_count = 0
        self._last_reason = None

        self._debug = os.environ.get(
            "GAZEBO_GYM_VERBOSE", "false").lower() in ("1", "true", "yes", "on")

        self._action_node = Node()
        self._action_pub = self._action_node.advertise(
            "/env/action", Float_V, AdvertiseMessageOptions())

        # /env/metrics publisher — see backend/metrics.py for the contract.
        self._metrics = MetricsPublisher("gazebo_line_follower_env_metrics")
        self._step_window = []  # type: list[tuple[float, float]]
        self._total_steps = 0

        self._latest_state = [0.0, 0.0, 0.5, LINE_LOST_PIXELS]
        self._state_event = threading.Event()

        self._state_node = Node()
        self._state_node.subscribe(Float_V, "/env/state", self._on_state)

        self._world_control = world_control.WorldController(world_name, steps_per_action=0)

        print(f"[GazeboLineFollowerEnv] ready (world={world_name!r}, "
              f"frame_skip={frame_skip}, max_episode_steps={max_episode_steps})")

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if self.current_step > 0:
            reason = self._last_reason or "unknown"
            print(
                f"[EpisodeSummary] ep={self.current_episode} "
                f"steps={self.current_step} reward={self.episode_reward:.1f} "
                f"reason={reason} line_lost_frames={self._line_lost_count}"
            )

        self._state_event.clear()
        self._world_control.reset(pause_after=False)
        if not self._state_event.wait(timeout=self.reset_timeout):
            print("[GazeboLineFollowerEnv] WARN: reset state wait timed out")

        self.current_step = 0
        self.current_episode += 1
        self.episode_reward = 0.0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._line_lost_count = 0
        self._last_reason = None

        obs = self._obs_from_state()
        return obs, {}

    def step(self, action):
        step_start = time.perf_counter()

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < 2:
            raise ValueError(f"Action must have at least 2 elements, got {action.shape}")
        # Clip to declared bounds — keeps the rover within physical limits
        # even if the policy outputs values past the action space (which SAC
        # can do because of its squashing transform).
        x_vel = float(np.clip(action[0], MIN_X_VEL, MAX_X_VEL))
        z_vel = float(np.clip(action[1], MIN_Z_VEL, MAX_Z_VEL))

        msg = Float_V()
        msg.data.extend([x_vel, z_vel])
        self._state_event.clear()
        self._action_pub.publish(msg)
        if not self._state_event.wait(timeout=self.step_timeout):
            print("[GazeboLineFollowerEnv] WARN: step state wait timed out")
            self._last_reason = "timeout"
            return self._obs_from_state(), 0.0, True, False, {"timeout": True}

        obs = self._obs_from_state()
        meas_x, meas_z, pitch, line_offset_raw = self._latest_state
        line_lost = line_offset_raw >= LINE_LOST_PIXELS / 2

        # --- Reward shaping (four parts, then line-lost penalty) ---
        # 1. Alignment: small when the line is centered, large negative
        #    when it's off to one side. Normalized to the half image width
        #    so the term is roughly in [-1, 0].
        if line_lost:
            alignment = -1.0
        else:
            alignment = -abs(line_offset_raw) / HALF_IMAGE_WIDTH

        # 2. Heading: rewards low yaw rate (smooth driving). Bigger reward
        #    when we're already centered — discourages spinning in place to
        #    "find" the line.
        heading = (1.0 - abs(meas_z) / MAX_Z_VEL) * (1.0 + alignment)

        # 3. Motion: rewards forward velocity, peaks at MIN_X_VEL (the
        #    user's "absolute min forward velocity"). Reaching 0.5 m/s gets
        #    reward 1.0; faster gets a slightly reduced bonus to discourage
        #    blowing past the physical caster-pop limit.
        if meas_x < MIN_X_VEL:
            motion = meas_x / MIN_X_VEL  # 0 → 1 as we approach min speed
        else:
            # Linear decay from 1.0 (at 0.5 m/s) to 0.3 (at 1.0 m/s),
            # then clipped — so the policy isn't penalized for fast
            # driving but doesn't get free reward either.
            motion = max(0.3, 1.0 - (meas_x - MIN_X_VEL) / (MAX_X_VEL - MIN_X_VEL) * 0.7)

        # 4. Smoothness: penalize abrupt action changes.
        delta = action[:2] - self._prev_action
        smoothness = float(np.linalg.norm(delta))

        reward = (
            W_ALIGNMENT * alignment
            + W_HEADING * heading
            + W_MOTION * motion
            + W_SMOOTHNESS * smoothness
        )

        if line_lost:
            self._line_lost_count += 1
            reward += LINE_LOST_PENALTY
            terminated = True
            self._last_reason = "line_lost"
        elif abs(pitch) > PITCH_TERMINATION_RAD:
            terminated = True
            self._last_reason = "tipped"
        else:
            terminated = False

        self.current_step += 1
        self._total_steps += 1
        truncated = (not terminated) and self.current_step >= self.max_episode_steps
        if truncated:
            self._last_reason = "truncated"

        self.episode_reward += reward
        self._prev_action = action[:2].astype(np.float32)

        step_ms = (time.perf_counter() - step_start) * 1000.0
        self._publish_metrics(step_ms)

        info = {
            "alignment": float(alignment),
            "heading": float(heading),
            "motion": float(motion),
            "smoothness": float(smoothness),
            "line_lost": bool(line_lost),
        }
        if truncated:
            info["TimeLimit.truncated"] = True

        if self._debug:
            print(
                f"[LineFollower][ep {self.current_episode} step {self.current_step}] "
                f"a=({x_vel:+.2f},{z_vel:+.2f}) reward={reward:+.2f} "
                f"cum={self.episode_reward:+.1f} "
                f"obs(x={meas_x:+.2f},z={meas_z:+.2f},pitch={pitch:+.2f},"
                f"line={line_offset_raw:+.1f}) "
                f"alignment={alignment:+.2f} motion={motion:.2f} "
                f"step_ms={step_ms:.1f}"
            )

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _obs_from_state(self) -> np.ndarray:
        meas_x, meas_z, pitch, line_offset_raw = self._latest_state
        # Clip the line offset into the observation-space bounds. The
        # plugin uses a large sentinel for "line lost"; an unclipped value
        # would violate the Box's `high` bound and break SB3's env_checker.
        line_offset = float(np.clip(line_offset_raw, -HALF_IMAGE_WIDTH, HALF_IMAGE_WIDTH))
        return np.array(
            [
                float(np.clip(meas_x, MIN_X_VEL, MAX_X_VEL)),
                float(np.clip(meas_z, MIN_Z_VEL, MAX_Z_VEL)),
                float(np.clip(pitch, -np.pi, np.pi)),
                line_offset,
            ],
            dtype=np.float32,
        )

    def _publish_metrics(self, step_ms: float) -> None:
        """Push an EnvMetrics ROS message to /env/metrics (see metrics.py)."""
        now = time.perf_counter()
        self._step_window.append((now, step_ms))
        cutoff = now - 1.0
        while self._step_window and self._step_window[0][0] < cutoff:
            self._step_window.pop(0)
        steps_per_sec = float(len(self._step_window))
        mean_step_ms = (
            sum(ms for _, ms in self._step_window) / len(self._step_window)
            if self._step_window else 0.0
        )
        self._metrics.publish(
            steps_per_sec=steps_per_sec,
            mean_step_ms=mean_step_ms,
            episode_reward=self.episode_reward,
            current_step=self.current_step,
            total_steps=self._total_steps,
            current_episode=self.current_episode,
        )

    def _on_state(self, msg):
        if len(msg.data) >= 4:
            self._latest_state = [float(msg.data[i]) for i in range(4)]
            self._state_event.set()
