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

"""GazeboInvertedDoublePendulumEnv — Gazebo port of Gymnasium's InvertedDoublePendulum-v5.

Canonical obs (Gymnasium):     [cart_pos, sin(pole1), sin(pole2), cos(pole1), cos(pole2),
cart_vel, pole1_vel, pole2_vel, 3 constraint forces] That's 11-dim. Our port drops the 3 constraint
forces (Gazebo doesn't expose them cleanly), yielding an 8-dim observation. SAC / A2C / TD3 all
train fine without those terms.

Action: Box(1,) in [-1, 1], multiplied by gear=500 in the sync-gate plugin before being published
as cmd_force on the slider (matches MJCF `motor gear="500"`).

Reward: +alive_bonus per step. Termination on either pole tipping > 0.4 rad, which is a Gazebo-
friendly approximation of the canonical "tip below height threshold" condition (we don't read the
tip site directly).
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

POLE_ANGLE_THRESHOLD = 0.4
ALIVE_BONUS = 10.0
DEFAULT_FRAME_SKIP = 5


class GazeboInvertedDoublePendulumEnv(gym.Env):

    metadata = {"render_modes": [], "render_fps": 20}

    def __init__(self, world_name: str = "inverted_double_pendulum",
                 max_episode_steps: int = 1000,
                 frame_skip: int = DEFAULT_FRAME_SKIP,
                 reset_timeout: float = 5.0,
                 step_timeout: float = 5.0,
                 render_mode: Optional[str] = None):
        super().__init__()

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.render_mode = render_mode

        self.world_name = world_name
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.reset_timeout = reset_timeout
        self.step_timeout = step_timeout

        self.current_step = 0
        self.current_episode = 0
        self.episode_reward = 0.0
        self.episode_reward_list = []
        self.episode_steps_list = []

        self._step_times_ms = []
        self._max_pole_angle_seen = 0.0
        self._max_cart_position_seen = 0.0
        self._last_reason = None

        self._debug = os.environ.get(
            "GAZEBO_GYM_VERBOSE", "false").lower() in ("1", "true", "yes", "on")
        try:
            self._verbose_every = max(1, int(os.environ.get("GAZEBO_GYM_VERBOSE_EVERY", "1")))
        except ValueError:
            self._verbose_every = 1

        self._action_node = Node()
        self._action_pub = self._action_node.advertise(
            "/env/action", Float_V, AdvertiseMessageOptions())

        # /env/metrics publisher — see backend/metrics.py for the contract.
        self._metrics = MetricsPublisher("gazebo_inverted_double_pendulum_env_metrics")
        self._step_window = []  # type: list[tuple[float, float]]
        self._total_steps = 0

        self._latest_state = [0.0] * 6
        self._state_event = threading.Event()
        self._state_node = Node()
        self._state_node.subscribe(Float_V, "/env/state", self._on_state)

        self._world_control = world_control.WorldController(world_name, steps_per_action=0)

        print(f"[GazeboInvertedDoublePendulumEnv] ready (world={world_name!r}, "
              f"frame_skip={frame_skip})")

    def _obs_from_state(self) -> np.ndarray:
        cart_pos, pole_ang, pole2_ang, cart_vel, pole_vel, pole2_vel = self._latest_state
        return np.array([
            cart_pos,
            np.sin(pole_ang), np.sin(pole2_ang),
            np.cos(pole_ang), np.cos(pole2_ang),
            cart_vel, pole_vel, pole2_vel,
        ], dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if self.current_step > 0:
            self.episode_reward_list.append(self.episode_reward)
            self.episode_steps_list.append(self.current_step)
            phys_ticks = self.current_step * self.frame_skip
            reason = self._last_reason or "unknown"
            mean_step_ms = (sum(self._step_times_ms) / len(self._step_times_ms)
                            if self._step_times_ms else 0.0)
            print(
                f"[EpisodeSummary] ep={self.current_episode} "
                f"steps={self.current_step} phys_ticks={phys_ticks} "
                f"reward={self.episode_reward:.1f} reason={reason} "
                f"max_pole_a={self._max_pole_angle_seen:+.3f} "
                f"max_cart_p={self._max_cart_position_seen:+.3f} "
                f"mean_step_ms={mean_step_ms:.1f}"
            )

        self._state_event.clear()
        self._world_control.reset(pause_after=False)
        if not self._state_event.wait(timeout=self.reset_timeout):
            print("[GazeboInvertedDoublePendulumEnv] WARN: reset state wait timed out")

        self.current_step = 0
        self.current_episode += 1
        self.episode_reward = 0.0
        self._step_times_ms = []
        self._max_pole_angle_seen = 0.0
        self._max_cart_position_seen = 0.0
        self._last_reason = None

        return self._obs_from_state(), {}

    def step(self, action):
        step_start = time.perf_counter()

        if isinstance(action, np.ndarray):
            action_val = float(action.reshape(-1)[0])
        else:
            action_val = float(action)
        action_val = max(-1.0, min(1.0, action_val))

        msg = Float_V()
        msg.data.append(action_val)
        self._state_event.clear()
        self._action_pub.publish(msg)
        if not self._state_event.wait(timeout=self.step_timeout):
            print("[GazeboInvertedDoublePendulumEnv] WARN: step state wait timed out")
            self._last_reason = "timeout"
            return self._obs_from_state(), 0.0, True, False, {"timeout": True}

        obs = self._obs_from_state()
        cart_pos, pole_ang, pole2_ang = self._latest_state[:3]

        reward = ALIVE_BONUS
        terminated = (
            not np.isfinite(obs).all()
            or abs(pole_ang) > POLE_ANGLE_THRESHOLD
            or abs(pole2_ang) > POLE_ANGLE_THRESHOLD
        )
        self.current_step += 1
        self._total_steps += 1
        truncated = (not terminated) and self.current_step >= self.max_episode_steps

        max_pole_now = max(abs(pole_ang), abs(pole2_ang))
        if max_pole_now > abs(self._max_pole_angle_seen):
            self._max_pole_angle_seen = max_pole_now if pole_ang > 0 else -max_pole_now
        if abs(cart_pos) > abs(self._max_cart_position_seen):
            self._max_cart_position_seen = cart_pos
        step_ms = (time.perf_counter() - step_start) * 1000.0
        self._step_times_ms.append(step_ms)

        if terminated:
            self._last_reason = "terminated"
        elif truncated:
            self._last_reason = "truncated"

        self.episode_reward += reward
        self._publish_metrics(step_ms)
        info = {}
        if truncated:
            info["TimeLimit.truncated"] = True

        if self._debug and (self.current_step % self._verbose_every == 0):
            print(
                f"[InvDoublePendulum][ep {self.current_episode} step {self.current_step}] "
                f"action={action_val:+.3f} reward={reward:.1f} "
                f"cum={self.episode_reward:.1f} "
                f"cart_p={cart_pos:+.3f} pole1_a={pole_ang:+.3f} pole2_a={pole2_ang:+.3f} "
                f"step_ms={step_ms:.1f}"
            )

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

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
        if len(msg.data) >= 6:
            self._latest_state = [float(msg.data[i]) for i in range(6)]
            self._state_event.set()
