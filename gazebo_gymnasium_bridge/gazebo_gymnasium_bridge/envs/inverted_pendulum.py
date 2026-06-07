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

"""GazeboInvertedPendulumEnv: Gazebo port of Gymnasium's InvertedPendulum-v5.

Plain Gymnasium env matching the canonical InvertedPendulum-v5 semantics:

  Observation: Box(4,) float32  [cart_pos, pole_angle, cart_vel, pole_ang_vel]
  Action:      Box(1,) float32  in [-3, 3]  (motor torque before gear scaling)
  Reward:      +1 per step (alive)
  Terminate:   |pole_angle| > 0.2 rad (~11.5°). Note: no cart-position bound;
               the slider's physical joint limit (±1 m) takes care of that.
  Truncate:    step >= max_episode_steps (default 1000)

The sync-gate plugin applies the action as a JOINT FORCE on the slider after
multiplying by gear=100 (matches MJCF `motor gear="100"`). SB3 SAC works
directly with this Box action space — no normalization needed on the env side.

Usage:

    import gymnasium as gym
    from gazebo_gymnasium_bridge.envs import GazeboInvertedPendulumEnv

    env = GazeboInvertedPendulumEnv()
    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
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

# Matches the canonical InvertedPendulum-v5 termination — only pole angle.
POLE_ANGLE_THRESHOLD = 0.2  # radians (~11.5°)

# Must match InvertedPendulumSyncGate.frame_skip; used only for the
# `phys_ticks` field in the EpisodeSummary log.
DEFAULT_FRAME_SKIP = 4


class GazeboInvertedPendulumEnv(gym.Env):
    """Standalone Gymnasium env for the Gazebo InvertedPendulum world.

    Continuous action, mirrors Gymnasium's InvertedPendulum-v5 contract.
    """

    metadata = {"render_modes": [], "render_fps": 25}

    def __init__(self, world_name: str = "inverted_pendulum",
                 max_episode_steps: int = 1000,
                 frame_skip: int = DEFAULT_FRAME_SKIP,
                 reset_timeout: float = 5.0,
                 step_timeout: float = 5.0,
                 render_mode: Optional[str] = None):
        super().__init__()

        # Spaces — match canonical InvertedPendulum-v5.
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32,
        )
        self.action_space = Box(
            low=-3.0, high=3.0, shape=(1,), dtype=np.float32,
        )
        self.render_mode = render_mode

        self.world_name = world_name
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.reset_timeout = reset_timeout
        self.step_timeout = step_timeout

        # Episode bookkeeping.
        self.current_step = 0
        self.current_episode = 0
        self.episode_reward = 0.0
        self.episode_reward_list = []
        self.episode_steps_list = []

        # Per-episode telemetry for the rich [EpisodeSummary] line.
        self._step_times_ms = []
        self._max_pole_angle_seen = 0.0
        self._max_cart_position_seen = 0.0
        self._last_reason = None

        # Verbose per-step logging — opt-in via env var, same protocol as
        # GazeboCartPoleEnv so a shared GAZEBO_GYM_VERBOSE works across envs.
        self._debug = os.environ.get(
            "GAZEBO_GYM_VERBOSE", "false").lower() in ("1", "true", "yes", "on")
        try:
            self._verbose_every = max(1, int(os.environ.get("GAZEBO_GYM_VERBOSE_EVERY", "1")))
        except ValueError:
            self._verbose_every = 1

        # Transport: action publisher + state subscriber + world_control client.
        self._action_node = Node()
        self._action_pub = self._action_node.advertise(
            "/env/action", Float_V, AdvertiseMessageOptions())

        # /env/metrics publisher — see backend/metrics.py for the contract.
        self._metrics = MetricsPublisher("gazebo_inverted_pendulum_env_metrics")
        self._step_window = []  # type: list[tuple[float, float]]
        self._total_steps = 0

        self._latest_state = [0.0, 0.0, 0.0, 0.0]
        self._state_event = threading.Event()

        self._state_node = Node()
        self._state_node.subscribe(Float_V, "/env/state", self._on_state)

        self._world_control = world_control.WorldController(world_name, steps_per_action=0)

        print(f"[GazeboInvertedPendulumEnv] ready (world={world_name!r}, "
              f"frame_skip={frame_skip}, max_episode_steps={max_episode_steps}, "
              f"verbose={self._debug})")

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if self.current_step > 0:
            self.episode_reward_list.append(self.episode_reward)
            self.episode_steps_list.append(self.current_step)
            phys_ticks = self.current_step * self.frame_skip
            reason = self._last_reason or "unknown"
            mean_step_ms = (sum(self._step_times_ms) / len(self._step_times_ms)
                            if self._step_times_ms else 0.0)
            last_cart, last_pole, _vc, _vp = self._latest_state
            print(
                f"[EpisodeSummary] ep={self.current_episode} "
                f"steps={self.current_step} phys_ticks={phys_ticks} "
                f"reward={self.episode_reward} reason={reason} "
                f"max_pole_a={self._max_pole_angle_seen:+.3f} "
                f"max_cart_p={self._max_cart_position_seen:+.3f} "
                f"final_pole_a={last_pole:+.3f} final_cart_p={last_cart:+.3f} "
                f"mean_step_ms={mean_step_ms:.1f}"
            )

        self._state_event.clear()
        self._world_control.reset(pause_after=False)
        if not self._state_event.wait(timeout=self.reset_timeout):
            print("[GazeboInvertedPendulumEnv] WARN: reset state wait timed out")

        self.current_step = 0
        self.current_episode += 1
        self.episode_reward = 0.0
        self._step_times_ms = []
        self._max_pole_angle_seen = 0.0
        self._max_cart_position_seen = 0.0
        self._last_reason = None

        obs = np.asarray(self._latest_state, dtype=np.float32)
        return obs, {}

    def step(self, action):
        step_start = time.perf_counter()

        # Pack the action — SB3 SAC produces np.ndarray shape (1,) float32.
        if isinstance(action, np.ndarray):
            action_val = float(action.reshape(-1)[0])
        else:
            action_val = float(action)
        # Clip to declared action space bounds — SAC's policy can occasionally
        # produce values just past ±3 due to its squashing transform; our
        # plugin's gear multiplication would amplify them otherwise.
        action_val = max(-3.0, min(3.0, action_val))

        msg = Float_V()
        msg.data.append(action_val)
        self._state_event.clear()
        self._action_pub.publish(msg)
        if not self._state_event.wait(timeout=self.step_timeout):
            print("[GazeboInvertedPendulumEnv] WARN: step state wait timed out")
            self._last_reason = "timeout"
            obs = np.asarray(self._latest_state, dtype=np.float32)
            return obs, 0.0, True, False, {"timeout": True}

        cart_pos, pole_angle, cart_vel, pole_ang_vel = self._latest_state
        obs = np.asarray([cart_pos, pole_angle, cart_vel, pole_ang_vel], dtype=np.float32)

        # Canonical InvertedPendulum-v5: +1 reward per step alive, terminate
        # only when pole tips past 0.2 rad OR any obs becomes non-finite.
        reward = 1.0
        terminated = (not np.isfinite(obs).all()) or (abs(pole_angle) > POLE_ANGLE_THRESHOLD)
        self.current_step += 1
        self._total_steps += 1
        truncated = (not terminated) and self.current_step >= self.max_episode_steps

        # Telemetry
        if abs(pole_angle) > abs(self._max_pole_angle_seen):
            self._max_pole_angle_seen = pole_angle
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
                f"[InvPendulum][ep {self.current_episode} step {self.current_step}] "
                f"action={action_val:+.3f} reward={reward} cum={self.episode_reward} "
                f"cart=(p={cart_pos:+.3f}, v={cart_vel:+.3f}) "
                f"pole=(a={pole_angle:+.3f}, av={pole_ang_vel:+.3f}) "
                f"step_ms={step_ms:.1f}"
            )

        return obs, reward, terminated, truncated, info

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

    def render(self):
        """Gz sim handles rendering."""
        pass

    def close(self):
        pass

    # ------------------------------------------------------------------ #

    def _on_state(self, msg):
        if len(msg.data) >= 4:
            self._latest_state = [float(msg.data[i]) for i in range(4)]
            self._state_event.set()
