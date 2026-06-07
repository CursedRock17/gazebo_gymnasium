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

"""
GazeboCartPoleEnv — plain Gymnasium env that talks to a running gz sim world.

Usage (near-identical to MuJoCo):

    import gymnasium as gym
    from gazebo_gymnasium_bridge.envs import GazeboCartPoleEnv

    env = GazeboCartPoleEnv()
    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()

The world (with the CartPoleSyncGate plugin) must be running. Launch it via:
    ros2 launch gazebo_gymnasium_bringup cartpole.launch.py

Environment variables:
    GAZEBO_GYM_VERBOSE=true       — emit a per-step debug line
    GAZEBO_GYM_VERBOSE_EVERY=N    — when verbose, only print every Nth step
"""

import os
import threading
import time
from typing import Optional

import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete
from gz.msgs10.float_v_pb2 import Float_V
from gz.transport13 import AdvertiseMessageOptions
from gz.transport13 import Node
import numpy as np

from ..backend.metrics import MetricsPublisher
from ..backend.nodes import world_control

# Termination thresholds match the canonical Gymnasium CartPole-v1.
POLE_ANGLE_THRESHOLD = 0.20944   # ≈ 12 degrees in radians
CART_POSITION_THRESHOLD = 2.4    # meters

# Must match CartPoleSyncGate.frame_skip in cartpole_learner.py. Only used for
# the phys_ticks field in the EpisodeSummary line — wrong value just makes the
# reported physics-tick count off, doesn't affect training.
DEFAULT_FRAME_SKIP = 5


class GazeboCartPoleEnv(gym.Env):
    """Standalone Gymnasium env for Gazebo CartPole.

    No plugin code on the agent side — communicates with the gz-sim sync-gate plugin via two
    topics:

    publishes  /env/action  (Float_V, one element) subscribes /env/state   (Float_V, four sensor
    values)

    `step()` publishes the action and blocks on the next /env/state callback. The plugin only
    publishes state once per frame_skip ticks, so each step waits for one full physics window
    before returning.
    """

    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(self, world_name: str = "cartpole",
                 max_episode_steps: int = 500,
                 frame_skip: int = DEFAULT_FRAME_SKIP,
                 reset_timeout: float = 5.0,
                 step_timeout: float = 5.0,
                 render_mode: Optional[str] = None):
        super().__init__()

        # Canonical CartPole-v1 spaces.
        self.observation_space = Box(
            low=np.array([-4.8, float("-inf"), -0.41887903, float("-inf")], dtype=np.float32),
            high=np.array([4.8, float("inf"), 0.41887903, float("inf")], dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )
        self.action_space = Discrete(2)
        self.render_mode = render_mode

        self.world_name = world_name
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.reset_timeout = reset_timeout
        self.step_timeout = step_timeout

        # Episode bookkeeping (for [EpisodeSummary] logging + truncation).
        self.current_step = 0
        self.current_episode = 0
        self.episode_reward = 0.0
        self.episode_reward_list = []
        self.episode_steps_list = []

        # Per-step debug + telemetry. Reset in _reset_episode_telemetry().
        self._step_times_ms = []
        self._max_pole_angle_seen = 0.0
        self._max_cart_position_seen = 0.0
        # Reason the previous step ended the episode (terminated / truncated /
        # timeout). Read in reset() to print [EpisodeSummary]. None = not ended.
        self._last_reason = None

        # Verbose per-step logging — opt-in via env var so production runs
        # aren't flooded.
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

        # /env/metrics publisher — emits a gazebo_gymnasium_msgs/EnvMetrics
        # ROS message every env.step(). Foxglove plots can address each
        # value by field name (e.g. `episode_reward`, `total_steps`)
        # instead of array index.
        self._metrics = MetricsPublisher("gazebo_cartpole_env_metrics")
        # Sliding window of (timestamp, step_ms) used to compute steps/sec.
        self._step_window = []  # type: list[tuple[float, float]]
        # Cumulative step counter — survives episode resets so Foxglove can
        # plot reward-vs-total-training-steps for long runs.
        self._total_steps = 0

        # Latest state from the plugin. `_state_event` is set by the callback
        # so step() / reset() can block on a specific tick.
        self._latest_state = [0.0, 0.0, 0.0, 0.0]
        self._state_event = threading.Event()

        self._state_node = Node()
        self._state_node.subscribe(Float_V, "/env/state", self._on_state)

        self._world_control = world_control.WorldController(world_name, steps_per_action=0)

        print(f"[GazeboCartPoleEnv] ready (world={world_name!r}, "
              f"frame_skip={frame_skip}, max_episode_steps={max_episode_steps}, "
              f"verbose={self._debug})")

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Episode summary for the just-ended episode (skip the synthetic
        # before-any-episode call).
        if self.current_step > 0:
            self.episode_reward_list.append(self.episode_reward)
            self.episode_steps_list.append(self.current_step)
            phys_ticks = self.current_step * self.frame_skip
            reason = self._last_reason or "unknown"
            mean_step_ms = (sum(self._step_times_ms) / len(self._step_times_ms)
                            if self._step_times_ms else 0.0)
            last_cart, last_cart_vel, last_pole, last_pole_av = self._latest_state
            print(
                f"[EpisodeSummary] ep={self.current_episode} "
                f"steps={self.current_step} phys_ticks={phys_ticks} "
                f"reward={self.episode_reward} reason={reason} "
                f"max_pole_a={self._max_pole_angle_seen:+.3f} "
                f"max_cart_p={self._max_cart_position_seen:+.3f} "
                f"final_pole_a={last_pole:+.3f} final_cart_p={last_cart:+.3f} "
                f"mean_step_ms={mean_step_ms:.1f}"
            )

        # Trigger a world reset. WorldController.reset already retries with
        # short timeouts so startup races resolve quickly.
        self._state_event.clear()
        self._world_control.reset(pause_after=False)

        # Plugin's Reset event handler publishes /env/state once.
        if not self._state_event.wait(timeout=self.reset_timeout):
            print("[GazeboCartPoleEnv] WARN: reset state wait timed out")

        self.current_step = 0
        self.current_episode += 1
        self.episode_reward = 0.0
        self._reset_episode_telemetry()

        obs = np.asarray(self._latest_state, dtype=np.float32)
        return obs, {}

    def step(self, action):
        step_start = time.perf_counter()

        # Pack the action into a single-element Float_V.
        if isinstance(action, np.ndarray):
            action_val = float(action.item())
        else:
            action_val = float(action)
        msg = Float_V()
        msg.data.append(action_val)

        # Publish + park on the next /env/state callback.
        self._state_event.clear()
        self._action_pub.publish(msg)
        if not self._state_event.wait(timeout=self.step_timeout):
            # Plugin didn't publish state in time — return current cached state
            # but flag termination so the agent resets. Better than hanging.
            print("[GazeboCartPoleEnv] WARN: step state wait timed out")
            self._last_reason = "timeout"
            obs = np.asarray(self._latest_state, dtype=np.float32)
            return obs, 0.0, True, False, {"timeout": True}

        cart_pos, cart_vel, pole_angle, pole_ang_vel = self._latest_state
        obs = np.asarray([cart_pos, cart_vel, pole_angle, pole_ang_vel], dtype=np.float32)

        # CartPole-v1 reward + termination — computed on the env side so the
        # plugin stays generic.
        reward = 1.0
        terminated = (abs(pole_angle) > POLE_ANGLE_THRESHOLD
                      or abs(cart_pos) > CART_POSITION_THRESHOLD)
        self.current_step += 1
        self._total_steps += 1
        truncated = (not terminated) and self.current_step >= self.max_episode_steps

        # Telemetry — running maxima for the episode summary.
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
            action_int = int(action_val)
            print(
                f"[CartPole][ep {self.current_episode} step {self.current_step}] "
                f"action={action_int} reward={reward} cum={self.episode_reward} "
                f"cart=(p={cart_pos:+.3f}, v={cart_vel:+.3f}) "
                f"pole=(a={pole_angle:+.3f}, av={pole_ang_vel:+.3f}) "
                f"step_ms={step_ms:.1f}"
            )

        return obs, reward, terminated, truncated, info

    def render(self):
        """Gz sim handles rendering; nothing to do here.

        To get a rendered window, run `gz sim` with the GUI (default behavior of the launch file).
        """
        pass

    def close(self):
        """Nothing to clean up — transport nodes shut down on GC."""
        pass

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _reset_episode_telemetry(self):
        self._step_times_ms = []
        self._max_pole_angle_seen = 0.0
        self._max_cart_position_seen = 0.0
        self._last_reason = None

    def _publish_metrics(self, step_ms: float) -> None:
        """Push an EnvMetrics ROS message to /env/metrics (see metrics.py)."""
        now = time.perf_counter()
        # Keep only timestamps in the last 1 s for the steps/sec calc.
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
        # msg.data is the 4-element sensor vector from the plugin.
        if len(msg.data) >= 4:
            self._latest_state = [float(msg.data[i]) for i in range(4)]
            self._state_event.set()
