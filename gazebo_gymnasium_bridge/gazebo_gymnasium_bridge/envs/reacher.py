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

"""GazeboReacherEnv — Gazebo port of Gymnasium's Reacher-v5.

Two-link planar arm; goal is to bring the fingertip to a randomized 2D
target within the arena. Canonical Reacher uses a continuous dense
reward (negative distance + small control penalty) — easier to learn
than CartPole's sparse +1 signal.

Observation (Box(10,) float32) — matches Gymnasium Reacher-v5 layout:
    [0:2] cos(theta0), cos(theta1)
    [2:4] sin(theta0), sin(theta1)
    [4:6] target_x, target_y               (virtual target, env-side)
    [6:8] dtheta0, dtheta1                  (joint velocities)
    [8:10] fingertip_x - target_x,
           fingertip_y - target_y

Action (Box(2,) float32 in [-1, 1]): torques on joint0, joint1. Scaled
by gear=200 inside the sync-gate plugin before being applied as joint
force.

Reward (canonical Reacher):
    reward = -|fingertip - target| - 0.1 * |action|^2

Termination: never (the env relies on max_episode_steps for truncation;
canonical Reacher uses 50 steps per episode).
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


# Target sampling region — matches the canonical Gymnasium Reacher (target
# drawn uniformly from a 0.2-radius disk around the origin).
TARGET_RADIUS = 0.2

# Control cost coefficient on the L2 norm of the action. Canonical: 0.1.
CTRL_COST_COEF = 0.1

DEFAULT_FRAME_SKIP = 2
DEFAULT_MAX_EPISODE_STEPS = 50


class GazeboReacherEnv(gym.Env):
    """Gazebo Reacher-v5 port — dense reward, fast to converge."""

    metadata = {"render_modes": [], "render_fps": 50}

    def __init__(self, world_name: str = "reacher",
                 max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
                 frame_skip: int = DEFAULT_FRAME_SKIP,
                 reset_timeout: float = 5.0,
                 step_timeout: float = 5.0,
                 render_mode: Optional[str] = None):
        super().__init__()

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32,
        )
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
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
        self._total_steps = 0
        self._last_reason = None

        # Virtual target — sampled fresh on every reset(). Lives env-side
        # only (not part of the SDF). Both Reacher's reward and obs use it.
        self._target = np.zeros(2, dtype=np.float32)

        self._debug = os.environ.get(
            "GAZEBO_GYM_VERBOSE", "false").lower() in ("1", "true", "yes", "on")

        self._action_node = Node()
        self._action_pub = self._action_node.advertise(
            "/env/action", Float_V, AdvertiseMessageOptions())

        self._metrics = MetricsPublisher("gazebo_reacher_env_metrics")
        self._step_window = []  # type: list[tuple[float, float]]

        # Latest [theta0, theta1, dtheta0, dtheta1, fingertip_x, fingertip_y].
        self._latest_state = [0.0, 0.0, 0.0, 0.0, 0.21, 0.0]
        self._state_event = threading.Event()
        self._state_node = Node()
        self._state_node.subscribe(Float_V, "/env/state", self._on_state)

        self._world_control = world_control.WorldController(world_name, steps_per_action=0)

        print(f"[GazeboReacherEnv] ready (world={world_name!r}, "
              f"frame_skip={frame_skip})")

    def _sample_target(self) -> np.ndarray:
        # Uniform sample within a disk of radius TARGET_RADIUS centered at
        # the arena origin. Rejection-sample to keep the distribution flat.
        while True:
            x = self.np_random.uniform(-TARGET_RADIUS, TARGET_RADIUS)
            y = self.np_random.uniform(-TARGET_RADIUS, TARGET_RADIUS)
            if x * x + y * y <= TARGET_RADIUS * TARGET_RADIUS:
                return np.array([x, y], dtype=np.float32)

    def _obs_from_state(self) -> np.ndarray:
        theta0, theta1, dtheta0, dtheta1, fx, fy = self._latest_state
        return np.array([
            np.cos(theta0), np.cos(theta1),
            np.sin(theta0), np.sin(theta1),
            self._target[0], self._target[1],
            dtheta0, dtheta1,
            fx - self._target[0], fy - self._target[1],
        ], dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if self.current_step > 0:
            reason = self._last_reason or "truncated"
            print(
                f"[EpisodeSummary] ep={self.current_episode} "
                f"steps={self.current_step} reward={self.episode_reward:.3f} "
                f"reason={reason} target=({self._target[0]:+.3f}, "
                f"{self._target[1]:+.3f})"
            )

        self._state_event.clear()
        self._world_control.reset(pause_after=False)
        if not self._state_event.wait(timeout=self.reset_timeout):
            print("[GazeboReacherEnv] WARN: reset state wait timed out")

        self._target = self._sample_target()
        self.current_step = 0
        self.current_episode += 1
        self.episode_reward = 0.0
        self._last_reason = None

        return self._obs_from_state(), {"target": self._target.tolist()}

    def step(self, action):
        step_start = time.perf_counter()

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < 2:
            raise ValueError(f"Action must have 2 elements, got {action.shape}")
        clipped = np.clip(action[:2], -1.0, 1.0)

        msg = Float_V()
        msg.data.extend([float(clipped[0]), float(clipped[1])])
        self._state_event.clear()
        self._action_pub.publish(msg)
        if not self._state_event.wait(timeout=self.step_timeout):
            print("[GazeboReacherEnv] WARN: step state wait timed out")
            self._last_reason = "timeout"
            return self._obs_from_state(), 0.0, True, False, {"timeout": True}

        obs = self._obs_from_state()
        fx, fy = self._latest_state[4], self._latest_state[5]
        dist = float(np.hypot(fx - self._target[0], fy - self._target[1]))
        ctrl_cost = float(CTRL_COST_COEF * np.sum(np.square(clipped)))
        reward = -dist - ctrl_cost

        self.current_step += 1
        self._total_steps += 1
        terminated = not np.isfinite(obs).all()
        truncated = (not terminated) and self.current_step >= self.max_episode_steps
        if terminated:
            self._last_reason = "non_finite"
        elif truncated:
            self._last_reason = "truncated"

        self.episode_reward += reward

        step_ms = (time.perf_counter() - step_start) * 1000.0
        self._publish_metrics(step_ms)

        info = {"distance": dist, "ctrl_cost": ctrl_cost,
                "target": self._target.tolist()}
        if truncated:
            info["TimeLimit.truncated"] = True

        if self._debug:
            print(
                f"[Reacher][ep {self.current_episode} step {self.current_step}] "
                f"a=({clipped[0]:+.2f},{clipped[1]:+.2f}) dist={dist:.3f} "
                f"reward={reward:+.3f} cum={self.episode_reward:+.3f} "
                f"step_ms={step_ms:.1f}"
            )

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

    def _publish_metrics(self, step_ms: float) -> None:
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
