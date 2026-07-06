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

"""Client VecEnv for the batched in-sim harness.

Talks to the ``MultiAgentHarness`` world plugin over exactly three gz-transport
topics regardless of agent count:

  publish  /rl/actions       (flattened N*act_dim)
  publish  /rl/reset         (in-place reset command, optional [seed])
  receive  /rl/observations  (flattened N*obs_dim, one per sim tick)

So this is O(1) in N (one publish + one subscribe per step) and resets in place
(no delete/re-create), fixing the per-agent env's O(N) discovery timeouts and
the respawn overlap race. The reward / termination / group-auto-reset logic is
identical to the per-agent env (same AgentSpec).
"""

import os
import threading
import time
from typing import Any
from typing import Optional

from gymnasium.spaces import Discrete
from gz.msgs10.float_v_pb2 import Float_V
from gz.transport13 import AdvertiseMessageOptions
from gz.transport13 import Node
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from .agent_spec import AgentSpec
from .agent_spec import get_spec


ACTION_TOPIC = "/rl/actions"
RESET_TOPIC = "/rl/reset"
OBS_TOPIC = "/rl/observations"


class HarnessVecEnv(VecEnv):
    """N agents in one sim via the batched harness plugin (SB3 VecEnv)."""

    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(self, spec: AgentSpec, n_agents: int = 4,
                 world_name: Optional[str] = None,
                 max_episode_steps: Optional[int] = None,
                 frame_skip: Optional[int] = None,
                 reset_timeout: float = 3.0,
                 step_timeout: float = 2.0):
        if n_agents < 1:
            raise ValueError("n_agents must be >= 1")
        self.spec = spec
        self.n_agents = n_agents
        self.world_name = world_name or f"{spec.name}_multi"
        self.max_episode_steps = (max_episode_steps if max_episode_steps
                                  is not None else spec.max_episode_steps)
        self.frame_skip = frame_skip if frame_skip is not None else spec.frame_skip
        self.reset_timeout = reset_timeout
        self.step_timeout = step_timeout

        super().__init__(n_agents, spec.observation_space, spec.action_space)

        self._obs_dim = spec.obs_dim
        self._discrete = isinstance(spec.action_space, Discrete)
        self._act_dim = 1 if self._discrete else int(spec.action_space.shape[0])

        # obs cache: the plugin publishes one obs frame per sim tick; we release
        # a step after frame_skip frames (the decision interval).
        self._latest_obs = np.zeros((n_agents, self._obs_dim), dtype=np.float32)
        self._obs_lock = threading.Lock()
        self._obs_event = threading.Event()
        self._obs_count = 0

        self._dones = np.zeros(n_agents, dtype=bool)
        self._steps_since_reset = 0
        self._current_episode = 0
        self._episode_rewards = np.zeros(n_agents, dtype=np.float32)
        self._seed = None
        self._debug = os.environ.get(
            "GAZEBO_GYM_VERBOSE", "false").lower() in ("1", "true", "yes", "on")

        self._node = Node()
        self._act_pub = self._node.advertise(
            ACTION_TOPIC, Float_V, AdvertiseMessageOptions())
        self._reset_pub = self._node.advertise(
            RESET_TOPIC, Float_V, AdvertiseMessageOptions())
        self._node.subscribe(Float_V, OBS_TOPIC, self._on_obs)

        print(f"[HarnessVecEnv] ready (agent={spec.name!r}, "
              f"world={self.world_name!r}, n_agents={n_agents}, "
              f"frame_skip={self.frame_skip})")

    # ------------------------------------------------------------------ #

    def _on_obs(self, msg):
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size != self.n_agents * self._obs_dim:
            return
        with self._obs_lock:
            self._latest_obs = data.reshape(self.n_agents, self._obs_dim)
            self._obs_count += 1
            if self._obs_count >= self.frame_skip:
                self._obs_event.set()

    def _wait_frame(self, timeout: float) -> bool:
        with self._obs_lock:
            self._obs_count = 0
            self._obs_event.clear()
        return self._obs_event.wait(timeout=timeout)

    # ------------------------------------------------------------------ #
    # VecEnv API
    # ------------------------------------------------------------------ #

    def reset(self) -> np.ndarray:
        msg = Float_V()
        if self._seed is not None:
            msg.data.append(float(self._seed))
            self._seed = None
        self._reset_pub.publish(msg)
        if not self._wait_frame(self.reset_timeout):
            print("[HarnessVecEnv] WARN: reset obs wait timed out")

        if self._steps_since_reset > 0:
            print(f"[EpisodeSummary] ep={self._current_episode} "
                  f"steps={self._steps_since_reset} "
                  f"agents_done={int(self._dones.sum())}/{self.n_agents} "
                  f"mean_reward={float(self._episode_rewards.mean()):.1f}")

        self._dones[:] = False
        self._steps_since_reset = 0
        self._current_episode += 1
        self._episode_rewards[:] = 0.0
        with self._obs_lock:
            return self._latest_obs.copy()

    def step_async(self, actions: np.ndarray) -> None:
        flat = np.asarray(actions, dtype=np.float32).reshape(-1)
        msg = Float_V()
        msg.data.extend(flat.tolist())
        with self._obs_lock:
            self._obs_count = 0
            self._obs_event.clear()
        self._act_pub.publish(msg)
        self._pending_step_start = time.perf_counter()

    def step_wait(self):
        if not self._obs_event.wait(timeout=self.step_timeout):
            print("[HarnessVecEnv] WARN: step obs wait timed out")
            self._dones[:] = True
        with self._obs_lock:
            obs = self._latest_obs.copy()

        rewards = np.zeros(self.n_agents, dtype=np.float32)
        for i in range(self.n_agents):
            if self._dones[i]:
                continue
            if self.spec.terminated_fn(obs[i]):
                self._dones[i] = True
            else:
                rewards[i] = float(self.spec.reward_fn(obs[i], None))

        self._steps_since_reset += 1
        self._episode_rewards += rewards
        truncated = self._steps_since_reset >= self.max_episode_steps
        group_reset = bool(self._dones.all() or truncated)

        dones = np.zeros(self.n_agents, dtype=bool)
        infos = [{} for _ in range(self.n_agents)]

        if self._debug:
            step_ms = (time.perf_counter() - self._pending_step_start) * 1000.0
            print(f"[Harness step {self._steps_since_reset}] "
                  f"alive={int((~self._dones).sum())}/{self.n_agents} "
                  f"mean_reward={float(rewards.mean()):.2f} step_ms={step_ms:.1f}")

        if group_reset:
            terminal_obs = obs.copy()
            dones[:] = True
            reset_obs = self.reset()
            for i in range(self.n_agents):
                infos[i]["terminal_observation"] = terminal_obs[i]
                if truncated:
                    infos[i]["TimeLimit.truncated"] = True
            return reset_obs, rewards, dones, infos

        return obs, rewards, dones, infos

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices=None):
        idx = range(self.n_agents) if indices is None else indices
        return [getattr(self, attr_name, None) for _ in idx]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs):
        idx = range(self.n_agents) if indices is None else indices
        method = getattr(self, method_name)
        return [method(*args, **kwargs) for _ in idx]

    def env_is_wrapped(self, wrapper_class, indices=None):
        idx = range(self.n_agents) if indices is None else indices
        return [False for _ in idx]

    def seed(self, seed: Optional[int] = None):
        self._seed = seed
        return [seed for _ in range(self.n_agents)]


def make_harness(name: str, n_agents: int = 4,
                 world_name: Optional[str] = None, **kwargs):
    """Build a HarnessVecEnv for a registered agent (batched-harness backend)."""
    return HarnessVecEnv(get_spec(name), n_agents=n_agents,
                         world_name=world_name, **kwargs)
