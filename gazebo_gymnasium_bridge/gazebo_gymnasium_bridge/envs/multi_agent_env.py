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

"""Generalized N-agents-in-one-sim Gymnasium VecEnv, driven by an AgentSpec.

This collapses the per-model multi envs into a single class: spawn N copies
of one model (described by an :class:`AgentSpec`) into one Gazebo world,
fan actions out to N publishers, read each model's ``joint_state`` for
observations, and apply the spec's reward / termination. ``MultiCartPole``,
``MultiAnt``, … are just this class with a different spec — see
``make_multi``.

Single-agent is the degenerate ``n_agents=1`` case; there is no separate
single-agent env class.

Actuation note: actions are published as ``Float_V`` on ``/env/action_<i>``,
consumed by a world-level controller plugin in the world SDF that maps them
to each model's command topic. (A batched in-sim harness is the next
iteration; this keeps the proven per-agent transport.)
"""

import math
import os
import threading
import time
from typing import Any
from typing import Optional

from gymnasium.spaces import Discrete
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.entity_pb2 import Entity
from gz.msgs10.float_v_pb2 import Float_V
from gz.msgs10.model_pb2 import Model as ModelMsg
from gz.transport13 import AdvertiseMessageOptions
from gz.transport13 import Node
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from .agent_spec import AgentSpec
from .agent_spec import get_spec
from ..backend.nodes import world_control


# Per-call timeout for gz Create/Remove service requests.
_GZ_SERVICE_TIMEOUT_MS = 500
# Settle between the bulk delete and the bulk re-create. The gz remove is only
# applied on a sim tick, so this must cover enough ticks for the old model to
# actually be gone before we re-create it under the same name — otherwise the
# old + new overlap, collide, and knock the pole over (and the GUI logs
# "Visual: <name> already exists"). It's wall-clock, so it must be generous
# enough at low real-time factors (e.g. with the GUI attached, RTF < 1). The
# in-sim harness (ECM Joint.reset_position, no respawn) will make this obsolete.
_REMOVE_TO_CREATE_DELAY = 0.3
# Cross-process gz-transport discovery settle after a respawn.
_DISCOVERY_SETTLE = 0.3


class MultiAgentGazeboVecEnv(VecEnv):
    """N copies of one :class:`AgentSpec` in a single gz world, as an SB3 VecEnv.

    Parameters
    ----------
    spec : AgentSpec
        The agent to replicate (model, obs extraction, spaces, reward/term).
    n_agents : int
        Number of agents in the world (>= 1; 1 is the single-agent case).
    world_name : str
        Name of the running gz world.
    max_episode_steps / frame_skip :
        Default to the spec's values; override per construction if given.
    reset_timeout / step_timeout : float
        Seconds to wait for the sim's joint_state publishes. Scale these up
        with n_agents — per-topic discovery is O(N) (16 agents needs ~5s/15s).
    """

    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(self, spec: AgentSpec, n_agents: int = 4,
                 world_name: Optional[str] = None,
                 max_episode_steps: Optional[int] = None,
                 frame_skip: Optional[int] = None,
                 reset_timeout: float = 3.0,
                 step_timeout: float = 1.0):
        if n_agents < 1:
            raise ValueError("n_agents must be >= 1")
        self.spec = spec
        self.n_agents = n_agents
        self.world_name = world_name or f"{spec.name}_multi"
        self.max_episode_steps = (max_episode_steps
                                  if max_episode_steps is not None
                                  else spec.max_episode_steps)
        self.frame_skip = frame_skip if frame_skip is not None else spec.frame_skip
        self.reset_timeout = reset_timeout
        self.step_timeout = step_timeout

        super().__init__(n_agents, spec.observation_space, spec.action_space)

        self._obs_dim = spec.obs_dim
        self._discrete = isinstance(spec.action_space, Discrete)
        self._act_dim = 1 if self._discrete else int(spec.action_space.shape[0])

        # Per-agent obs cache + threading sync.
        self._latest_states = np.zeros((n_agents, self._obs_dim), dtype=np.float32)
        self._state_events = [threading.Event() for _ in range(n_agents)]
        self._state_lock = threading.Lock()
        self._joint_state_counts = np.zeros(n_agents, dtype=int)

        # Per-agent / episode bookkeeping.
        self._dones = np.zeros(n_agents, dtype=bool)
        self._steps_since_reset = 0
        self._current_episode = 0
        self._episode_rewards = np.zeros(n_agents, dtype=np.float32)

        self._debug = os.environ.get(
            "GAZEBO_GYM_VERBOSE", "false").lower() in ("1", "true", "yes", "on")

        # gz transport: N action publishers + N joint_state subscribers.
        self._action_nodes = []
        self._action_pubs = []
        self._joint_state_nodes = []
        for i in range(n_agents):
            anode = Node()
            self._action_pubs.append(
                anode.advertise(f"/env/action_{i}", Float_V,
                                AdvertiseMessageOptions()))
            self._action_nodes.append(anode)

            jnode = Node()
            topic = f"/world/{self.world_name}/model/{spec.name}_{i}/joint_state"
            jnode.subscribe(ModelMsg, topic, self._make_joint_state_callback(i))
            self._joint_state_nodes.append(jnode)

        self._world_control = world_control.WorldController(
            self.world_name, steps_per_action=0)

        print(f"[MultiAgentGazeboVecEnv] ready (agent={spec.name!r}, "
              f"world={self.world_name!r}, n_agents={n_agents}, "
              f"frame_skip={self.frame_skip})")

    # ------------------------------------------------------------------ #
    # Sensor callback
    # ------------------------------------------------------------------ #

    def _make_joint_state_callback(self, idx: int):
        def _cb(msg):
            obs = self.spec.obs_from_joint_state(msg)
            with self._state_lock:
                self._latest_states[idx] = obs
                self._joint_state_counts[idx] += 1
                if self._joint_state_counts[idx] >= self.frame_skip:
                    self._state_events[idx].set()
        return _cb

    # ------------------------------------------------------------------ #
    # VecEnv API
    # ------------------------------------------------------------------ #

    def reset(self) -> np.ndarray:
        self._recreate_all_agents()
        if not self._wait_all_states(self.reset_timeout):
            print("[MultiAgentGazeboVecEnv] WARN: reset state wait timed out")

        if self._steps_since_reset > 0:
            print(
                f"[EpisodeSummary] ep={self._current_episode} "
                f"steps={self._steps_since_reset} "
                f"agents_done={int(self._dones.sum())}/{self.n_agents} "
                f"mean_reward={float(self._episode_rewards.mean()):.1f}"
            )

        self._dones[:] = False
        self._steps_since_reset = 0
        self._current_episode += 1
        self._episode_rewards[:] = 0.0
        return self._latest_states.copy()

    def step_async(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions)
        with self._state_lock:
            self._joint_state_counts[:] = 0
            for ev in self._state_events:
                ev.clear()
        for i in range(self.n_agents):
            msg = Float_V()
            if self._discrete:
                msg.data.append(float(int(actions[i])))
            else:
                for v in np.atleast_1d(actions[i]).astype(float):
                    msg.data.append(float(v))
            self._action_pubs[i].publish(msg)
        self._pending_step_start = time.perf_counter()

    def step_wait(self):
        if not self._wait_all_states(self.step_timeout):
            print("[MultiAgentGazeboVecEnv] WARN: step state wait timed out")
            self._dones[:] = True  # force group reset; safer than stale data

        with self._state_lock:
            obs = self._latest_states.copy()

        # Per-agent reward/termination from the spec. A terminated agent is
        # marked dead internally (stops earning reward) but only reports
        # done=True on the group-reset tick, so each sub-env has exactly one
        # episode boundary (required for correct SB3 bookkeeping).
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
            print(f"[MultiAgent step {self._steps_since_reset}] "
                  f"alive={int((~self._dones).sum())}/{self.n_agents} "
                  f"mean_reward={float(rewards.mean()):.2f} step_ms={step_ms:.1f}")

        if group_reset:
            # SB3 VecEnv auto-reset contract: on done return the FIRST obs of
            # the new episode, stash the final obs as terminal_observation.
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
        pass  # gz transport nodes shut down on GC

    # VecEnv abstract methods SB3 wrappers may call.
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
        return [seed for _ in range(self.n_agents)]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _grid_position(self, index: int) -> tuple:
        n = self.n_agents
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        col = index % cols
        row = index // cols
        x = (col - (cols - 1) / 2.0) * self.spec.x_spacing
        y = (row - (rows - 1) / 2.0) * self.spec.y_spacing
        return x, y

    def _render_sdf(self, index: int) -> str:
        name = f"{self.spec.name}_{index}"
        self_collide = ("<self_collide>true</self_collide>"
                        if self.spec.self_collide else "")
        joints = "".join(
            f'<joint name="{jn}" type="fixed">'
            f"<parent>{parent}</parent><child>{child}</child></joint>"
            for (jn, parent, child) in self.spec.extra_joints
        )
        return (
            '<sdf version="1.8">'
            f'<model name="{name}">'
            f"{self_collide}"
            f'<include merge="true"><uri>{self.spec.model_uri}</uri></include>'
            f"{joints}{self.spec.extra_sdf}"
            "</model></sdf>"
        )

    def _recreate_all_agents(self) -> None:
        """Delete + recreate every agent in the running world.

        Leaves the world itself untouched (world_control.reset() would revert
        to the empty SDF and wipe the dynamically-spawned agents).
        """
        for ev in self._state_events:
            ev.clear()

        transport = self._world_control.world_control_node
        remove_service = f"/world/{self.world_name}/remove"
        create_service = f"/world/{self.world_name}/create"

        for i in range(self.n_agents):
            req = Entity()
            req.name = f"{self.spec.name}_{i}"
            req.type = Entity.MODEL
            transport.request(remove_service, req, Entity, Boolean,
                              _GZ_SERVICE_TIMEOUT_MS)

        time.sleep(_REMOVE_TO_CREATE_DELAY)

        for i in range(self.n_agents):
            x, y = self._grid_position(i)
            req = EntityFactory()
            req.sdf = self._render_sdf(i)
            req.name = f"{self.spec.name}_{i}"
            req.pose.position.x = x
            req.pose.position.y = y
            req.pose.position.z = self.spec.spawn_z  # clear of ground plane
            transport.request(create_service, req, EntityFactory, Boolean,
                              _GZ_SERVICE_TIMEOUT_MS)

        # Entity churn can leave the sim paused; unpause so physics ticks.
        self._world_control.unpause()
        time.sleep(_DISCOVERY_SETTLE)
        with self._state_lock:
            self._joint_state_counts[:] = 0

    def _wait_all_states(self, timeout: float) -> bool:
        deadline = time.perf_counter() + timeout
        for i, ev in enumerate(self._state_events):
            remaining = deadline - time.perf_counter()
            if remaining <= 0 or not ev.wait(timeout=remaining):
                received = [j for j, e in enumerate(self._state_events)
                            if e.is_set()]
                missing = [j for j, e in enumerate(self._state_events)
                           if not e.is_set()]
                print(f"[MultiAgentGazeboVecEnv] _wait_all_states timeout after "
                      f"{timeout:.1f}s: waiting on agent {i}, "
                      f"received={received} missing={missing}")
                return False
        return True


def make_multi(name: str, n_agents: int = 4,
               world_name: Optional[str] = None, **kwargs):
    """Factory: build a MultiAgentGazeboVecEnv for a registered agent name.

    ``make_multi("cartpole", n_agents=16)`` is the canonical entry point;
    "MultiAnt", "MultiHopper", … are simply ``make_multi("ant", ...)`` etc.
    """
    return MultiAgentGazeboVecEnv(get_spec(name), n_agents=n_agents,
                                  world_name=world_name, **kwargs)
