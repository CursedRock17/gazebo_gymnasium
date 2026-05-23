from abc import ABC, abstractmethod
from typing import Optional

import gymnasium as gym
import numpy as np


class FixtureEnv(gym.Env, ABC):
    """
    Base class for Gymnasium environments using gz.sim8.TestFixture.

    The training process IS the Gazebo server — zero IPC, zero DDS, zero
    threads.  Physics is advanced by calling server.run(True, N, False), which
    blocks in the calling thread until exactly N physics steps complete.

    Lifecycle (callbacks fired by gz.sim8 during server.run):
      configure   — called on the FIRST pre_update; look up joints, enable ECM checks
      pre_update  — subsequent calls apply action or reset
      post_update — called after each physics step; read observation

    Note: TestFixture has no on_configure callback.  Entity setup happens on the
    first pre_update call, exactly as shown in the gz-sim Python API examples.

    Subclasses must implement all abstract methods.  Do NOT import gz.sim8 at
    the top of this file — the lazy import in __init__ keeps unit tests working
    on Python environments without Gazebo installed.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sdf_path: str,
        observation_space: gym.Space,
        action_space: gym.Space,
        steps_per_action: int = 10,
    ):
        # gz.sim8 is only available when Gazebo Harmonic is installed.
        from gz.sim8 import TestFixture  # noqa: PLC0415

        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self._steps_per_action = steps_per_action
        self._current_step = 0
        self._current_episode = 0
        self._current_action = None
        self._last_obs: Optional[np.ndarray] = None
        self._do_reset = False
        self._configured = False  # True after first pre_update entity setup

        fixture = TestFixture(str(sdf_path))
        fixture.on_pre_update(self._on_pre_update)
        fixture.on_post_update(self._on_post_update)
        fixture.finalize()
        self._server = fixture.server()

        # One step triggers entity setup in pre_update then seeds _last_obs.
        self._server.run(True, 1, False)

    # ------------------------------------------------------------------ gz callbacks

    def _on_pre_update(self, info, ecm):
        if not self._configured:
            # First call: look up entity IDs and enable ECM position/velocity checks.
            self.configure(ecm)
            self._configured = True
            return  # no action on the initial step
        if info.paused:
            return
        if self._do_reset:
            self.apply_reset(ecm)
            self._do_reset = False
        elif self._current_action is not None:
            self.apply_action_to_ecm(ecm, self._current_action)

    def _on_post_update(self, info, ecm):
        if self._configured and not info.paused:
            self._last_obs = self.read_observation(ecm)

    # ------------------------------------------------------------------ Gymnasium interface

    def step(self, action):
        self._current_action = action
        self._server.run(True, self._steps_per_action, False)
        self._current_step += 1
        obs = self._last_obs
        reward = self.get_reward(action)
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        return obs, reward, terminated, truncated, self.get_info()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._do_reset = True
        self._current_action = None
        # One step fires apply_reset in pre_update, then reads obs in post_update.
        self._server.run(True, 1, False)
        self._current_step = 0
        self._current_episode += 1
        return self.set_default_observation(), self.get_info()

    # ------------------------------------------------------------------ abstract

    @abstractmethod
    def configure(self, ecm):
        """Called once on first pre_update. Look up entity IDs; enable ECM checks."""

    @abstractmethod
    def apply_action_to_ecm(self, ecm, action):
        """Write action (forces, velocities) to the ECM. Called in pre_update."""

    @abstractmethod
    def apply_reset(self, ecm):
        """Reset joints to initial state. Called in pre_update when _do_reset is set."""

    @abstractmethod
    def read_observation(self, ecm) -> np.ndarray:
        """Read current state from ECM. Called in post_update. Return float32 array."""

    @abstractmethod
    def get_reward(self, action) -> float:
        """Compute reward from internal state (updated by read_observation)."""

    @abstractmethod
    def is_terminated(self) -> bool:
        """True if the episode reached a terminal state."""

    @abstractmethod
    def is_truncated(self) -> bool:
        """True if the episode hit its step limit."""

    @abstractmethod
    def set_default_observation(self) -> np.ndarray:
        """Zero internal state, return the initial observation for a new episode."""

    def get_info(self) -> dict:
        return {}
