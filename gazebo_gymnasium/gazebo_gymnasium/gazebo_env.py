import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

import gymnasium as gym

from .world_control import WorldController

_STEP_TIMEOUT_S = 5.0
_DISCOVERY_DELAY_S = 1.0


class GazeboEnv(gym.Env, ABC):
    """
    Base class for Gymnasium environments backed by a running Gazebo simulation.

    Architecture: Gazebo runs continuously at real_time_update_rate=0 (unlimited
    speed).  Each env.step() publishes actions then counts N joint-state callbacks
    from the physics engine before reading state — zero blocking WorldControl
    service calls in the hot path.

    Subclasses MUST call _count_physics_step() from their joint-state subscriber
    callback on every message.  The base-class step() handles the rest.

    Subclasses MUST call _ping_world_control() at the end of their __init__, after
    all subscriptions are established.  This verifies connectivity and surfaces a
    clear error if GZ_PARTITION is wrong or Gazebo is not running.

    Episode resets use a blocking WorldControl.reset() service call, which is
    acceptable since resets are rare (once per episode, not every step).
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        world_name: str,
        observation_space: gym.Space,
        action_space: gym.Space,
        steps_per_action: int = 10,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self._world_name = world_name
        self._world_control = WorldController(world_name, steps_per_action)
        self._steps_per_action = steps_per_action
        self._current_step = 0
        self._current_episode = 0

        # Physics step counting — synchronized via joint-state callbacks.
        self._phys_lock = threading.Lock()
        self._phys_steps: int = 0
        self._phys_target: int = 0
        self._phys_event = threading.Event()
        self._phys_counting: bool = False

    # ------------------------------------------------------------------ step counting

    def _count_physics_step(self) -> None:
        """Subclass joint-state callback calls this on every physics tick."""
        with self._phys_lock:
            if not self._phys_counting:
                return
            self._phys_steps += 1
            if self._phys_steps >= self._phys_target:
                self._phys_event.set()

    def _advance_physics(self, n: int = None) -> bool:
        """Unpause Gazebo, count n joint-state callbacks, then pause again.

        Gazebo is paused between calls so no physics runs during Python
        processing.  unpause/pause return immediately (they just set a flag in
        Gazebo), so the only blocking wait is the event for the N callbacks.
        """
        if n is None:
            n = self._steps_per_action
        with self._phys_lock:
            self._phys_target = self._phys_steps + n
            self._phys_event.clear()
            self._phys_counting = True
        self._world_control.unpause()
        ok = self._phys_event.wait(timeout=_STEP_TIMEOUT_S)
        self._world_control.pause()
        with self._phys_lock:
            self._phys_counting = False
        if not ok:
            print(f"[WARN] _advance_physics timed out after {_STEP_TIMEOUT_S}s", flush=True)
        return ok

    def _stop_counting(self) -> None:
        with self._phys_lock:
            self._phys_counting = False

    def _ping_world_control(self) -> None:
        """Sleep for gz-transport discovery, then verify the WorldControl service.

        Call this at the end of a subclass __init__, after all subscriptions are
        established.  Raises RuntimeError with a diagnostic message if the service
        is unreachable (wrong GZ_PARTITION or Gazebo not running).
        """
        time.sleep(_DISCOVERY_DELAY_S)
        gz_part = os.environ.get("GZ_PARTITION", "<not set>")
        cls = type(self).__name__
        print(f"[{cls}] Pinging WorldControl... (GZ_PARTITION={gz_part})", flush=True)
        if not self._world_control.ping():
            raise RuntimeError(
                f"[{cls}] Cannot reach WorldControl at "
                f"/world/{self._world_name}/control\n"
                f"  GZ_PARTITION={gz_part}\n"
                f"  • Is Gazebo running?\n"
                f"  • GZ_PARTITION must match on BOTH sides:\n"
                f"      export GZ_PARTITION=0   (set before launching Gazebo AND this script)\n"
                f"  • Verify service exists: gz service --list | grep control"
            )
        print(f"[{cls}] WorldControl OK", flush=True)

    # ------------------------------------------------------------------ Gymnasium interface

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._stop_counting()
        self._world_control.reset()
        self._current_step = 0
        self._current_episode += 1
        observation = self.set_default_observation()
        return observation, self.get_info()

    def step(self, action):
        self.apply_action(action)
        self._advance_physics()
        self._current_step += 1
        observation = self.get_observation()
        reward = self.get_reward(action)
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        return observation, reward, terminated, truncated, self.get_info()

    # ------------------------------------------------------------------ abstract

    @abstractmethod
    def apply_action(self, action):
        """Publish action to the simulation (torques, positions, etc.)."""

    @abstractmethod
    def get_observation(self):
        """Return the current observation. Called after _advance_physics() — no wait needed."""

    @abstractmethod
    def get_reward(self, action) -> float:
        """Compute reward for the current state."""

    @abstractmethod
    def is_terminated(self) -> bool:
        """Return True if the episode has reached a terminal state."""

    @abstractmethod
    def is_truncated(self) -> bool:
        """Return True if the episode should end due to a time limit."""

    @abstractmethod
    def set_default_observation(self):
        """Reset internal state and return the initial observation for a new episode."""

    def get_info(self) -> dict:
        return {}
