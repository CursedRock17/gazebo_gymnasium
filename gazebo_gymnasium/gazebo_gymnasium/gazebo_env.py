from abc import abstractmethod
from typing import Optional

import gymnasium as gym

from .world_control import WorldController


class GazeboEnv(gym.Env):
    """
    Base class for Gymnasium environments backed by a running Gazebo simulation.

    Drives the simulation externally via gz.transport WorldControl service calls.
    Subclass this and implement the abstract methods for your specific robot/task.
    Compatible with any RL library that supports the Gymnasium interface (SB3, RLlib, etc.).

    Usage:
        1. Launch Gazebo with your world (world should start paused)
        2. Subclass GazeboEnv and implement all abstract methods
        3. Use it like any Gymnasium environment:

            obs, info = env.reset()
            obs, reward, terminated, truncated, info = env.step(action)
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
        self._world_control = WorldController(world_name, steps_per_action)
        self._current_step = 0
        self._current_episode = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._world_control.reset()
        self._current_step = 0
        self._current_episode += 1
        observation = self.set_default_observation()
        return observation, self.get_info()

    def step(self, action):
        self.apply_action(action)
        self._world_control.step()
        self._current_step += 1
        observation = self.get_observation()
        reward = self.get_reward(action)
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        return observation, reward, terminated, truncated, self.get_info()

    @abstractmethod
    def apply_action(self, action):
        """Apply the given action to the simulation."""

    @abstractmethod
    def get_observation(self):
        """Read and return the current observation from the simulation."""

    @abstractmethod
    def get_reward(self, action) -> float:
        """Compute and return the reward for the current state."""

    @abstractmethod
    def is_terminated(self) -> bool:
        """Return True if the episode has reached a terminal state (e.g. fell over)."""

    @abstractmethod
    def is_truncated(self) -> bool:
        """Return True if the episode should end due to a time limit."""

    @abstractmethod
    def set_default_observation(self):
        """Reset internal state and return the initial observation for a new episode."""

    def get_info(self) -> dict:
        """Return auxiliary info dict. Override to add debug/logging data."""
        return {}
