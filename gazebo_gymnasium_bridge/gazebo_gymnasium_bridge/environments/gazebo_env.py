import logging
import time
from typing import Optional

import numpy as np
import gymnasium as gym

from ..backend.nodes import world_control

logger = logging.getLogger("gz_gymnasium.env")


class GazeboEnv(gym.Env):
    """
    Standalone Gymnasium environment that controls a running Gazebo
    simulation externally via gz.transport services and topics.

    Unlike the plugin-based GazeboBaseEnv, this class runs as a separate
    process and drives the simulation manually via WorldControl service
    calls. This makes it naturally compatible with StableBaselines3 and
    any library that expects a standard Gymnasium environment.

    Usage:
        1. Launch Gazebo with your world (starts paused by default)
        2. Subclass GazeboEnv and implement the abstract methods
        3. Use it like any Gymnasium environment:
            obs, info = env.reset()
            obs, reward, terminated, truncated, info = env.step(action)

    Subclasses must implement:
        apply_action(action)       - Execute action in simulation
        get_observation()          - Read current state from topics
        get_reward(action)         - Compute step reward
        is_terminated()            - Check terminal conditions
        is_truncated()             - Check truncation conditions
        set_default_observation()  - Return initial observation on reset
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, world_name, obs_space, act_space, steps_per_action=10):
        super().__init__()

        self.world_name = world_name
        self.observation_space = obs_space
        self.action_space = act_space

        # Simulation control — steps the sim externally via service calls
        self.world_control = world_control.WorldController(world_name, steps_per_action)

        # State tracking
        self.observation = None
        self.current_step = 0
        self.current_episode = 0
        self.episode_reward = 0
        self.episode_reward_list = []
        self.max_steps_per_episode = 500

        # Performance tracking
        self._step_times = []
        self._episode_start_time = None

        logger.info(f"GazeboEnv initialized: world={world_name}, "
                     f"steps_per_action={steps_per_action}")

    def step(self, action):
        """
        Standard Gymnasium step: apply action, advance simulation, return results.

        Args:
            action: An item from the action space.
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        t0 = time.perf_counter()

        self.apply_action(action)
        t_action = time.perf_counter()

        self.world_control.step()
        t_sim = time.perf_counter()

        self.observation = self.get_observation()
        reward = self.get_reward(action)
        self.episode_reward += reward
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        info = self.get_info()

        self.current_step += 1
        t_total = time.perf_counter()

        step_ms = (t_total - t0) * 1000
        self._step_times.append(step_ms)

        # Log every 100 steps with breakdown
        if self.current_step % 100 == 0:
            avg_ms = np.mean(self._step_times[-100:])
            sim_ms = (t_sim - t_action) * 1000
            logger.info(
                f"Episode {self.current_episode} | Step {self.current_step} | "
                f"avg step: {avg_ms:.1f}ms | "
                f"this step: action={((t_action - t0) * 1000):.1f}ms, "
                f"sim={sim_ms:.1f}ms, obs={((t_total - t_sim) * 1000):.1f}ms | "
                f"rate: {(1000 / avg_ms):.0f} steps/s"
            )

        return np.array(self.observation, dtype=np.float32), float(reward), terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Standard Gymnasium reset: reset the Gazebo world and return initial observation.

        Args:
            seed: Random seed (optional).
            options: Reset options (optional).
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed, options=options)
        t0 = time.perf_counter()

        # Log episode summary before resetting
        if self._episode_start_time is not None and self.current_step > 0:
            episode_secs = time.perf_counter() - self._episode_start_time
            avg_step_ms = np.mean(self._step_times[-self.current_step:]) if self._step_times else 0
            logger.info(
                f"Episode {self.current_episode} done | "
                f"steps: {self.current_step} | "
                f"reward: {self.episode_reward:.1f} | "
                f"time: {episode_secs:.2f}s | "
                f"avg step: {avg_step_ms:.1f}ms ({(1000 / avg_step_ms if avg_step_ms > 0 else 0):.0f} steps/s)"
            )

        # Reset the Gazebo world, then pause so we retain manual stepping control.
        # These are separate service calls — combining them in one WorldControl
        # message can cause Gazebo to ignore the pause during the reset operation.
        self.world_control.reset(pause_after=True)

        # Brief settle time for Gazebo to process the reset and for topic
        # callbacks to receive the updated (zeroed) state.
        time.sleep(0.05)

        # Track episode stats
        if self.current_episode > 0:
            self.episode_reward_list.append(self.episode_reward)

        self.episode_reward = 0
        self.current_step = 0
        self.current_episode += 1
        self._episode_start_time = time.perf_counter()

        reset_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"Reset complete ({reset_ms:.0f}ms) — starting episode {self.current_episode}")

        obs = np.array(self.set_default_observation(), dtype=np.float32)
        return obs, {}

    def render(self):
        """Gazebo handles rendering — nothing to do here."""
        pass

    def close(self):
        """Clean shutdown."""
        pass

    # --- Abstract methods for the user to implement ---

    def apply_action(self, action):
        """Execute the given action in the Gazebo simulation (e.g., publish to a topic)."""
        raise NotImplementedError

    def get_observation(self):
        """Read and return the current observation from Gazebo topic callbacks."""
        raise NotImplementedError

    def get_reward(self, action):
        """Compute and return the reward for the current step."""
        raise NotImplementedError

    def is_terminated(self) -> bool:
        """Return True if the episode has reached a terminal state (success or failure)."""
        raise NotImplementedError

    def is_truncated(self) -> bool:
        """Return True if the episode should end due to a time/step limit."""
        raise NotImplementedError

    def get_info(self):
        """Return any additional info (default: empty dict)."""
        return {}

    def set_default_observation(self):
        """Return the initial observation used after a reset."""
        raise NotImplementedError
