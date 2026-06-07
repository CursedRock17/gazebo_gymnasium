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

# Standard Modules
import random
from typing import Optional

# Gymnasium Modules
import gymnasium as gym
# Gazebo Modules
from gz.sim8 import Link
from gz.sim8 import Model
from gz.sim8 import World
from gz.sim8 import world_entity

# Extra Nodes
from ..backend.nodes import world_control


class GazeboBaseEnv(gym.Env):
    """Main class to integrate Gymnasium into a Gazebo simulation.

    Broken into 2 sections, the first is all of the Gazebo rendering logic, so that the gymnasium
    information pairs well with the agent. The second is made up of various abstract methods that
    the user creates. This class should never be called on it's own, only abstracted from.
    """

    # Gazebo in a "human" renderer that achieves 30 & 500 fps depending on the camera used.
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, obs_space, act_space):
        """Initialize the Gymnasium environment alongside the Gazebo configure step."""
        super().__init__()
        self.gazebo_id = random.randint(1, 100)

        # Required variables for gym to function properly
        self.observation_space = obs_space
        self.action_space = act_space

        # Constantly Update Information in the Space
        self.info = None
        self.observation = None
        self.action = None
        self.terminated = False
        self.truncated = False

        # Reward information about the space.
        # episode_reward_list[i] and episode_steps_list[i] correspond to the same
        # completed episode, so consumers (analysis scripts, GUIs) can zip them.
        self.episode_reward = 0
        self.episode_reward_list = []
        self.episode_steps_list = []

        # Information about the Sim
        self.needs_reset = False

        # Set up timing information about the simulation
        self.max_steps_per_episode = 0
        self.max_training_episodes = 0
        self.current_step = 0
        self.current_episode = 0
        self.training_done = False

    def configure(self, entity, element, ecm, eventManager):
        """Necessary function for Gazebo plugin on startup, called once sim time has begun.

        Args:
            entity: Gazebo "entity" which is essentially whatever the plugin needs to access,
            in this case, it's accessing the world.
            element: SDF representation of the world.
            ecm: Entity Component Manager, keeps track of individual tags within
            the SDF file, thus the world.
            eventManager: Monitors all occuring actions in the world.
        """
        self.model = Model(entity)
        self.link = Link(self.model.canonical_link(ecm))
        print("Configured Plugin For: ", entity)

        # Establish World Info
        world_ent = world_entity(ecm)
        world = World(world_ent)
        self.world_name = world.name(ecm)

        # Configure max timing variables
        self.max_steps_per_episode = element.get_double("max_steps_per_episode")
        self.max_training_episodes = element.get_double("max_training_episodes")
        self.steps_per_action = element.get_double("steps_per_action")

        # Set up Important Nodes
        self.world_control = world_control.WorldController(self.world_name, self.steps_per_action)

        # Create constantly monitored state variables
        self.observation, self.info = self.gymnasium_reset(seed=None, options=None)

    def pre_update(self, info, ecm):
        """Necessary function for Gazebo plugin before the starting of each scene.

        Modifies state before physics run, important for motor control.
        Args:
            info: Information about the current scene.
            ecm: Entity Component Manager, keeps track of individual tags within
            the SDF file, thus the world.
        """
        if info.paused or self.needs_reset or self._reached_episode_cap():
            return

        # Run our action to update the space
        self.action = self.select_action()
        self.apply_action(self.action)

    def update(self, info, ecm):
        """Necessary function for Gazebo plugin to run during each scene.

        Used for physics simulation step.
        Args:
            info: Information about the current scene.
            ecm: Entity Component Manager, keeps track of individual tags within
            the SDF file, thus the world.
        """
        if info.paused or self.needs_reset:
            return

    def post_update(self, info, ecm):
        """Necessary function for Gazebo plugin to run after each scene.

        Used for post physics processing step: reading results for sensors.
        Args:
            info: Information about the current scene.
            ecm: Entity Component Manager, keeps track of individual tags within
            the SDF file, thus the world.
        """
        if info.paused or self._reached_episode_cap():
            return

        # Step to the next frame receiving information back from the current scene
        self.step(self.action)

        # Check to see if this episode is over
        if (self.current_step >= self.max_steps_per_episode or self.terminated):
            if not self.needs_reset:
                self.needs_reset = True

            if self.needs_reset:
                self.world_control.reset()
                self.needs_reset = False

    def _reached_episode_cap(self) -> bool:
        """Return True once max_training_episodes has been reached.

        Logged exactly once so the terminal isn't flooded.
        """
        if self.max_training_episodes <= 0:
            return False
        if self.current_episode < int(self.max_training_episodes):
            return False
        if not self.training_done:
            self.training_done = True
            print(f"[GazeboBaseEnv] Training complete: reached "
                  f"max_training_episodes={int(self.max_training_episodes)}")
        return True

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment from either Gymnasium or the Gazebo system plugin.

        Supports two call signatures:
        - Gymnasium: reset(seed=None, options=None)
        - Gazebo system plugin: reset(info, ecm)
        """
        # --- CASE 1: Called by Gazebo Plugin ---
        if seed is not None and not isinstance(seed, int):
            info, ecm = seed, options
            return self.gazebo_reset(info, ecm)

        # --- CASE 2: Called by Gymnasium ---
        else:
            return self.gymnasium_reset(seed=seed, options=options)

    def gazebo_reset(self, info, ecm):
        """Necessary function for Gazebo plugin to run plugin reset.

        Args:
            info: Information about the current scene.
            ecm: Entity Component Manager, keeps track of individual tags within
            the SDF file, thus the world.
        """
        # Reset the Gymnasium Environment
        self.needs_reset = False
        self.observation, self.info = self.gymnasium_reset(seed=None, options=None)

        # Record stats for the episode that just ended, then advance counters.
        # Skip the synthetic "reset before any episode ran" case (current_step==0)
        # so the log doesn't get a bogus row at sim startup.
        if self.current_step > 0:
            self.episode_reward_list.append(self.episode_reward)
            self.episode_steps_list.append(self.current_step)
            print(f"[EpisodeSummary] ep={self.current_episode} "
                  f"steps={self.current_step} reward={self.episode_reward}")

        # Reset Timing Variables
        self.current_step = 0
        self.current_episode += 1
        self.episode_reward = 0

    def step(self, action):
        """Step the environment forward by one action.

        Args:
            action: an item from the user's action space.
        Returns:
            tuple: (obs, reward, terminated, truncated, info) for the step result.
        """
        # Grab all the current information from the current scene.
        self.observation = self.get_observation()
        reward = self.get_reward(action)
        self.episode_reward += reward
        self.terminated = self.is_terminated()
        self.truncated = self.is_truncated()
        self.info = self.get_info()

        # Update the current timing values
        self.current_step += 1

        return self.observation, self.episode_reward, self.terminated, self.truncated, self.info

    def gymnasium_reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Bring the Gazebo environment back to base state at episode end.

        Args:
            seed: Sets up randomization of actions (optional).
            options: Provides a Dict that changes the way the environment resets (optional).
        Returns:
            observations: Default observations for the space.
            info: Default excess information for the space.
        """
        super().reset(seed=seed, options=options)

        self.terminated = False
        self.truncated = False
        self.needs_reset = False

        obs = self.set_default_observation()
        info = {}
        return obs, info

    def render(self):
        """Render the environment (no-op; Gazebo handles graphics).

        Kept as a method because some Gymnasium plugins call it explicitly.
        """
        pass

    def close(self):
        """Close the environment (no-op; Gazebo owns the simulator lifecycle).

        Kept as a method because some Gymnasium plugins call it explicitly.
        """
        pass

    # --- Abstract methods for user to implement ---
    def select_action(self):
        """User Implemented Function.

        Called to setup an action at the start of the frame, often taken from some sort of deep
        neural network and based on the observation space. That being said the user can implement
        any technique they'd like, so long as it returns an action for apply action.
        """
        raise NotImplementedError

    def apply_action(self, action):
        """User Implemented Function.

        Called to apply whatever actions are contained in the user defined action_space. Called by
        the step() function in order to update the current scene and provide new data to run
        through the environment.
        """
        raise NotImplementedError

    def get_observation(self):
        """User Implemented Function.

        Called after all actions have been processed, returning
        any and all data utilized by the agent, from the scene.
        Data is often received in the form of Gazebo topics, so
        it can be easily accessed. Called in step().
        Returns:
            observations: Object with the various observations defined
            in the observation_space.
        """
        raise NotImplementedError

    def get_reward(self, action):
        """User Implemented Function.

        The reward function that provides context for the agent
        to learn (punishes vs rewards). Called based up the
        action that occurs.
        Args:
            action: The thing the agent will execute
        Returns:
            reward: A positive or negative value based upon said action.
        """
        raise NotImplementedError

    def is_terminated(self) -> bool:
        """User Implemented Function.

        The function that establishes if any conditions have
        been met that would end the current episode.
        Returns:
            finished: bool: The state of the episode.
        """
        raise NotImplementedError

    def is_truncated(self) -> bool:
        """User Implemented Function.

        Handles ending the episode when time is not involved
        with the observation space.
        Returns:
            finished: bool: The state of the episode.
        """
        raise NotImplementedError

    def get_info(self):
        """User Implemented Function.

        Called after all actions have been processed which provides
        additional information the user may need that is not relevant
        to the actions, observations, or rewards.
        Returns:
            info: Object with the various additional information
            necessary for the agent to preform well.
        """
        raise NotImplementedError

    def set_default_observation(self):
        """User Implemented Function.

        Called when the user wants to entirely reset the observation space, especially useful in
        reset steps.
        """
        raise NotImplementedError


def get_system():
    """Expose the system so the Gazebo Python loader can find it."""
    return GazeboBaseEnv()
