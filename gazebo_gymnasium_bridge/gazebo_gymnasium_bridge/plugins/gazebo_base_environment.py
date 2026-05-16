# Standard Modules
import random
from typing import Optional

import numpy as np

# Gymnasium Modules
import gymnasium as gym

# Gazebo Modules
from gz.sim8 import Model, Link, World, world_entity

# Extra Nodes
from ..backend.nodes import world_control


class GazeboBaseEnv(gym.Env):
    """
    Main class to integrate Gymnasium into a Gazebo simulation.
    Broken into 2 sections, the first is all of the Gazebo rendering logic, so that
    the gymnasium information pairs well with the agent. The second is made up
    of various abstract methods that the user creates. This class
    should never be called on it's own, only abstracted from.
    """
    # Gazebo in a "human" renderer that achieves 30 & 500 fps depending on the camera used.
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, obs_space, act_space):
        """
        Allows user to easily initialize Gymnasium environment in tandem with Gazebo Configure.
        """
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

        # Reward information about the space
        self.episode_reward = 0
        self.episode_reward_list = []

        # Information about the Sim
        self.needs_reset = False

        # Set up timing information about the simulation
        self.max_steps_per_episode = 0
        self.max_training_episodes = 0
        self.current_step = 0
        self.current_episode = 0

    def configure(self, entity, element, ecm, eventManager):
        """
        Necessary function for Gazebo plugin on startup, called once sim time has begun.
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
        """
        Necessary function for Gazebo plugin before the starting of each scene.
        Modifies state before physics run, important for motor control.
        Args:
            info: Information about the current scene.
            ecm: Entity Component Manager, keeps track of individual tags within
            the SDF file, thus the world.
        """
        if info.paused or self.needs_reset:
            return

        # Run our action to update the space
        self.action = self.select_action()
        self.apply_action(self.action)

    def update(self, info, ecm):
        """
        Necessary function for Gazebo plugin to run during each scene.
        Used for physics simulation step.
        Args:
            info: Information about the current scene.
            ecm: Entity Component Manager, keeps track of individual tags within
            the SDF file, thus the world.
        """
        if info.paused or self.needs_reset:
            return

    def post_update(self, info, ecm):
        """
        Necessary function for Gazebo plugin to run after each scene.
        Used for post physics processing step: reading results for sensors.
        Args:
            info: Information about the current scene.
            ecm: Entity Component Manager, keeps track of individual tags within
            the SDF file, thus the world.
        """
        if info.paused:
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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Unified reset() supporting both:
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
        """
        Necessary function for Gazebo plugin to run plugin reset.
        Args:
            info: Information about the current scene.
            ecm: Entity Component Manager, keeps track of individual tags within
            the SDF file, thus the world.
        """
        # Reset the Gymnasium Environment
        self.needs_reset = False
        self.observation, self.info = self.gymnasium_reset(seed=None, options=None)

        # Reset Timing Variables
        self.current_step = 0
        self.current_episode += 1

        # Update reward scheme
        self.episode_reward_list.append(self.episode_reward)
        self.episode_reward = 0

    def step(self, action):
        """
        Steps the environment forward by one action.
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
        """
        Called at the end of each episode, brings Gazebo environment back
        to base standards.
        Args:
            seed: Sets up randomization of actions (optional).
            options: Provides a Dict that changes the way the environment resets(optional).
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
        """
        Defined method for Gymnasium to display an environment. Not necessary to
        implement since Gazebo takes care of the graphics. Necessary for excess
        Gymnasium plugins.
        """
        pass

    def close(self):
        """
        Defined method for Gymnasium to close an environment. Not necessary to
        implement since Gazebo takes care of the graphics. Necessary for excess
        Gymnasium plugins.
        """
        pass

    # --- Abstract methods for user to implement ---
    def select_action(self):
        """
        User Implemented Function.
        Called to setup an action at the start of the frame, often
        taken from some sort of deep neural network and based on
        the observation space. That being said the user can implement
        any technique they'd like, so long as it returns an action
        for apply action.
        """
        raise NotImplementedError

    def apply_action(self, action):
        """
        User Implemented Function.
        Called to apply whatever actions are contained in the user defined
        action_space. Called by the step() function in order to update the
        current scene and provide new data to run through the environment.
        """
        raise NotImplementedError

    def get_observation(self):
        """
        User Implemented Function.
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
        """
        User Implemented Function.
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
        """
        User Implemented Function.
        The function that establishes if any conditions have
        been met that would end the current episode.
        Returns:
            finished: bool: The state of the episode.
        """
        raise NotImplementedError

    def is_truncated(self) -> bool:
        """
        User Implemented Function.
        Handles ending the episode when time is not involved
        with the observation space.
        Returns:
            finished: bool: The state of the episode.
        """
        raise NotImplementedError

    def get_info(self):
        """
        User Implemented Function.
        Called after all actions have been processed which provides
        additional information the user may need that is not relevant
        to the actions, observations, or rewards.
        Returns:
            info: Object with the various additional information
            necessary for the agent to preform well.
        """
        raise NotImplementedError

    def set_default_observation(self):
        """
        User Implemented Function.
        Called when the user wants to entirely reset the observation
        space, especially useful in reset steps.
        """
        raise NotImplementedError


def get_system():
    """Allows Gazebo Python Path to find this file"""
    return GazeboBaseEnv()
