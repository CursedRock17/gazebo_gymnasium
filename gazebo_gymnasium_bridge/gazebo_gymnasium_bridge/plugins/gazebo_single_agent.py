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

from typing import Optional

from . import gazebo_base_environment
# Extra Nodes
from ..backend.nodes import handle_single_agent


class GazeboSingleAgent(gazebo_base_environment.GazeboBaseEnv):

    def __init__(self, obs_space, act_space):
        """Initialize a single-agent Gymnasium environment alongside Gazebo configure."""
        super().__init__(obs_space, act_space)

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
        # Call Higher Level Class Configuration
        super().configure(entity, element, ecm, eventManager)

        # Configure setup for the agent we're looking to monitor
        self.agent_name = element.get_string("agent_name")
        if (element.get_string("spawn_agent")):
            self.agent_node = handle_single_agent.HandleSingleAgent(
               self.world_name, self.agent_name, "sdf_file")

    def pre_update(self, info, ecm):
        """Necessary function for Gazebo plugin before the starting of each scene.

        Modifies state before physics run, important for motor control.
        Args:
            info: Information about the current scene.
            ecm: Entity Component Manager, keeps track of individual tags within
            the SDF file, thus the world.
        """
        # Call Higher Level Class Pre Update
        super().pre_update(info, ecm)

    def update(self, info, ecm):
        """Necessary function for Gazebo plugin to run during each scene.

        Used for physics simulation step.
        Args:
            info: Information about the current scene.
            ecm: Entity Component Manager, keeps track of individual tags within
            the SDF file, thus the world.
        """
        # Call Higher Level Class Update
        super().update(info, ecm)

    def post_update(self, info, ecm):
        """Necessary function for Gazebo plugin to run after each scene.

        Used for post physics processing step: reading results for sensors.
        Args:
            info: Information about the current scene.
            ecm: Entity Component Manager, keeps track of individual tags within
            the SDF file, thus the world.
        """
        # Call Higher Level Class Post Update
        super().post_update(info, ecm)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment (Gymnasium-style entry point).

        Args:
            seed: Optional RNG seed forwarded to the base class.
            options: Optional dict of reset options forwarded to the base class.
        """
        # Call Higher Level Class Reset
        return super().reset(seed, options)

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
    return GazeboSingleAgent()
