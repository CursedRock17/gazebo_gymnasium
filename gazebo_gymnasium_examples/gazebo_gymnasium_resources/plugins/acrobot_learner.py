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

from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node
import numpy as np

import gazebo_gymnasium_bridge.plugins.gazebo_single_agent as gazebo_single_agent
from gazebo_gymnasium_reinforcement_learning.deep_learning.PPO_agent import PPOAgent


# Acrobot Class Resembling the OpenAI Gymnasium Space:
# https://gymnasium.farama.org/environments/classic_control/acrobot/
class Acrobot(gazebo_single_agent.GazeboSingleAgent):

    def __init__(self):
        # Define the acrobot's observation space; what we are monitoring.

        # Setup Nodes necessary for establishing information about acrobot
        self.agent_name = "/acrobot"

        #  Check the position of our cart and angle of our pole
        self.pose_node = Node()
        pose_topic = "/world" + self.agent_name + "/pose/info"
        self.pose_sub = self.pose_node.subscribe(Pose_V, pose_topic, self.update_position)

        # Setup information about our training steps
        self.max_steps_per_episode = 500  # Max number of steps per episode
        self.episode_reward = 0  # Score accumulated during an episode
        # List of all episode scores; used to check whether the task is solved
        self.episode_reward_list = []
        self.reward_goal = 500

        # Get Access to our Training Agent
        self.ppo_agent = PPOAgent(
            number_of_inputs=self.observation_space.shape[0], number_of_actor_outputs=2)

        super().__init__(self.observation_space, self.action_space)

    # --- Abstract methods for user to implement ---
    def select_action(self):
        selected_action = self.ppo_agent.work(self.observation, type_="discrete")

        return selected_action

    def apply_action(self, action):
        pass

    def get_observation(self):
        pass

    def get_reward(self, action):
        pass

    def is_terminated(self) -> bool:
        pass

    def is_truncated(self) -> bool:
        pass

    def get_info(self):
        pass

    def set_default_observation(self):
        pass

    # Extra Functions for Our System
    def solved(self):
        # After a certain number of trials (100), check to see if the agent is
        # typically achieving goal
        if len(self.episode_reward_list) > 100:
            if np.mean(self.episode_reward_list[-100:]) > self.reward_goal:
                return True
        return False

    def update_position(self, pose_v_msg):
        # Callback for our Pose Topic To Grab Info of Cart and Pole
        for pose in pose_v_msg.pose:
            pass


def get_system():
    """Expose the system so Gazebo's PythonSystemLoader can find it."""
    return Acrobot()
