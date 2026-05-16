import gazebo_gymnasium_bridge.plugins.gazebo_single_agent as gazebo_single_agent

from gazebo_gymnasium_reinforcement_learning.deep_learning.PPO_agent import PPOAgent

from gymnasium.spaces import Box, Discrete
import numpy as np

from gz.transport13 import Node, AdvertiseMessageOptions

# All Necessary Messages
from gz.msgs10.model_pb2 import Model
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.double_pb2 import Double


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
        self.episode_reward_list = []  # A list to save all the episode scores, used to check if task is solved
        self.reward_goal = 500

        # Get Access to our Training Agent
        self.ppo_agent = PPOAgent(number_of_inputs=self.observation_space.shape[0], number_of_actor_outputs=2)

        super().__init__(self.observation_space, self.action_space)

    # --- Abstract methods for user to implement ---
    def select_action(self):
        selected_action = self.ppo_agent.work(self.observation, type_="discrete")

        return selected_action

    def apply_action(self, action):

    def get_observation(self):

    def get_reward(self, action):

    def is_terminated(self) -> bool:

    def is_truncated(self) -> bool:

    def get_info(self):
        pass

    def set_default_observation(self):

    # Extra Functions for Our System
    def solved(self):
        # After a certain number of trials (100), check to see if the agent is typically achieving goal
        if len(self.episode_reward_list) > 100:
            if np.mean(self.episode_reward_list[-100:]) > self.reward_goal:
                return True
        return False

    def update_position(self, pose_v_msg):
        # Callback for our Pose Topic To Grab Info of Cart and Pole
        for pose in pose_v_msg.pose:


def get_system():
    """Allows Gazebo Python Path to find this file"""
    return Acrobot()
