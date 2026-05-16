import gazebo_gymnasium_bridge.plugins.gazebo_single_agent as gazebo_single_agent

from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env

from gymnasium.spaces import Box, Discrete
import numpy as np

from gz.transport13 import Node, AdvertiseMessageOptions

# All Necessary Messages
from gz.msgs10.model_pb2 import Model
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.double_pb2 import Double


# Cart Pole Class Resembling the OpenAI Gymnasium Space:
# https://gymnasium.farama.org/environments/classic_control/cart_pole/
# But using Stable Baselines A2C
class CartPole(gazebo_single_agent.GazeboSingleAgent):
    def __init__(self):
        # Define the cart pole's observation space; what we are monitoring.
        """
        Observation Space (Box):
            Cart Position         : [-4.8, 4.8]
            Cart Velocity         : [-Inf, Inf]
            Pole Angle            : [-24deg, 24deg]
            Pole Angular Velocity : [-Inf, Inf]
        """
        # Max/Min Position + Velocity of the Cart Itself.
        self.max_position = 4.8
        self.min_position = -4.8
        self.max_velocity = float('inf')
        self.min_velocity = float('-inf')

        # Max/Min Angle + Angular Velocity of the Pole.
        self.max_angle = 0.41887903
        self.min_angle = -0.41887903
        self.max_ang_velocity = float('inf')
        self.min_ang_velocity = float('-inf')

        self.observation_space = Box(
            low=np.array([self.min_position, self.min_velocity, self.min_angle, self.min_ang_velocity]),
            high=np.array([self.max_position, self.max_velocity, self.max_angle, self.max_ang_velocity]),
            shape=(4,),
            dtype=np.float32
        )

        # Define the cart pole's action space.
        """
        Action Space (Discrete):
            0 : Push Cart Left
            1 : Push Cart Right
        """
        self.action_space = Discrete(2)

        # Setup Nodes necessary for establishing information about rover
        self.agent_name = "/cartpole"

        # Gain the ability to apply a velocity to our cart
        self.cmd_pos_node = Node()
        cmd_pos_topic = "/model" + self.agent_name + "/joint/slider_to_cart/0/cmd_pos"
        cmd_pos_opts = AdvertiseMessageOptions()
        cmd_pos_opts.msgs_per_sec = 1
        self.cmd_pos_pub = self.cmd_pos_node.advertise(cmd_pos_topic, Double, cmd_pos_opts)

        #  Check the position of our cart and angle of our pole
        self.pose_node = Node()
        pose_topic = "/world" + self.agent_name + "/pose/info"
        self.pose_sub = self.pose_node.subscribe(Pose_V, pose_topic, self.update_position)

        #  Check the velocity of our cart and velocity of our pole
        self.velocity_node = Node()
        velocity_topic = "/world" + self.agent_name + "/model" + self.agent_name + "/joint_state"
        self.velocity_sub = self.velocity_node.subscribe(
            Model, velocity_topic, self.update_velocity)

        # Setup all of our cart information
        self.cart_position = 0
        self.cart_velocity = 0
        self.pole_angle = 0
        self.pole_ang_velocity = 0

        # Setup information about our training steps
        self.max_steps_per_episode = 1000  # Max number of steps per episode
        self.episode_reward = 0  # Score accumulated during an episode
        self.episode_reward_list = []  # A list to save all the episode scores, used to check if task is solved
        self.reward_goal = 500

        super().__init__(self.observation_space, self.action_space)

    # --- Abstract methods for user to implement ---
    def select_action(self):
        return self.action_space.sample()

    def apply_action(self, action):
        real_action = action[0]
        # Break Down Discrete(2) Space
        if real_action == 0:
            # Move Left
            doubleMsg = Double()
            doubleMsg.data = -2.0
            self.cmd_pos_pub.publish(doubleMsg)
        elif real_action == 1:
            # Move Right
            doubleMsg = Double()
            doubleMsg.data = 2.0
            self.cmd_pos_pub.publish(doubleMsg)

    def get_observation(self):
        return [self.cart_position, self.cart_velocity,
                self.pole_angle, self.pole_ang_velocity]

    def get_reward(self, action):
        # Keeping the Pole up as long as possible is the goal, so reward the program
        # for continuing to do so.
        return 1.0

    def is_terminated(self) -> bool:
        # Check if certain conditions are met that have caused us to entirely
        # fail (leave range) or succeed (keep pole up long enough)
        if self.episode_reward >= self.reward_goal:
            # We have survived 500 Steps
            return True
        if abs(self.pole_angle) > 0.20944:
            # Pole Angle Is Greater than +/- 12 degrees
            print("Failed after ", self.current_step, " steps")
            return True
        if abs(self.cart_position) > 2.4:
            # Cart Position (x) Is Greater than +/- 2.4 meters
            print("Failed after ", self.current_step, " steps")
            return True

        # Otherwise we keep going
        return False

    def is_truncated(self) -> bool:
        return False

    def get_info(self):
        pass

    def set_default_observation(self):
        # Returns the entire space as zeros
        return [0.0 for i in range(self.observation_space.shape[0])]

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
            if (pose.name == "cart"):
                self.cart_position = pose.position.x
            elif (pose.name == "pole"):
                self.pole_angle = pose.orientation.x

    def update_velocity(self, model_msg):
        # Callback for our Pose Topic To Grab Info of Cart and Pole
        for joint in model_msg.joint:
            if (joint.name == "slider_to_cart"):
                self.cart_velocity = joint.axis1.velocity
            elif (joint.name == "cart_to_pole"):
                self.pole_ang_velocity = joint.axis1.velocity


def get_system():
    """Allows Gazebo Python Path to find this file"""
    return CartPole()
