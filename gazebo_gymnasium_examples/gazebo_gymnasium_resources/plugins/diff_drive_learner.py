import gazebo_gymnasium_bridge.plugins.gazebo_single_agent as gazebo_single_agent

import gazebo_gymnasium_reinforcement_learning.utils.math_utils as utils
from gazebo_gymnasium_reinforcement_learning.deep_learning.PPO_agent import PPOAgent, Transition

from gymnasium.spaces import Box
import numpy as np

from gz.transport13 import Node, AdvertiseMessageOptions

# All Necessary Messages
from gz.msgs10.twist_pb2 import Twist
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.odometry_pb2 import Odometry


class DiffDriveLearner(gazebo_single_agent.GazeboSingleAgent):
    def __init__(self):
        # Set the Max/Min (+/-) velocities that our rover can physically move at
        self.max_linear_x = 0.5
        self.max_angular_z = 1
        self.min_linear_x = -1 * self.max_linear_x
        self.min_angular_z = -1 * self.max_angular_z

        # Set the Max/Min position (X, Y) - relative to the map
        grid_size = 10
        self.max_pos_x = grid_size
        self.max_pos_y = grid_size
        self.min_pos_x = -1 * self.max_pos_x
        self.min_pos_y = -1 * self.max_pos_y

        # Set the Max/Min angle (yaw) that our rover can be at
        self.max_yaw = 180
        self.min_yaw = -180

        # Should set a Max/Min position for the goal (randomized)
        max_goal_pos_x = grid_size
        max_goal_pos_y = grid_size
        self.min_goal_pos_x = -1 * max_goal_pos_x
        self.min_goal_pos_y = -1 * max_goal_pos_y

        # Establish rover information variables
        self.rover_linear_velocity_x = 0
        self.rover_angular_velocity_z = 0
        self.rover_orientation_yaw = 0

        # Check for obstacles
        self.has_crashed = False

        # Define the differential drive's observation space; what we are monitoring
        # Based on odometry, we can measure the Linear and Angular velocity of the rover (Twist)
        # We can also measure the position and orientation of the rover (Pose_V)
        # Our rover's Pose and Twist can live anywhere in these ranges
        self.observation_space = Box(
            low=np.array([self.min_linear_x, self.min_angular_z, self.min_pos_x, self.min_pos_y, self.min_yaw, self.min_goal_pos_x, self.min_goal_pos_y]),
            high=np.array([self.max_linear_x, self.max_angular_z, self.max_pos_x, self.max_pos_y, self.max_yaw, max_goal_pos_x, max_goal_pos_y]),
            shape=(7,),
            dtype=np.float64
        )

        # Define the differential drive's action space, represents each of the ways we 
        # can drive the rover, based on cmd_vel
        self.action_space = Box(
            low=np.array([self.min_linear_x, self.min_angular_z]),
            high=np.array([self.max_linear_x, self.max_angular_z]),
            dtype=np.float64
        )

        # Setup Nodes necessary for establishing information about rover
        self.agent_name = "diff_drive"

        # Gain the Ability to Provide a Twist to our rover
        cmd_vel_node = Node()
        cmd_vel_topic = f"/model/{self.agent_name}/cmd_vel"
        cmd_vel_opts = AdvertiseMessageOptions()
        cmd_vel_opts.msgs_per_sec = 1
        self.cmd_vel_pub = cmd_vel_node.advertise(cmd_vel_topic, Twist, cmd_vel_opts)

        # Check the velocity of our rover given as a twist
        self.tf_node = Node()
        tf_topic = f"/model/{self.agent_name}/odometry"
        self.tf_sub = self.tf_node.subscribe(Odometry, tf_topic, self.update_odometry)

        #  Check the position of our rover
        self.pose_node = Node()
        pose_topic = f"/world/{self.agent_name}/pose/info"
        self.pose_sub = self.pose_node.subscribe(Pose_V, pose_topic, self.update_position)

        # Create a reference to every piece of the rover
        wheel_distance = 0.190
        wheel_diameter = 0.05
        self.distance_between_wheels = wheel_distance
        self.wheel_diameter = wheel_diameter

        # Setup information about our training steps
        self.max_steps_per_episode = 1000  # Max number of steps per episode
        self.episode_reward = 0  # Score accumulated during an episode
        self.episode_reward_list = []  # A list to save all the episode scores, used to check if task is solved
        self.reward_goal = 10 * grid_size
        self.distance_threshold = 0.01

        # Create a default position for all of our objects
        self.rover_position = [0, 0]

        # Create a random location for our goal
        self.goal_position = [1, 1]

        # Keep track of our distance heuristics
        self.current_distance = self.get_distance()
        self.previous_distance = 0

        # Normalize Goal Position for PPO
        self.goal_position[0] = utils.normalize_to_range(
            self.goal_position[0], self.min_goal_pos_x, max_goal_pos_x, -1.0, 1.0)
        self.goal_position[1] = utils.normalize_to_range(
            self.goal_position[1], self.min_goal_pos_x, max_goal_pos_x, -1.0, 1.0)

        # Get Access to our Training Agent
        self.ppo_agent = PPOAgent(number_of_inputs=self.observation_space.shape[0], number_of_actor_outputs=2)

        super().__init__(self.observation_space, self.action_space)

    # --- Abstract methods for user to implement ---
    def select_action(self):
        action_probability = 1
        selected_action = self.ppo_agent.work(self.observation, type_="simple")
        new_observation, reward, done, truncated, info = super().get_environment()

        # Save the current state transition in agent's memory
        trans = Transition(self.observation, selected_action,
                           action_probability, reward, new_observation)
        self.ppo_agent.store_transition(trans)

        return selected_action

    def apply_action(self, action):
        # Simulate sending a Twist to the rover
        velocities = [0, 1]
        linear_velocity_x = 0
        angular_velocity_z = 0

        # Decompose the action space from it's box of values shape is (2,).
        linear_velocity_x = self.max_linear_x * action[0]
        angular_velocity_z = self.max_angular_z * action[1]

        # Publish Our Messages before the physics update
        twist_msg = Twist()
        twist_msg.linear.x = linear_velocity_x
        twist_msg.linear.y = 0
        twist_msg.linear.z = 0

        twist_msg.angular.x = 0
        twist_msg.angular.y = 0
        twist_msg.angular.z = angular_velocity_z

        self.cmd_vel_pub.publish(twist_msg)

        # Update all previous behvior variables
        self.previous_distance = self.current_distance

    def get_observation(self):
        # Essentially going to grab the odometry of the rover
        self.current_distance = self.get_distance()

        # Normalize velocities for PPO
        self.rover_linear_velocity_x = utils.normalize_to_range(
            self.rover_linear_velocity_x, self.min_linear_x, self.max_linear_x, -1.0, 1.0, clip=True)
        self.rover_angular_velocity_z = utils.normalize_to_range(
            self.rover_angular_velocity_z, self.min_angular_z, self.max_angular_z, -1.0, 1.0, clip=True)

        # Get position on X, Y of the rover ** shouldn't have any Z position
        # Normalize for PPO
        self.rover_position[0] = utils.normalize_to_range(
            self.rover_position[0], self.min_pos_x, self.max_pos_x, -1.0, 1.0, clip=True)
        self.rover_position[1] = utils.normalize_to_range(
            self.rover_position[1], self.min_pos_y, self.max_pos_y, -1.0, 1.0, clip=True)

        # Get the yaw position of the rover ** shouldn't have any pitch or roll
        self.rover_orientation_yaw = utils.normalize_to_range(
            self.rover_orientation_yaw, self.min_yaw, self.max_yaw, -1.0, 1.0, clip=True)

        return [self.rover_linear_velocity_x, self.rover_angular_velocity_z,
                self.rover_position[0], self.rover_position[1], self.rover_orientation_yaw,
                self.goal_position[0], self.goal_position[1]]

    def get_reward(self, action):
        # Need to provide positive/negative consequence based on the actions
        # of the rover to properly train it. To do this we need a goal (Does it benefit from 
        # going forward, reaching a spot, not crashing?)
        # Create variable storage so we can track multiple features all based on map size
        reward_storage = 0

        # Reward the robot for getting closer, punish if getting further away/not moving
        delta = self.previous_distance - self.current_distance
        reward_storage += delta * (self.reward_goal / 100)

        # Reward the robot immensly if it makes it to the goal
        if self.current_distance < self.distance_threshold:
            reward_storage += self.reward_goal

        # If we crash, then we have failed
        if self.has_crashed:
            reward_storage -= self.reward_goal / 2

        # Add loss at each time step, makes robot find goal in faster time
        reward_storage -= 0.01

        return reward_storage

    def is_terminated(self) -> bool:
        # Check if certain conditions are meant that have caused us to entirely
        # fail (i.e crash) or suceed (reach target)
        if self.episode_reward > self.reward_goal:
            print("Succeed")
            return True
        if self.has_crashed:
            print("Crashed")
            return True
        if self.episode_reward < -1.0:
            print("Failed")
            return True

        return False

    def is_truncated(self) -> bool:
        return False

    def get_info(self):
        i = 1

    def set_default_observation(self):
        # Returns the entire space as zeros
        return [0.0 for i in range(self.observation_space.shape[0])]

    # Extra Functions for Our Diff Drive System
    def solved(self):
        # After a certain number of trials (100), check to see if the agent is typically achieving goal
        if len(self.episode_reward_list) > 100:
            if np.mean(self.episode_reward_list[-100:]) > self.reward_goal:
                return True
        return False

    def get_distance(self):
        # Check distance between our agent and goal via heuristics
        # Just going to use pythag theorem for simplicity
        res = (((self.rover_position[0] - self.goal_position[0]) ** 2 +
                (self.rover_position[1] - self.goal_position[1]) ** 2) ** 0.5)
        return res

    def update_position(self, msg):
        # Callback for Pose topic so we can constantly monitor our rover's (X, Y) Coordinates
        for rover_position in msg.pose:
            if rover_position.name == self.agent_name:
                self.rover_position[0] = rover_position.position.x
                self.rover_position[1] = rover_position.position.y
                self.rover_orientation_yaw = rover_position.orientation.z

    def update_odometry(self, rover_velocities):
        # Callback for Twist topic so we can constantly monitor our rover's velocities
        self.rover_linear_velocity_x = rover_velocities.twist.linear.x
        self.rover_angular_velocity_z = rover_velocities.twist.angular.z


def get_system():
    """Allows Gazebo Python Path to find this file"""
    return DiffDriveLearner()
