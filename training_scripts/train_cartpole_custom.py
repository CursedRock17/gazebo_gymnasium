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

"""Train CartPole in Gazebo using the project's custom PPOAgent.

Same env as train_cartpole_sb3.py — only the trainer differs. The env doesn't
know or care which agent backend is driving it.

Prereq:
    ros2 launch gazebo_gymnasium_bringup cartpole.launch.py
"""

import numpy as np
from torch import from_numpy
from torch.distributions import Categorical

from gazebo_gymnasium_bridge.envs import GazeboCartPoleEnv
from gazebo_gymnasium_reinforcement_learning.deep_learning.PPO_agent import PPOAgent
from gazebo_gymnasium_reinforcement_learning.deep_learning.PPO_agent import Transition


def train(env: GazeboCartPoleEnv, total_timesteps: int = 500_000):
    agent = PPOAgent(
        number_of_inputs=env.observation_space.shape[0],
        number_of_actor_outputs=2,
        batch_size=256,
        ppo_update_iters=10,
        actor_lr=3e-4,
        critic_lr=3e-4,
        ent_coef=0.001,
    )

    obs, _info = env.reset()
    timesteps = 0
    while timesteps < total_timesteps:
        # Inline forward pass so we can capture the action's probability for
        # PPO's Transition record.
        obs_tensor = from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        probs = agent.actor_net(obs_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        action_prob = probs[0, action.item()].item()

        next_obs, reward, terminated, truncated, _info = env.step(action.item())
        done = bool(terminated or truncated)

        agent.store_transition(Transition(
            list(map(float, obs)),
            int(action.item()),
            action_prob,
            float(reward),
            list(map(float, next_obs)),
            done,
        ))
        agent.train_step()

        timesteps += 1
        if done:
            obs, _info = env.reset()
        else:
            obs = next_obs


if __name__ == "__main__":
    env = GazeboCartPoleEnv(world_name="cartpole")
    train(env)
