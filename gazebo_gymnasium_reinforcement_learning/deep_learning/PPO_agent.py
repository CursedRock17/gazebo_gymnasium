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

from collections import namedtuple

import numpy as np
from torch import clamp
from torch import float as torch_float
from torch import from_numpy
from torch import load
from torch import long as torch_long
from torch import manual_seed
from torch import min as torch_min
from torch import no_grad
from torch import save
from torch import tanh as tanh
from torch import tensor
from torch.distributions import Categorical
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import SubsetRandomSampler

# `done` defaults to False so existing call sites (acrobot_learner, diff_drive_learner)
# that pass only 5 positional fields keep working. Set `done=True` on the transition
# that ends an episode so train_step() can stop bootstrapping returns across it.
Transition = namedtuple("Transition",
                        ["state", "action", "a_log_prob", "reward", "next_state", "done"])
Transition.__new__.__defaults__ = (False,)


class PPOAgent:
    """Implement the PPO RL algorithm (https://arxiv.org/abs/1707.06347).

    Works with a set of discrete actions. Uses the Actor and Critic neural
    network classes defined below.
    """

    def __init__(self, number_of_inputs, number_of_actor_outputs,
                 clip_param=0.2, max_grad_norm=0.5, ppo_update_iters=5,
                 batch_size=8, gamma=0.99, actor_lr=0.001, critic_lr=0.003,
                 ent_coef=0.001, seed=None):
        super().__init__()
        if seed is not None:
            manual_seed(seed)

        # Hyper-parameters
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.ppo_update_iters = ppo_update_iters
        self.batch_size = batch_size
        self.gamma = gamma
        self.ent_coef = ent_coef  # entropy bonus coefficient — keeps the policy stochastic

        # Models
        self.actor_net = Actor(number_of_inputs, number_of_actor_outputs)
        self.critic_net = Critic(number_of_inputs)

        # Create the optimizers
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), actor_lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), critic_lr)

        # Training stats
        self.buffer = []

    def work(self, agent_input, type_="simple"):
        """type_ == "simple" Implementation for a simple forward pass for all box values.

        type == "discrete"     Implementation for a simple forward pass for probability between 2
        values type_ == "maxiumum"     Implementation for the forward pass, that returns the max
        selected value.
        """
        agent_input = from_numpy(np.array(agent_input)).float(
        ).unsqueeze(0)  # Add batch dimension with unsqueeze
        scaling_array = from_numpy(np.array([0.5]))

        # Implement the feedforward network
        action_prob = self.actor_net(agent_input)

        if type_ == "simple":
            # Normalized distribution allows us a range on both objects
            n = Normal(action_prob, scaling_array)
            action = n.sample()
            # Force range of [-1, 1]
            rescaled_action = tanh(action)
            return [rescaled_action[0][0].item(), rescaled_action[0][1].item()]
        elif type_ == "discrete":
            # Normalized distribution allows us to grab a sample
            n = Categorical(action_prob)
            action = n.sample()
            return [action.item()]
        elif type_ == "maximum":
            return np.argmax(action_prob).item(), 1.0
        else:
            raise Exception("Wrong type in agent.work(), returning input")

    def get_value(self, state):
        """Get the value of the current state according to the critic model.

        :param state: The current state
        :return: state's value
        """
        state = from_numpy(state)
        with no_grad():
            value = self.critic_net(state)
        return value.item()

    def save(self, path):
        """Save actor and critic models in the path provided.

        :param path: path to save the models
        :type path: str
        """
        save(self.actor_net.state_dict(), path + "_actor.pkl")
        save(self.critic_net.state_dict(), path + "_critic.pkl")

    def load(self, path):
        """Load actor and critic models from the path provided.

        :param path: path where the models are saved
        :type path: str
        """
        actor_state_dict = load(path + "_actor.pkl")
        critic_state_dict = load(path + "_critic.pkl")
        self.actor_net.load_state_dict(actor_state_dict)
        self.critic_net.load_state_dict(critic_state_dict)

    def store_transition(self, transition):
        """Store a transition in the buffer to be used later.

        :param transition: contains state, action, action_prob, reward, next_state
        :type transition: namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward',
            'next_state'])
        """
        self.buffer.append(transition)

    def train_step(self, batch_size=None):
        """Perform one PPO update on the actor and critic from buffered transitions.

        Resets the buffer after the update. If `batch_size` is supplied it
        overrides `self.batch_size` for this call only.

        :param: batch_size: int
        :return: None
        """
        # Default behaviour waits for buffer to collect at least one batch_size of transitions
        if batch_size is None:
            if len(self.buffer) < self.batch_size:
                return
            batch_size = self.batch_size

        # Surface that an update is happening so callers can confirm PPO is
        # actually training (and not silently no-op'ing when the buffer is small).
        buffer_size = len(self.buffer)
        ep_boundaries = sum(1 for t in self.buffer if t.done)

        # Extract states, actions, rewards and action probabilities from transitions in buffer
        state = tensor([t.state for t in self.buffer], dtype=torch_float)
        action = tensor([t.action for t in self.buffer], dtype=torch_long).view(-1, 1)
        old_action_log_prob = tensor(
            [t.a_log_prob for t in self.buffer], dtype=torch_float).view(-1, 1)

        # If the buffer ends mid-episode, bootstrap the return rollout with the
        # critic's estimate of the next-state value. Without this, the trailing
        # transitions get pessimistic returns (R=0) that bias the advantage
        # signal — even though the agent didn't actually fail there.
        if not self.buffer[-1].done:
            last_next_state = tensor(self.buffer[-1].next_state, dtype=torch_float).unsqueeze(0)
            with no_grad():
                R = self.critic_net(last_next_state).item()
        else:
            R = 0

        # Unroll rewards into discounted returns, zeroing the bootstrap factor at
        # episode boundaries so returns don't leak across them.
        Gt = []
        for t in reversed(self.buffer):
            R = t.reward + self.gamma * R * (1.0 - float(t.done))
            Gt.insert(0, R)
        Gt = tensor(Gt, dtype=torch_float)

        # Repeat the update procedure for ppo_update_iters
        for _ in range(self.ppo_update_iters):
            # Create randomly ordered batches of size batch_size from buffer
            for index in BatchSampler(
                    SubsetRandomSampler(range(len(self.buffer))),
                    batch_size, False):
                # Calculate the advantage at each step
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # Per-batch advantage normalization keeps the surrogate scale stable.
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                # Get the current probabilities (full distribution for entropy, then
                # gather the taken action).
                full_action_probs = self.actor_net(state[index])
                action_prob = full_action_probs.gather(1, action[index])  # new policy

                # PPO
                # Ratio between current and old policy probabilities
                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # Entropy bonus pushes the policy to stay stochastic; subtracted
                # from action_loss so minimizing action_loss maximizes entropy.
                entropy = Categorical(probs=full_action_probs).entropy().mean()
                action_loss = -torch_min(surr1, surr2).mean() - self.ent_coef * entropy

                # update actor network
                self.actor_optimizer.zero_grad()  # Delete old gradients
                action_loss.backward()  # Perform backward step to compute new gradients
                nn.utils.clip_grad_norm_(self.actor_net.parameters(),
                                         self.max_grad_norm)  # Clip gradients
                self.actor_optimizer.step()  # Perform training step based on gradients

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

        # After each training step, the buffer is cleared
        del self.buffer[:]
        print(f"[PPOTrain] buffer={buffer_size} episodes_in_buffer={ep_boundaries} "
              f"actor_loss={float(action_loss):.4f} value_loss={float(value_loss):.4f} "
              f"entropy={float(entropy):.3f}")


class Actor(nn.Module):

    def __init__(self, number_of_inputs, number_of_outputs):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(number_of_inputs, 10)
        self.fc2 = nn.Linear(10, 10)
        self.action_head = nn.Linear(10, number_of_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):

    def __init__(self, number_of_inputs):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(number_of_inputs, 10)
        self.fc2 = nn.Linear(10, 10)
        self.state_value = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.state_value(x)
        return value
