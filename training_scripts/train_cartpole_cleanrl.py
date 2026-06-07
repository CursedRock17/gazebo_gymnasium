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

"""Train Gazebo CartPole using a CleanRL-style single-file PPO.

This is adapted from CleanRL's canonical `cleanrl/ppo.py` (the discrete-action
variant) trimmed to the bare essentials and pointed at our GazeboCartPoleEnv
directly — no vectorization, no gym.make() ceremony, no SyncVectorEnv. The
purpose is to demonstrate that the same env class drops into a CleanRL-style
loop just as cleanly as it does into SB3.

CleanRL design principles preserved here:
  * Single-file, no class hierarchy
  * All hyperparameters at the top
  * One-pass PPO update, NOT minibatched (kept the simplest variant)
  * Verbose per-step printouts (the "monitor what's happening" style)

Prereq:
    ros2 launch gazebo_gymnasium_bringup cartpole.launch.py

Run:
    ./venv/bin/python train_cartpole_cleanrl.py
"""

import time

import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.optim as optim

from gazebo_gymnasium_bridge.envs import GazeboCartPoleEnv

# ---------------------------------------------------------------------- #
# Hyperparameters
# ---------------------------------------------------------------------- #
TOTAL_TIMESTEPS = 200_000
NUM_STEPS = 128            # rollout length
GAMMA = 0.99
GAE_LAMBDA = 0.95
NUM_UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 32
LEARNING_RATE = 2.5e-4
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
SEED = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize a layer with orthogonal weights (CleanRL convention)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Shared-feature actor-critic for discrete actions."""

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    env = GazeboCartPoleEnv()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = Agent(obs_dim, n_actions).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # Rollout storage (CleanRL convention)
    obs_buf = torch.zeros((NUM_STEPS, obs_dim)).to(DEVICE)
    actions_buf = torch.zeros(NUM_STEPS, dtype=torch.long).to(DEVICE)
    logprobs_buf = torch.zeros(NUM_STEPS).to(DEVICE)
    rewards_buf = torch.zeros(NUM_STEPS).to(DEVICE)
    dones_buf = torch.zeros(NUM_STEPS).to(DEVICE)
    values_buf = torch.zeros(NUM_STEPS).to(DEVICE)

    obs_np, _ = env.reset(seed=SEED)
    next_obs = torch.tensor(obs_np, dtype=torch.float32).to(DEVICE)
    next_done = torch.zeros(1).to(DEVICE)

    num_updates = TOTAL_TIMESTEPS // NUM_STEPS
    global_step = 0
    start_time = time.time()

    for update in range(1, num_updates + 1):
        # --- Rollout phase ----------------------------------------------
        for step in range(NUM_STEPS):
            global_step += 1
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))
            actions_buf[step] = action.item()
            logprobs_buf[step] = logprob.item()
            values_buf[step] = value.flatten()

            obs_np, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            rewards_buf[step] = float(reward)

            if done:
                obs_np, _ = env.reset()
            next_obs = torch.tensor(obs_np, dtype=torch.float32).to(DEVICE)
            next_done = torch.tensor([float(done)]).to(DEVICE)

        # --- GAE advantage estimation -----------------------------------
        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).flatten()
            advantages = torch.zeros_like(rewards_buf).to(DEVICE)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + GAMMA * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = (
                    delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                )
            returns = advantages + values_buf

        # --- PPO update phase -------------------------------------------
        b_obs = obs_buf
        b_logprobs = logprobs_buf
        b_actions = actions_buf
        b_advantages = advantages
        b_returns = returns

        b_inds = np.arange(NUM_STEPS)
        for epoch in range(NUM_UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, NUM_STEPS, MINIBATCH_SIZE):
                mb_inds = b_inds[start:start + MINIBATCH_SIZE]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Clipped policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (unclipped — CleanRL's simplest variant)
                v_loss = 0.5 * ((newvalue.flatten() - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        sps = int(global_step / (time.time() - start_time))
        print(
            f"[CleanRL-PPO update {update}/{num_updates}] "
            f"global_step={global_step} sps={sps} "
            f"pg_loss={pg_loss.item():.4f} v_loss={v_loss.item():.4f} "
            f"entropy={entropy_loss.item():.4f}"
        )

    print("Training complete.")


if __name__ == "__main__":
    main()
