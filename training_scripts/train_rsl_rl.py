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
r"""PPO on any registered continuous-action agent via rsl_rl -- the real, non-thin adapter.

Unlike skrl (needs nothing) or rl_games (needs a small IVecEnv), rsl_rl's
`VecEnv` interface is genuinely different: PyTorch tensors grouped into a
`TensorDict` by role (not a plain Gymnasium space), device-resident,
`episode_length_buf` bookkeeping this project doesn't track in that form,
and a `time_outs`-vs-`terminated` split in `extras` gazebo_gymnasium's own
VecEnv contract doesn't carry (its `dones` conflates both, same as SB3's).

Box actions only (rsl_rl's own ActorCritic is Gaussian-only, no Discrete/
Categorical head exists upstream) -- and image observations (line_follower)
would need a CNN model, not built here.

Needs the `rl-libs` pixi environment (rsl_rl is NOT in the default env):

    pixi run -e rl-libs bash -c \
        "source install/setup.sh && python training_scripts/train_rsl_rl.py --agent hopper"
"""

import argparse

from gymnasium import spaces
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict
import torch

from gazebo_gymnasium_bridge.envs import get_spec
from gazebo_gymnasium_bridge.envs import make_inprocess


class GazeboRslRlVecEnv(VecEnv):
    """Wraps ONE gazebo_gymnasium VecEnv (N agents, one world) as an rsl_rl VecEnv.

    Box actions only -- rsl_rl's ActorCritic is Gaussian-only, no Discrete/
    Categorical head exists upstream (see docs/rl_libraries.md).
    """

    def __init__(self, agent, n_agents, device):
        self.env = make_inprocess(agent, n_agents=n_agents)
        self.num_envs = n_agents
        self.num_actions = int(self.env.action_space.shape[0])
        self.max_episode_length = self.env._spec.max_episode_steps
        self.episode_length_buf = torch.zeros(n_agents, dtype=torch.long, device=device)
        self.device = device
        self.cfg = {}
        self._last_dones = torch.zeros(n_agents, dtype=torch.bool, device=device)

    def _to_td(self, obs_np):
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        return TensorDict({"policy": obs_t}, batch_size=[self.num_envs])

    def get_observations(self) -> TensorDict:
        return self._to_td(self.env.reset())

    def step(self, actions: torch.Tensor):
        obs, rewards, dones, infos = self.env.step(actions.detach().cpu().numpy())
        dones_t = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.where(
            dones_t, torch.zeros_like(self.episode_length_buf), self.episode_length_buf + 1
        )
        # gazebo_gymnasium's `dones` conflates termination and truncation
        # (same as SB3's contract) -- infos[i]["TimeLimit.truncated"] is the
        # one per-agent signal that distinguishes them, matching how this
        # project's own SB3 VecEnvs already report it.
        time_outs = torch.tensor(
            [bool(infos[i].get("TimeLimit.truncated", False)) for i in range(self.num_envs)],
            dtype=torch.bool,
            device=self.device,
        )
        extras = {"time_outs": time_outs, "log": {}}
        self._last_dones = dones_t
        return (
            self._to_td(obs),
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            dones_t,
            extras,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--agent",
        default="cartpole_continuous",
        help="registered spec with a Box action space (rsl_rl's ActorCritic is Gaussian-only)",
    )
    p.add_argument("--n_agents", type=int, default=16)
    p.add_argument("--timesteps", type=int, default=250_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    spec = get_spec(args.agent)
    if not isinstance(spec.action_space, spaces.Box):
        raise SystemExit(
            f"train_rsl_rl.py: {args.agent!r} has a "
            f"{type(spec.action_space).__name__} action space -- rsl_rl's "
            f"ActorCritic is Gaussian-only (Box actions). Pick a "
            f"continuous-action agent (cartpole_continuous, hopper, "
            f"walker2d, half_cheetah, reacher, ant, ...)."
        )
    if spec.image_obs is not None:
        raise SystemExit(
            f"train_rsl_rl.py: {args.agent!r} has image observations -- "
            f"this script's actor/critic are plain MLPs, no CNN."
        )

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = GazeboRslRlVecEnv(args.agent, args.n_agents, device)

    num_steps_per_env = 256
    train_cfg = {
        "algorithm": {
            "class_name": PPO,
            "num_learning_epochs": 10,
            "num_mini_batches": 8,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.0,
            "learning_rate": 3e-4,
            "max_grad_norm": 0.5,
            "schedule": "adaptive",
            "desired_kl": 0.008,
        },
        # Without an explicit distribution_cfg, MLPModel builds NO stochastic
        # head at all (self.distribution stays None) -- found the hard way:
        # PPO.act() crashes on `self.distribution.log_prob(...)` the first
        # time it's called, since forward() has nothing to sample from.
        "actor": {
            "class_name": MLPModel,
            "hidden_dims": (64, 64),
            "activation": "relu",
            "distribution_cfg": {"class_name": GaussianDistribution, "init_std": 1.0},
        },
        "critic": {"class_name": MLPModel, "hidden_dims": (64, 64), "activation": "relu"},
        "obs_groups": {},  # resolves to {"actor": ["policy"], "critic": ["policy"]}
        "num_steps_per_env": num_steps_per_env,
        "save_interval": 50,
    }

    log_dir = f"models/{args.agent}_multi/rsl_rl"
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=device)

    num_iterations = args.timesteps // (num_steps_per_env * args.n_agents)
    print(
        f"[train_rsl_rl] agent={args.agent} n_agents={args.n_agents} "
        f"timesteps={args.timesteps} (~{num_iterations} iterations) -> {log_dir}"
    )
    runner.learn(num_learning_iterations=num_iterations)


if __name__ == "__main__":
    main()
