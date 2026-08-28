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
r"""PPO on any registered continuous-action agent via skrl -- no adapter needed.

skrl's own `wrap_env()` detects a `gymnasium.vector.VectorEnv` by walking the
class's base hierarchy for a `gymnasium.*` module -- and this repo's native
vector entry point (`gym_vector_env.make_gym_vector`) genuinely IS one
(`GazeboVectorEnv` subclasses `VectorEnv` directly). So this script is close
to skrl's own Pendulum PPO example
(`examples/gymnasium/torch_gymnasium_pendulum_ppo.py` in the skrl repo),
pointed at our env instead of theirs -- not a custom integration.

Box actions only (this project's continuous-action specs, e.g.
cartpole_continuous, hopper, walker2d, ant, ...) -- skrl's own Categorical
model would be needed for Discrete (e.g. plain cartpole), and image
observations (line_follower) would need a CNN model, neither built here.

Needs the `rl-libs` pixi environment (skrl is NOT in the default env -- it's
heavy and most users of this repo don't need it):

    pixi run -e rl-libs bash -c \
        "source install/setup.sh && python training_scripts/train_skrl.py --agent hopper"
"""

import argparse
import time

from gymnasium import spaces
from skrl.agents.torch.ppo import PPO
from skrl.agents.torch.ppo import PPO_CFG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin
from skrl.models.torch import GaussianMixin
from skrl.models.torch import Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
import torch
import torch.nn as nn

from gazebo_gymnasium_bridge.envs import get_spec
from gazebo_gymnasium_bridge.envs.gym_vector_env import make_gym_vector


class Policy(GaussianMixin, Model):
    """Gaussian policy -- assumes a Box(-1, 1) action space."""

    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        # clip_log_std matters, not just clip_actions: without bounding
        # log_std, a bad early gradient step can drift it to a huge value,
        # exp(log_std) overflows to inf, and sampling Normal(mean, inf)
        # produces NaN actions -- silently ignored by gz-sim's force setter,
        # but a real, invisible-otherwise training bug. Matches skrl's own
        # reference Pendulum example's bounds.
        GaussianMixin.__init__(
            self, clip_actions=True, clip_log_std=True, min_log_std=-20, max_log_std=2
        )
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
            nn.Tanh(),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # Tanh already bounds to [-1, 1] -- unlike skrl's own Pendulum example
        # (action range [-2, 2]), no extra scale factor needed here.
        return self.net(inputs["observations"]), {"log_std": self.log_std_parameter}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--agent",
        default="cartpole_continuous",
        help="registered spec with a Box action space (this "
        "script's Policy model is Gaussian-only)",
    )
    p.add_argument("--n_agents", type=int, default=16)
    p.add_argument("--timesteps", type=int, default=250_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    spec = get_spec(args.agent)
    if not isinstance(spec.action_space, spaces.Box):
        raise SystemExit(
            f"train_skrl.py: {args.agent!r} has a {type(spec.action_space).__name__} "
            f"action space -- this script's Policy model is Gaussian-only "
            f"(Box actions). Pick a continuous-action agent (cartpole_continuous, "
            f"hopper, walker2d, half_cheetah, reacher, ant, ...)."
        )
    if spec.image_obs is not None:
        raise SystemExit(
            f"train_skrl.py: {args.agent!r} has image observations -- this "
            f"script's Policy/Value models are plain MLPs, no CNN."
        )

    set_seed(args.seed)

    env = make_gym_vector(num_envs=args.n_agents, agent=args.agent)
    env = wrap_env(env)  # auto-detected as a Gymnasium VectorEnv, no adapter
    device = env.device

    # memory_size MUST equal cfg.rollouts -- skrl's own reference example
    # keeps both at 1024. A mismatch here (found the hard way: NaN actions
    # appearing mid-rollout, well before the first gradient update, on BOTH
    # this env and plain Pendulum-v1 -- so not a gazebo_gymnasium bug) means
    # PPO starts a rollout update before the underlying buffer is actually
    # filled, reading uninitialized (garbage/NaN) memory slots.
    ROLLOUTS = 256
    memory = RandomMemory(memory_size=ROLLOUTS, num_envs=env.num_envs, device=device)

    models = {
        "policy": Policy(env.observation_space, env.state_space, env.action_space, device),
        "value": Value(env.observation_space, env.state_space, env.action_space, device),
    }

    cfg = PPO_CFG()
    cfg.rollouts = ROLLOUTS
    cfg.learning_epochs = 10
    cfg.mini_batches = 8
    cfg.discount_factor = 0.99
    cfg.gae_lambda = 0.95
    cfg.learning_rate = 3e-4
    cfg.learning_rate_scheduler = KLAdaptiveLR
    cfg.learning_rate_scheduler_kwargs = {"kl_threshold": 0.008}
    cfg.grad_norm_clip = 0.5
    cfg.ratio_clip = 0.2
    cfg.value_clip = 0.2
    cfg.entropy_loss_scale = 0.0
    cfg.value_loss_scale = 0.5
    cfg.kl_threshold = 0
    cfg.observation_preprocessor = RunningStandardScaler
    cfg.observation_preprocessor_kwargs = {"size": env.observation_space, "device": device}
    cfg.value_preprocessor = RunningStandardScaler
    cfg.value_preprocessor_kwargs = {"size": 1, "device": device}
    cfg.experiment.write_interval = "auto"
    cfg.experiment.checkpoint_interval = "auto"
    cfg.experiment.directory = f"models/{args.agent}_multi/skrl"

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=device,
    )

    trainer = SequentialTrainer(
        cfg={"timesteps": args.timesteps, "headless": True}, env=env, agents=agent
    )

    print(
        f"[train_skrl] agent={args.agent} n_agents={args.n_agents} "
        f"timesteps={args.timesteps} -> {cfg.experiment.directory}"
    )
    t0 = time.perf_counter()
    trainer.train()
    dt = time.perf_counter() - t0
    print(f"[train_skrl] done in {dt:.1f}s ({args.timesteps / dt:.1f} steps/s)")


if __name__ == "__main__":
    main()
