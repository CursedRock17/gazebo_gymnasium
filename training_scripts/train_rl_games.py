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
r"""PPO on any registered continuous-action agent via rl_games -- a small custom IVecEnv adapter.

rl_games' *default* registration path (`vecenv_type: GYMNASIUM`) assumes it
owns parallelism: it spawns N *separate* single-agent env instances and
parallelizes them itself, which would launch N separate Gazebo worlds and
defeat this repo's whole N-agents-in-one-world model. The fix below is a
custom `IVecEnv` registered through `vecenv.register()` -- the same
mechanism Isaac Lab's own `RlGamesVecEnvWrapper` uses -- presenting the
whole `make_inprocess` batch as ONE vec env rl_games can drive.

Box actions only (this script's network config is `continuous_a2c_logstd`)
-- Discrete (plain cartpole) and image observations (line_follower) would
need a different model/network config, not built here.

Needs the `rl-libs` pixi environment (rl_games is NOT in the default env):

    pixi run -e rl-libs bash -c \
        "source install/setup.sh && python training_scripts/train_rl_games.py --agent hopper"
"""

import argparse

import gym as legacy_gym  # noqa: I100 -- see _to_legacy_space below
from gymnasium import spaces
import numpy as np
from rl_games.common import env_configurations
from rl_games.common import vecenv
from rl_games.common.ivecenv import IVecEnv
from rl_games.torch_runner import Runner

from gazebo_gymnasium_bridge.envs import get_spec
from gazebo_gymnasium_bridge.envs import make_inprocess


def _to_legacy_space(space):
    """Convert a `gymnasium.spaces` object to its legacy `gym.spaces` equivalent.

    rl_games' ExperienceBuffer does `type(x) is gym.spaces.Box` (the
    legacy `gym` package specifically, not `gymnasium`) to decide how to
    allocate its rollout tensors -- an exact-type check, so a structurally
    identical `gymnasium.spaces.Box` silently fails it and NOTHING gets
    allocated (`self.tensor_dict['obses']` stays None, actions too via
    `is_continuous` never getting set). Convert at the one boundary that
    matters instead of fighting rl_games' internals.
    """
    import gymnasium as gym

    if isinstance(space, gym.spaces.Box):
        return legacy_gym.spaces.Box(low=space.low, high=space.high, dtype=space.dtype)
    if isinstance(space, gym.spaces.Discrete):
        return legacy_gym.spaces.Discrete(int(space.n))
    raise TypeError(f"no legacy gym.spaces conversion for {type(space)}")


class GazeboRlGamesVecEnv(IVecEnv):
    """Wraps ONE gazebo_gymnasium VecEnv (N agents, one world) as ONE rl_games vec env.

    Not N separately-spawned single-agent envs. rl_games' own "num_agents per actor" concept
    assumes every agent WITHIN one actor shares a single episode boundary -- its reward-completion
    tracking literally does `dones.view(num_actors, num_agents).all(dim=1)`,
    only counting an episode once ALL agents in that group are done on the
    SAME step. gazebo_gymnasium's agents reset independently (real bug hit
    while wiring this up: with num_actors=1/num_agents=16, that `.all()`
    across 16 independently-resetting cartpoles essentially never fires --
    a full 250k-step run logged zero completed episodes, "rew-inf" in every
    checkpoint name). The fix: treat each of our agents as its own actor
    (num_actors=n_agents, num_agents=1 per actor) so the done-check is
    per-agent, matching how they actually reset.
    """

    def __init__(self, config_name, num_actors, agent="cartpole_continuous", **kwargs):
        # num_actors IS n_agents now (see the config's own comment) -- one
        # real agent per rl_games "actor".
        self.env = make_inprocess(agent, n_agents=num_actors)
        self.num_actors = num_actors

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(np.asarray(actions))
        # rl_games always expects observations dict-wrapped under "obs"
        # (even without an asymmetric actor-critic / central value setup --
        # self.obs['obs'] is accessed unconditionally in its rollout loop),
        # and float32 rewards/infos as one dict, not a per-agent list.
        return ({"obs": obs.astype(np.float32)}, rewards.astype(np.float32), dones, {})

    def reset(self):
        return {"obs": self.env.reset().astype(np.float32)}

    def get_number_of_agents(self):
        return 1  # one gazebo_gymnasium agent per rl_games "actor" -- see above

    def get_env_info(self):
        return {
            "observation_space": _to_legacy_space(self.env.observation_space),
            "action_space": _to_legacy_space(self.env.action_space),
            "agents": 1,
        }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--agent",
        default="cartpole_continuous",
        help="registered spec with a Box action space (this "
        "script's network config is continuous_a2c_logstd)",
    )
    p.add_argument("--n_agents", type=int, default=16)
    p.add_argument("--timesteps", type=int, default=250_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    spec = get_spec(args.agent)
    if not isinstance(spec.action_space, spaces.Box):
        raise SystemExit(
            f"train_rl_games.py: {args.agent!r} has a "
            f"{type(spec.action_space).__name__} action space -- this "
            f"script's network config (continuous_a2c_logstd) is Box-only. "
            f"Pick a continuous-action agent (cartpole_continuous, hopper, "
            f"walker2d, half_cheetah, reacher, ant, ...)."
        )
    if spec.image_obs is not None:
        raise SystemExit(
            f"train_rl_games.py: {args.agent!r} has image observations -- "
            f"this script's network is a plain MLP, no CNN."
        )

    vecenv.register(
        "GAZEBO_GYMNASIUM",
        lambda config_name, num_actors, **kw: GazeboRlGamesVecEnv(config_name, num_actors, **kw),
    )
    env_configurations.register(
        "gazebo_gymnasium_env",
        {"vecenv_type": "GAZEBO_GYMNASIUM", "env_creator": lambda **kw: None},
    )

    # horizon_length x num_actors = the rollout size before each PPO update
    # -- matches this run's SB3/skrl counterparts (256 x 16 = 4096 samples).
    max_epochs = args.timesteps // (256 * args.n_agents)
    config = {
        "params": {
            "algo": {"name": "a2c_continuous"},
            "model": {"name": "continuous_a2c_logstd"},
            "network": {
                "name": "actor_critic",
                "separate": True,
                "space": {
                    "continuous": {
                        "mu_activation": "None",
                        "sigma_activation": "None",
                        "mu_init": {"name": "default", "scale": 0.02},
                        "sigma_init": {"name": "const_initializer", "val": 0},
                        "fixed_sigma": True,
                    }
                },
                "mlp": {
                    "units": [64, 64],
                    "activation": "relu",
                    "initializer": {"name": "default"},
                },
            },
            "config": {
                "name": args.agent,
                "train_dir": f"models/{args.agent}_multi/rl_games",
                "env_name": "gazebo_gymnasium_env",
                "reward_shaper": {"scale_value": 1.0},
                "normalize_advantage": True,
                "normalize_input": True,
                "gamma": 0.99,
                "tau": 0.95,
                "learning_rate": 3e-4,
                # Very high on purpose: score_to_win is an early-stop
                # threshold, and only cartpole-family envs have a
                # meaningful fixed cap (500) -- other agents' reward scales
                # don't, so this just lets max_epochs be the real budget
                # for every agent, matching skrl/rsl_rl's own behavior here
                # (neither has an early-stop-on-solved mechanism either).
                "score_to_win": 1e9,
                "grad_norm": 0.5,
                "entropy_coef": 0.0,
                "truncate_grads": True,
                "e_clip": 0.2,
                "clip_value": True,
                # num_actors = n_agents: one gazebo_gymnasium agent per
                # rl_games "actor" (see GazeboRlGamesVecEnv's own docstring
                # for why -- rl_games' episode-completion tracking needs
                # this to be per-agent, not per-batch, since our agents
                # reset independently).
                "num_actors": args.n_agents,
                "horizon_length": 256,
                "minibatch_size": 512,
                "mini_epochs": 10,
                "critic_coef": 1,
                "lr_schedule": "adaptive",
                "kl_threshold": 0.008,
                "bounds_loss_coef": 0.001,
                "max_epochs": max_epochs,
                "seed": args.seed,
                "device": "cuda",
                "env_config": {"agent": args.agent},
            },
        }
    }

    print(
        f"[train_rl_games] agent={args.agent} n_agents={args.n_agents} "
        f"timesteps={args.timesteps} (~{max_epochs} epochs) -> "
        f"{config['params']['config']['train_dir']}"
    )
    runner = Runner()
    runner.load(config)
    runner.run({"train": True})


if __name__ == "__main__":
    main()
