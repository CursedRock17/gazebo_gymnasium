# HalfCheetah

<!-- TODO: expand this intro paragraph. -->

## Overview

<!-- TODO -->

## Objective

<!-- TODO -->

## Observation Space

`Box(17,)`, `float32`, unbounded. MuJoCo's own layout: positions first
(root's forward slide excluded), then all velocities.

| Index | Name | Type | Min | Max |
|---|---|---|---|---|
| 0 | `root_up` position (torso height above spawn) | `float32` | −inf | inf |
| 1 | `root_pitch` position | `float32` | −inf | inf |
| 2 | `bthigh` position | `float32` | −inf | inf |
| 3 | `bshin` position | `float32` | −inf | inf |
| 4 | `bfoot` position | `float32` | −inf | inf |
| 5 | `fthigh` position | `float32` | −inf | inf |
| 6 | `fshin` position | `float32` | −inf | inf |
| 7 | `ffoot` position | `float32` | −inf | inf |
| 8 | `root_fwd` velocity (forward speed) | `float32` | −inf | inf |
| 9 | `root_up` velocity | `float32` | −inf | inf |
| 10 | `root_pitch` velocity | `float32` | −inf | inf |
| 11 | `bthigh` velocity | `float32` | −inf | inf |
| 12 | `bshin` velocity | `float32` | −inf | inf |
| 13 | `bfoot` velocity | `float32` | −inf | inf |
| 14 | `fthigh` velocity | `float32` | −inf | inf |
| 15 | `fshin` velocity | `float32` | −inf | inf |
| 16 | `ffoot` velocity | `float32` | −inf | inf |

## Action Space

`Box(6,)`, `float32`, normalized to `[-1, 1]` per joint. Gears differ per
joint, matching MuJoCo's own `half_cheetah.xml`.

| # | Action | Min | Max | Joint | Name | Type |
|---|---|---|---|---|---|---|
| 0 | Back thigh torque | −120 N·m | 120 N·m | revolute | `bthigh` | `float32` |
| 1 | Back shin torque | −90 N·m | 90 N·m | revolute | `bshin` | `float32` |
| 2 | Back foot torque | −60 N·m | 60 N·m | revolute | `bfoot` | `float32` |
| 3 | Front thigh torque | −120 N·m | 120 N·m | revolute | `fthigh` | `float32` |
| 4 | Front shin torque | −60 N·m | 60 N·m | revolute | `fshin` | `float32` |
| 5 | Front foot torque | −30 N·m | 30 N·m | revolute | `ffoot` | `float32` |

## Installation

No per-example install step exists beyond the repository-wide setup in
[the root README](../../README.md#installation): `pixi install` resolves
ROS 2 Jazzy, Gazebo Harmonic, and the RL libraries into one environment,
and `pixi run build` builds the colcon workspace. Every registered
`AgentSpec`, this one included, is available the moment that build
finishes.

## Training

No launch file exists yet for this environment (only `cartpole` and
`line_follower` ship one today), so training runs exclusively through the
in-process backend, with no GUI to watch mid-training.

```bash
pixi run train --agent half_cheetah --n_agents 16 --timesteps 600000
```

<!-- TODO: once/if a harness launch file exists for this environment,
     document the launched + GUI workflow here, matching cartpole.md. -->

## Evaluation

```bash
pixi run deploy --agent half_cheetah --n_agents 4
```

## CLI Arguments

`deploy.py`'s Hugging Face Hub and agent-count flags:

| Argument | Default | Effect |
|---|---|---|
| `--agent` | `cartpole` | Registered spec name: pass `half_cheetah` |
| `--n_agents` | `4` | Number of agents evaluated in parallel |
| `--model` | `models/<agent>_multi/final_<algo>_n<N>.zip` | Local checkpoint path |
| `--from-hub` | `None` | Hugging Face Hub `repo_id` to download and run instead of `--model` |
| `--hub-filename` | `model.zip` | Filename within the Hub repo (only with `--from-hub`) |
| `--episodes` | `3` | Evaluation episodes |
| `--backend` | `inprocess` | `inprocess`, `harness`, or `peragent` |

## Deploying Published Models

An unsolved but real checkpoint (roughly 3.01 m/s sustained, median 3.25,
at 600,000 steps) is published to
[`CursedRock17/gazebo-gymnasium-policies/half_cheetah`](https://huggingface.co/CursedRock17/gazebo-gymnasium-policies),
the same result documented under What Good Results Look Like below.
Running it needs no separate download step:

```bash
pixi run deploy --agent half_cheetah --n_agents 4 \
    --from-hub CursedRock17/gazebo-gymnasium-policies \
    --hub-filename half_cheetah/model.zip
```

Or pull it directly with the Hub CLI, or `huggingface_sb3` in Python:

```bash
hf download CursedRock17/gazebo-gymnasium-policies half_cheetah/model.zip --local-dir models/
```

```python
from huggingface_sb3 import load_from_hub

checkpoint = load_from_hub(
    repo_id="CursedRock17/gazebo-gymnasium-policies",
    filename="half_cheetah/model.zip",
)
```

## Screenshot

<!-- TODO: no spectator-camera capture exists for this environment yet
     (unlike InvertedDoublePendulum, Hopper, and Walker2d; see
     README.md's "What Solved Policies Look Like"). Add one under
     docs/images/half_cheetah_gui.png once a policy is worth capturing. -->

## Reward Space

`forward_progress_reward(vel_index=8, alive_bonus=0.0, ctrl_cost=0.1)`:
forward velocity minus 0.1 times the summed squared action. No alive
bonus, matching MuJoCo's own HalfCheetah reward.

```
reward = root_fwd_velocity - 0.1 * sum(action**2)
```

**Termination.** No health check, matching MuJoCo's own HalfCheetah, which has none
either. A pure numerical guard ends the episode if any observation value
reaches or exceeds 1000 in magnitude, so a solver blow-up can't silently
poison training; otherwise episodes end only on truncation at
`max_episode_steps=1000`. Spawn height 0.77 m, `frame_skip=5`.

## Glossary

| Term | Definition |
|---|---|
| **Entity Component Manager (ECM)** | Gazebo's runtime interface for reading and writing simulated joint and link state |
| **Proximal Policy Optimization (PPO)** | The default reinforcement learning algorithm used across the framework's specifications |
| **AgentSpec** | The dataclass describing one agent's model, observation, action, reward, and DR configuration |
| **Gear** | MuJoCo's per-joint torque scale; the per-joint force limits above mirror it directly |

## Difficulty

<!-- TODO: rate easy / medium / hard relative to the other environments. -->

## Domain Randomization

Not implemented for this environment: `half_cheetah`'s `AgentSpec` sets
no domain-randomization fields. See
[`domain_randomization.md`](../domain_randomization.md) for the mechanism
`line_follower` uses, if this environment ever needs sim-to-real transfer.

## Training Arguments

`train.py`'s flags, shared across every registered agent.

| Argument | Default | Effect |
|---|---|---|
| `--agent` | `cartpole` | Registered spec name: pass `half_cheetah` |
| `--n_agents` | `4` | Agents in one sim (1 = single-agent) |
| `--algo` | `ppo` | `ppo`, `a2c`, `sac`, `td3`, or `ddpg`; this environment's `Box(6,)` action works with all five |
| `--timesteps` | `200000` | Total environment steps |
| `--backend` | `inprocess` | `inprocess`, `harness`, or `peragent` |
| `--tensorboard` | off | Log to `models/<agent>_multi/tb/` |
| `--wandb` | off | Also mirror the run to Weights & Biases (implies `--tensorboard`) |
| `--push-to-hub` | `None` | Upload the trained model + an auto-generated model card to this Hugging Face Hub `repo_id` |
| `--hub-eval-episodes` | `5` | Deterministic eval episodes used to fill in the model card (only with `--push-to-hub`) |

## What Good Results Look Like

Open-ended rather than a fixed solved bar: reporting sustained meters per
second (MuJoCo-solved is roughly 5 to 6 m/s for reference). Best verified
result: **roughly 3.01 m/s sustained (median 3.25) at 600,000 steps**, up
from 1.46 m/s at 300,000, full detail in
[README.md's Solved Bars Explained](README.md#solved-bars-explained).

<!-- TODO: add a per-environment tensorboard/reward-curve image once one
     exists for this environment specifically. -->

## Trying It With Different Libraries

```python
# 1. Stable-Baselines3, through its own VecEnv API.
import stable_baselines3 as sb3
from gazebo_gymnasium_bridge.envs import make_inprocess

vec = make_inprocess("half_cheetah", n_agents=16)
sb3.PPO("MlpPolicy", vec, n_steps=64).learn(total_timesteps=1_000_000)

# 2. Any Gymnasium tool (RLlib, CleanRL, Tianshou, TorchRL), a standard env.
import gazebo_gymnasium_bridge
import gymnasium as gym
env = gym.make("GazeboHalfCheetah-v0")

# 3. The native gymnasium vector env.
vec = gym.make_vec("GazeboHalfCheetah-v0", num_envs=16,
                   vectorization_mode="vector_entry_point")
```

[`docs/rl_libraries.md`](../rl_libraries.md) covers skrl, rl_games, and
rsl_rl too, each through its own `training_scripts/train_<library>.py
--agent half_cheetah`, inside the `rl-libs` pixi environment.
`half_cheetah` is one of the agents that page names explicitly as tested
with all three.

## Version History

<!-- TODO -->
