# Ant

<!-- TODO: expand this intro paragraph. Ant is the first free-base spec
     in this codebase: unlike Hopper/Walker2d/HalfCheetah's planar root
     (modeled as explicit slide/hinge joints), Ant's torso has no parent
     joint at all: a genuine 6-DOF free body, read through
     `AgentSpec.base_obs` / `HarnessCore`'s Link API. See
     [creating_your_own_agent.md](../creating_your_own_agent.md) before
     porting another free-base robot (Humanoid, HumanoidStandup). -->

## Overview

<!-- TODO -->

## Objective

<!-- TODO -->

## Observation Space

`Box(27,)`, `float32`, unbounded. MuJoCo's own canonical layout: base
height, base orientation quaternion, then the 8 joint angles; then base
linear and angular velocity, then the 8 joint velocities.

| Index | Name | Type | Min | Max |
|---|---|---|---|---|
| 0 | Base height (`z`) | `float32` | −inf | inf |
| 1-4 | Base orientation quaternion (`qw, qx, qy, qz`) | `float32` | −inf | inf |
| 5 | `hip_1` position | `float32` | −inf | inf |
| 6 | `ankle_1` position | `float32` | −inf | inf |
| 7 | `hip_2` position | `float32` | −inf | inf |
| 8 | `ankle_2` position | `float32` | −inf | inf |
| 9 | `hip_3` position | `float32` | −inf | inf |
| 10 | `ankle_3` position | `float32` | −inf | inf |
| 11 | `hip_4` position | `float32` | −inf | inf |
| 12 | `ankle_4` position | `float32` | −inf | inf |
| 13 | Base forward velocity | `float32` | −inf | inf |
| 14 | Base side velocity | `float32` | −inf | inf |
| 15 | Base vertical velocity | `float32` | −inf | inf |
| 16-18 | Base angular velocity (roll, pitch, yaw rates) | `float32` | −inf | inf |
| 19 | `hip_1` velocity | `float32` | −inf | inf |
| 20 | `ankle_1` velocity | `float32` | −inf | inf |
| 21 | `hip_2` velocity | `float32` | −inf | inf |
| 22 | `ankle_2` velocity | `float32` | −inf | inf |
| 23 | `hip_3` velocity | `float32` | −inf | inf |
| 24 | `ankle_3` velocity | `float32` | −inf | inf |
| 25 | `hip_4` velocity | `float32` | −inf | inf |
| 26 | `ankle_4` velocity | `float32` | −inf | inf |

## Action Space

`Box(8,)`, `float32`, normalized to `[-1, 1]` per joint, scaled to ±150
N·m (MuJoCo's gear-150 convention).

| # | Action | Min | Max | Joint | Name | Type |
|---|---|---|---|---|---|---|
| 0 | Leg 1 hip torque | −150 N·m | 150 N·m | revolute | `hip_1` | `float32` |
| 1 | Leg 1 ankle torque | −150 N·m | 150 N·m | revolute | `ankle_1` | `float32` |
| 2 | Leg 2 hip torque | −150 N·m | 150 N·m | revolute | `hip_2` | `float32` |
| 3 | Leg 2 ankle torque | −150 N·m | 150 N·m | revolute | `ankle_2` | `float32` |
| 4 | Leg 3 hip torque | −150 N·m | 150 N·m | revolute | `hip_3` | `float32` |
| 5 | Leg 3 ankle torque | −150 N·m | 150 N·m | revolute | `ankle_3` | `float32` |
| 6 | Leg 4 hip torque | −150 N·m | 150 N·m | revolute | `hip_4` | `float32` |
| 7 | Leg 4 ankle torque | −150 N·m | 150 N·m | revolute | `ankle_4` | `float32` |

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
pixi run train --agent ant --n_agents 16 --timesteps 300000
```

<!-- TODO: once/if a harness launch file exists for this environment,
     document the launched + GUI workflow here, matching cartpole.md. -->

## Evaluation

```bash
pixi run deploy --agent ant --n_agents 4
```

## CLI Arguments

`deploy.py`'s Hugging Face Hub and agent-count flags:

| Argument | Default | Effect |
|---|---|---|
| `--agent` | `cartpole` | Registered spec name: pass `ant` |
| `--n_agents` | `4` | Number of agents evaluated in parallel |
| `--model` | `models/<agent>_multi/final_<algo>_n<N>.zip` | Local checkpoint path |
| `--from-hub` | `None` | Hugging Face Hub `repo_id` to download and run instead of `--model` |
| `--hub-filename` | `model.zip` | Filename within the Hub repo (only with `--from-hub`) |
| `--episodes` | `3` | Evaluation episodes |
| `--backend` | `inprocess` | `inprocess`, `harness`, or `peragent` |

## Deploying Published Models

<!-- TODO: no model for this environment is published to
     https://huggingface.co/CursedRock17/gazebo-gymnasium-policies yet.
     Once one is, document it the way line_follower.md does under
     "Using the Published Policy": hf_hub_download() snippet, then
     `pixi run deploy --agent ant --from-hub
     CursedRock17/gazebo-gymnasium-policies --hub-filename
     ant/model.zip`. -->

## Screenshot

<!-- TODO: no spectator-camera capture exists for this environment yet
     (unlike InvertedDoublePendulum, Hopper, and Walker2d; see
     README.md's "What Solved Policies Look Like"). Add one under
     docs/images/ant_gui.png once a policy is worth capturing. -->

## Reward Space

`forward_progress_reward(vel_index=13, ctrl_cost=0.5)`: forward velocity,
plus a 1.0 alive bonus, minus 0.5 times the summed squared action.

```
reward = base_fwd_velocity + 1.0 - 0.5 * sum(action**2)
```

**Termination.** Unhealthy, and the episode ends, when base height falls
outside `(0.2, 1.0)` (Gymnasium `Ant-v4`'s own `healthy_z_range` default)
or any observation value reaches or exceeds 100 in magnitude. Otherwise
runs to `max_episode_steps=1000`. Spawn height 0.75 m.

## Glossary

| Term | Definition |
|---|---|
| **Entity Component Manager (ECM)** | Gazebo's runtime interface for reading and writing simulated joint and link state |
| **Proximal Policy Optimization (PPO)** | The default reinforcement learning algorithm used across the framework's specifications |
| **AgentSpec** | The dataclass describing one agent's model, observation, action, reward, and DR configuration |
| **Free base** | A model whose root link carries no parent joint: a genuine 6-DOF free body, unlike the planar locomotors' explicit slide/hinge root |
| **Alive bonus** | A fixed per-step reward term for not having terminated; the reward-shaping term behind Ant's stand-still local optimum |

## Difficulty

<!-- TODO: rate easy / medium / hard. Note: README.md currently documents
     Ant as "the one genuine open problem": both PPO and SAC converge to
     a stand-still local optimum rather than walking; see "What Good
     Results Look Like" below before rating this. -->

## Domain Randomization

Not implemented for this environment: `ant`'s `AgentSpec` sets no
domain-randomization fields. See
[`domain_randomization.md`](../domain_randomization.md) for the mechanism
`line_follower` uses, if this environment ever needs sim-to-real transfer.

## Training Arguments

`train.py`'s flags, shared across every registered agent.

| Argument | Default | Effect |
|---|---|---|
| `--agent` | `cartpole` | Registered spec name: pass `ant` |
| `--n_agents` | `4` | Agents in one sim (1 = single-agent) |
| `--algo` | `ppo` | `ppo`, `a2c`, `sac`, `td3`, or `ddpg`; this environment's `Box(8,)` action works with all five |
| `--timesteps` | `200000` | Total environment steps |
| `--backend` | `inprocess` | `inprocess`, `harness`, or `peragent` |
| `--tensorboard` | off | Log to `models/<agent>_multi/tb/` |
| `--wandb` | off | Also mirror the run to Weights & Biases (implies `--tensorboard`) |
| `--push-to-hub` | `None` | Upload the trained model + an auto-generated model card to this Hugging Face Hub `repo_id` |
| `--hub-eval-episodes` | `5` | Deterministic eval episodes used to fill in the model card (only with `--push-to-hub`) |

## What Good Results Look Like

Not solved. Open-ended metric: sustained forward meters per second.
Current result: **0.0 m/s, confirmed under both PPO at 150,000 steps and a
real SAC sweep at 300,000 steps** (960.3 reward, but 0.00 m/s sustained,
mean absolute action 0.06, alive the whole episode). Both algorithms
converge to the same stand-still-and-collect-the-alive-bonus solution, not
a walking gait, pointing at the `alive_bonus=1.0` reward term rather than
algorithm choice. Full detail in
[README.md's Solved Bars Explained](README.md#solved-bars-explained) and
[ROADMAP.md](../../ROADMAP.md).

<!-- TODO: add a per-environment tensorboard/reward-curve image once a
     reward-shaping fix is tried. -->

## Trying It With Different Libraries

Ant has no registered Gymnasium id yet (`GazeboAnt-v0` doesn't exist in
`gym_env.py`'s registry, unlike every other built-in environment), so
reach it through `make_inprocess`/`make_multi`/`make_harness` directly
rather than `gym.make`.

```python
# 1. Stable-Baselines3, through its own VecEnv API.
import stable_baselines3 as sb3
from gazebo_gymnasium_bridge.envs import make_inprocess

vec = make_inprocess("ant", n_agents=16)
sb3.PPO("MlpPolicy", vec, n_steps=64).learn(total_timesteps=1_000_000)
```

[`docs/rl_libraries.md`](../rl_libraries.md) covers skrl, rl_games, and
rsl_rl too, each through its own `training_scripts/train_<library>.py
--agent ant`, inside the `rl-libs` pixi environment. `ant` is one of the
agents that page names explicitly as tested with all three; those scripts
take `--agent` directly and don't go through the Gymnasium registry.

## Version History

<!-- TODO -->
