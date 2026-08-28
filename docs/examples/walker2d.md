# Walker2d

<!-- TODO: expand this intro paragraph. Walker2d reuses the Hopper
     recipe with two legs; see
     [Porting Hopper's "What Transferred" section](porting_hopper.md#what-transferred). -->

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
| 2 | `thigh_joint` position | `float32` | −inf | inf |
| 3 | `leg_joint` position | `float32` | −inf | inf |
| 4 | `foot_joint` position | `float32` | −inf | inf |
| 5 | `thigh_left_joint` position | `float32` | −inf | inf |
| 6 | `leg_left_joint` position | `float32` | −inf | inf |
| 7 | `foot_left_joint` position | `float32` | −inf | inf |
| 8 | `root_fwd` velocity (forward speed) | `float32` | −inf | inf |
| 9 | `root_up` velocity | `float32` | −inf | inf |
| 10 | `root_pitch` velocity | `float32` | −inf | inf |
| 11 | `thigh_joint` velocity | `float32` | −inf | inf |
| 12 | `leg_joint` velocity | `float32` | −inf | inf |
| 13 | `foot_joint` velocity | `float32` | −inf | inf |
| 14 | `thigh_left_joint` velocity | `float32` | −inf | inf |
| 15 | `leg_left_joint` velocity | `float32` | −inf | inf |
| 16 | `foot_left_joint` velocity | `float32` | −inf | inf |

## Action Space

`Box(6,)`, `float32`, normalized to `[-1, 1]` per joint, scaled to ±100 N·m.

| # | Action | Min | Max | Joint | Name | Type |
|---|---|---|---|---|---|---|
| 0 | Right thigh torque | −100 N·m | 100 N·m | revolute | `thigh_joint` | `float32` |
| 1 | Right leg torque | −100 N·m | 100 N·m | revolute | `leg_joint` | `float32` |
| 2 | Right foot torque | −100 N·m | 100 N·m | revolute | `foot_joint` | `float32` |
| 3 | Left thigh torque | −100 N·m | 100 N·m | revolute | `thigh_left_joint` | `float32` |
| 4 | Left leg torque | −100 N·m | 100 N·m | revolute | `leg_left_joint` | `float32` |
| 5 | Left foot torque | −100 N·m | 100 N·m | revolute | `foot_left_joint` | `float32` |

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
pixi run train --agent walker2d --n_agents 16 --timesteps 600000
```

600,000 steps is the budget that reached the solved bar, resumed from a
300,000-step checkpoint; see
[README.md](README.md#solved-bars-explained) for the full path there.

<!-- TODO: once/if a harness launch file exists for this environment,
     document the launched + GUI workflow here, matching cartpole.md. -->

## Evaluation

```bash
pixi run deploy --agent walker2d --n_agents 4
```

## CLI Arguments

`deploy.py`'s Hugging Face Hub and agent-count flags:

| Argument | Default | Effect |
|---|---|---|
| `--agent` | `cartpole` | Registered spec name: pass `walker2d` |
| `--n_agents` | `4` | Number of agents evaluated in parallel |
| `--model` | `models/<agent>_multi/final_<algo>_n<N>.zip` | Local checkpoint path |
| `--from-hub` | `None` | Hugging Face Hub `repo_id` to download and run instead of `--model` |
| `--hub-filename` | `model.zip` | Filename within the Hub repo (only with `--from-hub`) |
| `--episodes` | `3` | Evaluation episodes |
| `--backend` | `inprocess` | `inprocess`, `harness`, or `peragent` |

## Deploying Published Models

A solved Walker2d checkpoint (1522.9 reward, 999 of 1000 steps, 10 of 10
evaluation episodes at the cap) is published to
[`CursedRock17/gazebo-gymnasium-policies/walker2d`](https://huggingface.co/CursedRock17/gazebo-gymnasium-policies).
Running it needs no separate download step:

```bash
pixi run deploy --agent walker2d --n_agents 4 \
    --from-hub CursedRock17/gazebo-gymnasium-policies \
    --hub-filename walker2d/model.zip
```

Or pull it directly with the Hub CLI, or `huggingface_sb3` in Python:

```bash
hf download CursedRock17/gazebo-gymnasium-policies walker2d/model.zip --local-dir models/
```

```python
from huggingface_sb3 import load_from_hub

checkpoint = load_from_hub(
    repo_id="CursedRock17/gazebo-gymnasium-policies",
    filename="walker2d/model.zip",
)
```

## Screenshot

![Walker2d mid-stride, both legs visible in a walking gait](../images/walker2d_solved_gui.png)

A real spectator-camera capture of the solved policy mid-rollout, not a
mockup; see [README.md](README.md#what-solved-policies-look-like).

## Reward Space

`forward_progress_reward(vel_index=8)`: forward velocity, plus a 1.0 alive
bonus, minus 0.001 times the summed squared action.

```
reward = root_fwd_velocity + 1.0 - 0.001 * sum(action**2)
```

**Termination.** `planar_health_termination`: torso height outside
`(0.8, 2.0)`, `|root_pitch| >= 1.0` radians, or any observation value at or
above 100 in magnitude. `max_episode_steps=1000`. Spawn height 1.25 m.

## Glossary

| Term | Definition |
|---|---|
| **Entity Component Manager (ECM)** | Gazebo's runtime interface for reading and writing simulated joint and link state |
| **Proximal Policy Optimization (PPO)** | The default reinforcement learning algorithm used across the framework's specifications |
| **AgentSpec** | The dataclass describing one agent's model, observation, action, reward, and DR configuration |
| **Frame skip** | Number of physics steps the same action is held for before the next observation; 4 here |

## Difficulty

<!-- TODO: rate easy / medium / hard relative to the other environments. -->

## Domain Randomization

Not implemented for this environment: `walker2d`'s `AgentSpec` sets no
domain-randomization fields. See
[`domain_randomization.md`](../domain_randomization.md) for the mechanism
`line_follower` uses, if this environment ever needs sim-to-real transfer.

## Training Arguments

`train.py`'s flags, shared across every registered agent.

| Argument | Default | Effect |
|---|---|---|
| `--agent` | `cartpole` | Registered spec name: pass `walker2d` |
| `--n_agents` | `4` | Agents in one sim (1 = single-agent) |
| `--algo` | `ppo` | `ppo`, `a2c`, `sac`, `td3`, or `ddpg`; this environment's `Box(6,)` action works with all five |
| `--timesteps` | `200000` | Total environment steps |
| `--backend` | `inprocess` | `inprocess`, `harness`, or `peragent` |
| `--tensorboard` | off | Log to `models/<agent>_multi/tb/` |
| `--wandb` | off | Also mirror the run to Weights & Biases (implies `--tensorboard`) |
| `--push-to-hub` | `None` | Upload the trained model + an auto-generated model card to this Hugging Face Hub `repo_id` |
| `--hub-eval-episodes` | `5` | Deterministic eval episodes used to fill in the model card (only with `--push-to-hub`) |

## What Good Results Look Like

Solved bar: 1000-step cap, reward at or above 1500. Best verified result:
**1522.9 reward, 999 of 1000 steps (10 of 10 real evaluation episodes) at
600,000 steps**, full detail in
[README.md's Solved Bars Explained](README.md#solved-bars-explained).

![Hyperparameter sweep curves for InvertedDoublePendulum, Hopper, and Walker2d, 4 real trials per environment, winning trial highlighted](../images/mujoco_sweep_curves.png)

![Deterministic-evaluation verification of the resumed Hopper and Walker2d checkpoints, crossing their solved bars](../images/mujoco_resume_verification.png)

## Trying It With Different Libraries

```python
# 1. Stable-Baselines3, through its own VecEnv API.
import stable_baselines3 as sb3
from gazebo_gymnasium_bridge.envs import make_inprocess

vec = make_inprocess("walker2d", n_agents=16)
sb3.PPO("MlpPolicy", vec, n_steps=64).learn(total_timesteps=1_000_000)

# 2. Any Gymnasium tool (RLlib, CleanRL, Tianshou, TorchRL), a standard env.
import gazebo_gymnasium_bridge
import gymnasium as gym
env = gym.make("GazeboWalker2d-v0")

# 3. The native gymnasium vector env.
vec = gym.make_vec("GazeboWalker2d-v0", num_envs=16,
                   vectorization_mode="vector_entry_point")
```

[`docs/rl_libraries.md`](../rl_libraries.md) covers skrl, rl_games, and
rsl_rl too, each through its own `training_scripts/train_<library>.py
--agent walker2d`, inside the `rl-libs` pixi environment. `walker2d` is one
of the agents that page names explicitly as tested with all three.

## Version History

<!-- TODO -->
