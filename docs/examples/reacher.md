# Reacher

<!-- TODO: expand this intro paragraph. A 2-link horizontal-plane arm
     with a "goal-as-joints" target: the goal is two prismatic joints
     rather than a separate free body, so it resets through the same
     joint-reset path as the arm itself. -->

## Overview

<!-- TODO -->

## Objective

<!-- TODO -->

## Observation Space

`Box(6,)`, `float32`, unbounded.

| Index | Name | Type | Min | Max |
|---|---|---|---|---|
| 0 | `joint0` position | `float32` | −inf | inf |
| 1 | `joint1` position | `float32` | −inf | inf |
| 2 | `target_x` position | `float32` | −inf | inf |
| 3 | `target_y` position | `float32` | −inf | inf |
| 4 | `joint0` velocity | `float32` | −inf | inf |
| 5 | `joint1` velocity | `float32` | −inf | inf |

## Action Space

`Box(2,)`, `float32`, normalized to `[-1, 1]` per joint, scaled to ±5 N·m.
`target_x`/`target_y` are not actuated: they're the goal position, not
part of the action space.

| # | Action | Min | Max | Joint | Name | Type |
|---|---|---|---|---|---|---|
| 0 | Shoulder torque | −5 N·m | 5 N·m | revolute | `joint0` | `float32` |
| 1 | Elbow torque | −5 N·m | 5 N·m | revolute | `joint1` | `float32` |

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
pixi run train --agent reacher --n_agents 16 --timesteps 200000
```

<!-- TODO: once/if a harness launch file exists for this environment,
     document the launched + GUI workflow here, matching cartpole.md. -->

## Evaluation

```bash
pixi run deploy --agent reacher --n_agents 4
```

## CLI Arguments

`deploy.py`'s Hugging Face Hub and agent-count flags:

| Argument | Default | Effect |
|---|---|---|
| `--agent` | `cartpole` | Registered spec name: pass `reacher` |
| `--n_agents` | `4` | Number of agents evaluated in parallel |
| `--model` | `models/<agent>_multi/final_<algo>_n<N>.zip` | Local checkpoint path |
| `--from-hub` | `None` | Hugging Face Hub `repo_id` to download and run instead of `--model` |
| `--hub-filename` | `model.zip` | Filename within the Hub repo (only with `--from-hub`) |
| `--episodes` | `3` | Evaluation episodes |
| `--backend` | `inprocess` | `inprocess`, `harness`, or `peragent` |

## Deploying Published Models

An unsolved but real checkpoint (−12.4 reward over 50 steps, PPO at
100,000 steps) is published to
[`CursedRock17/gazebo-gymnasium-policies/reacher`](https://huggingface.co/CursedRock17/gazebo-gymnasium-policies).
Running it needs no separate download step:

```bash
pixi run deploy --agent reacher --n_agents 4 \
    --from-hub CursedRock17/gazebo-gymnasium-policies \
    --hub-filename reacher/model.zip
```

Or pull it directly with the Hub CLI, or `huggingface_sb3` in Python:

```bash
hf download CursedRock17/gazebo-gymnasium-policies reacher/model.zip --local-dir models/
```

```python
from huggingface_sb3 import load_from_hub

checkpoint = load_from_hub(
    repo_id="CursedRock17/gazebo-gymnasium-policies",
    filename="reacher/model.zip",
)
```

## Screenshot

<!-- TODO: no spectator-camera capture exists for this environment yet
     (unlike InvertedDoublePendulum, Hopper, and Walker2d; see
     README.md's "What Solved Policies Look Like"). Add one under
     docs/images/reacher_gui.png once a policy is worth capturing. -->

## Reward Space

Dense distance-to-goal reward, minus a small control cost.

```
fingertip = forward_kinematics(joint0, joint1)   # L0=0.1 m, L1=0.11 m
dist = distance(fingertip, (target_x, target_y))
reward = -dist - 0.1 * sum(clip(action, -1, 1)**2)
```

**Termination.** A pure numerical guard ends the episode if any
observation value reaches or exceeds 100 in magnitude; otherwise episodes
run to truncation. `max_episode_steps=50` (`frame_skip=2`, matching
MuJoCo's own `dt=0.02` Reacher). Spawn height 0.05 m.

## Glossary

| Term | Definition |
|---|---|
| **Entity Component Manager (ECM)** | Gazebo's runtime interface for reading and writing simulated joint and link state |
| **Proximal Policy Optimization (PPO)** | The default reinforcement learning algorithm used across the framework's specifications |
| **Soft Actor-Critic (SAC)** | An off-policy algorithm suited to continuous action spaces; the best known result on this environment |
| **AgentSpec** | The dataclass describing one agent's model, observation, action, reward, and DR configuration |
| **Goal-as-joints** | Representing a target position as two prismatic joints instead of a free body, so it resets through the same joint-reset path as the rest of the model |

## Difficulty

<!-- TODO: rate easy / medium / hard relative to the other environments. -->

## Domain Randomization

Not implemented for this environment: `reacher`'s `AgentSpec` sets no
domain-randomization fields. See
[`domain_randomization.md`](../domain_randomization.md) for the mechanism
`line_follower` uses, if this environment ever needs sim-to-real transfer.

## Training Arguments

`train.py`'s flags, shared across every registered agent.

| Argument | Default | Effect |
|---|---|---|
| `--agent` | `cartpole` | Registered spec name: pass `reacher` |
| `--n_agents` | `4` | Agents in one sim (1 = single-agent) |
| `--algo` | `ppo` | `ppo`, `a2c`, `sac`, `td3`, or `ddpg`; this environment's `Box(2,)` action works with all five |
| `--timesteps` | `200000` | Total environment steps |
| `--backend` | `inprocess` | `inprocess`, `harness`, or `peragent` |
| `--tensorboard` | off | Log to `models/<agent>_multi/tb/` |
| `--wandb` | off | Also mirror the run to Weights & Biases (implies `--tensorboard`) |
| `--push-to-hub` | `None` | Upload the trained model + an auto-generated model card to this Hugging Face Hub `repo_id` |
| `--hub-eval-episodes` | `5` | Deterministic eval episodes used to fill in the model card (only with `--push-to-hub`) |

## What Good Results Look Like

Solved bar: mean episode reward at or above −5 (near the goal for most of
the episode). Best verified result: **SAC, −8.48 at 100,000 steps**,
ahead of PPO's −10.4 at 200,000 steps with published hyperparameters, not
yet across the bar. Full detail in
[README.md's Solved Bars Explained](README.md#solved-bars-explained).

<!-- TODO: add a per-environment tensorboard/reward-curve image once one
     exists for this environment specifically. -->

## Trying It With Different Libraries

```python
# 1. Stable-Baselines3, through its own VecEnv API.
import stable_baselines3 as sb3
from gazebo_gymnasium_bridge.envs import make_inprocess

vec = make_inprocess("reacher", n_agents=16)
sb3.PPO("MlpPolicy", vec, n_steps=64).learn(total_timesteps=1_000_000)

# 2. Any Gymnasium tool (RLlib, CleanRL, Tianshou, TorchRL), a standard env.
import gazebo_gymnasium_bridge
import gymnasium as gym
env = gym.make("GazeboReacher-v0")

# 3. The native gymnasium vector env.
vec = gym.make_vec("GazeboReacher-v0", num_envs=16,
                   vectorization_mode="vector_entry_point")
```

[`docs/rl_libraries.md`](../rl_libraries.md) covers skrl, rl_games, and
rsl_rl too, each through its own `training_scripts/train_<library>.py
--agent reacher`, inside the `rl-libs` pixi environment. `reacher` is one
of the agents that page names explicitly as tested with all three.

## Version History

<!-- TODO -->
