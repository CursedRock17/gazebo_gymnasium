# Hopper

Hopper is a MuJoCo port: a single-legged robot that hops forward on one
springy leg, driven by three actuated hinges. It was the first
ground-contact locomotion environment built for this library, and the
physics work behind it carried over almost unchanged to Walker2d and
HalfCheetah afterward. [Porting Hopper, an annotated
walkthrough](porting_hopper.md) narrates how it was built; this page
documents what it is.

## Overview

A planar monopod: torso, thigh, leg, and foot, connected by three
actuated hinges. The root is modeled as three explicit joints (a forward
slide, a vertical slide, and a pitch hinge) rather than a free body,
matching MuJoCo's own Hopper convention rather than giving it a free-
floating base the way [Ant](ant.md) has. The agent drives hip, knee, and
ankle torque directly; everything else is physics.

## Objective

Hop forward without falling over. Reward accumulates every step the
robot stays healthy (torso above 0.7 m, pitch within about 11 degrees),
so in practice the objective is staying upright while making forward
progress, not just surviving in place; a robot that never moves at all
still collects the alive bonus every step.

## Observation Space

`Box(11,)`, `float32`, unbounded. MuJoCo's own layout: positions first
(root's forward slide excluded), then all velocities.

| Index | Name | Type | Min | Max |
|---|---|---|---|---|
| 0 | `root_up` position (torso height above spawn) | `float32` | −inf | inf |
| 1 | `root_pitch` position | `float32` | −inf | inf |
| 2 | `thigh_joint` position | `float32` | −inf | inf |
| 3 | `leg_joint` position | `float32` | −inf | inf |
| 4 | `foot_joint` position | `float32` | −inf | inf |
| 5 | `root_fwd` velocity (forward speed) | `float32` | −inf | inf |
| 6 | `root_up` velocity | `float32` | −inf | inf |
| 7 | `root_pitch` velocity | `float32` | −inf | inf |
| 8 | `thigh_joint` velocity | `float32` | −inf | inf |
| 9 | `leg_joint` velocity | `float32` | −inf | inf |
| 10 | `foot_joint` velocity | `float32` | −inf | inf |

## Action Space

`Box(3,)`, `float32`, normalized to `[-1, 1]` per joint, scaled to ±200 N·m
(MuJoCo's gear-200 convention).

| # | Action | Min | Max | Joint | Name | Type |
|---|---|---|---|---|---|---|
| 0 | Thigh torque | −200 N·m | 200 N·m | revolute | `thigh_joint` | `float32` |
| 1 | Leg torque | −200 N·m | 200 N·m | revolute | `leg_joint` | `float32` |
| 2 | Foot torque | −200 N·m | 200 N·m | revolute | `foot_joint` | `float32` |

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
pixi run train --agent hopper --n_agents 16 --timesteps 600000
```

600,000 steps is the budget that reached the solved bar, resumed from a
300,000-step checkpoint; see
[Confirming Learnability](porting_hopper.md#confirming-learnability) and
[README.md](README.md#solved-bars-explained) for the full path there.

<!-- TODO: once/if a harness launch file exists for this environment,
     document the launched + GUI workflow here, matching cartpole.md. -->

## Evaluation

```bash
pixi run deploy --agent hopper --n_agents 4
```

## CLI Arguments

`deploy.py`'s Hugging Face Hub and agent-count flags:

| Argument | Default | Effect |
|---|---|---|
| `--agent` | `cartpole` | Registered spec name: pass `hopper` |
| `--n_agents` | `4` | Number of agents evaluated in parallel |
| `--model` | `models/<agent>_multi/final_<algo>_n<N>.zip` | Local checkpoint path |
| `--from-hub` | `None` | Hugging Face Hub `repo_id` to download and run instead of `--model` |
| `--hub-filename` | `model.zip` | Filename within the Hub repo (only with `--from-hub`) |
| `--episodes` | `3` | Evaluation episodes |
| `--backend` | `inprocess` | `inprocess`, `harness`, or `peragent` |

## Deploying Published Models

A solved Hopper checkpoint (1002.2 reward, 999 of 1000 steps, 10 of 10
evaluation episodes at the cap) is published to
[`CursedRock17/gazebo-gymnasium-policies/hopper`](https://huggingface.co/CursedRock17/gazebo-gymnasium-policies).
Running it needs no separate download step:

```bash
pixi run deploy --agent hopper --n_agents 4 \
    --from-hub CursedRock17/gazebo-gymnasium-policies \
    --hub-filename hopper/model.zip
```

Or pull it directly with the Hub CLI, or `huggingface_sb3` in Python:

```bash
hf download CursedRock17/gazebo-gymnasium-policies hopper/model.zip --local-dir models/
```

```python
from huggingface_sb3 import load_from_hub

checkpoint = load_from_hub(
    repo_id="CursedRock17/gazebo-gymnasium-policies",
    filename="hopper/model.zip",
)
```

## Screenshot

![Hopper standing balanced upright, solved policy, real spectator-camera capture](../images/hopper_solved_gui.png)

A real spectator-camera capture of the solved policy mid-rollout, not a
mockup. The pose is genuinely almost stationary, not a capture-timing
accident; see [Porting Hopper's closing section](porting_hopper.md#confirming-learnability)
for why.

## Reward Space

`forward_progress_reward(vel_index=5)`: forward velocity, plus a 1.0 alive
bonus, minus 0.001 times the summed squared action.

```
reward = root_fwd_velocity + 1.0 - 0.001 * sum(action**2)
```

**Termination.** `planar_health_termination`: torso height outside
`(0.7, inf)`, `|root_pitch| >= 0.2` radians, or any observation value at or
above 100 in magnitude. `max_episode_steps=1000`. Spawn height 1.25 m.

## Glossary

| Term | Definition |
|---|---|
| **Entity Component Manager (ECM)** | Gazebo's runtime interface for reading and writing simulated joint and link state |
| **Proximal Policy Optimization (PPO)** | The default reinforcement learning algorithm used across the framework's specifications |
| **AgentSpec** | The dataclass describing one agent's model, observation, action, reward, and DR configuration |
| **Frame skip** | Number of physics steps the same action is held for before the next observation; 4 here |

## Difficulty

**Medium.** Porting the physics took real debugging, not a clean first
try: random actions survived only about 4 steps until three separate
traps were diagnosed and fixed (missing joint damping, missing armature,
and unbounded joint configurations), detailed in [Porting Hopper's
"Building the Bare Model" section](porting_hopper.md#building-the-bare-model-through-three-traps).
Walker2d, by contrast, reused this same recipe and had its physics
working on the first try. Once specified correctly, though, reaching the
solved bar took only published PPO hyperparameters and a longer training
budget (resumed from 300,000 to 600,000 steps), no architecture changes
and no reward redesign, unlike [Ant](ant.md), which remains an open
problem even with the model itself working correctly.

## Domain Randomization

Not implemented for this environment: `hopper`'s `AgentSpec` sets no
domain-randomization fields. See
[`domain_randomization.md`](../domain_randomization.md) for the mechanism
`line_follower` uses, if this environment ever needs sim-to-real transfer.

## Training Arguments

`train.py`'s flags, shared across every registered agent.

| Argument | Default | Effect |
|---|---|---|
| `--agent` | `cartpole` | Registered spec name: pass `hopper` |
| `--n_agents` | `4` | Agents in one sim (1 = single-agent) |
| `--algo` | `ppo` | `ppo`, `a2c`, `sac`, `td3`, or `ddpg`; this environment's `Box(3,)` action works with all five |
| `--timesteps` | `200000` | Total environment steps |
| `--backend` | `inprocess` | `inprocess`, `harness`, or `peragent` |
| `--tensorboard` | off | Log to `models/<agent>_multi/tb/` |
| `--wandb` | off | Also mirror the run to Weights & Biases (implies `--tensorboard`) |
| `--push-to-hub` | `None` | Upload the trained model + an auto-generated model card to this Hugging Face Hub `repo_id` |
| `--hub-eval-episodes` | `5` | Deterministic eval episodes used to fill in the model card (only with `--push-to-hub`) |

## What Good Results Look Like

Solved bar: 1000-step cap, reward at or above 1000. Best verified result:
**1002.2 reward, 999 of 1000 steps (10 of 10 real evaluation episodes) at
600,000 steps**, full detail in
[README.md's Solved Bars Explained](README.md#solved-bars-explained).

![Hyperparameter sweep curves for InvertedDoublePendulum, Hopper, and Walker2d, 4 real trials per environment, winning trial highlighted](../images/mujoco_sweep_curves.png)

![Deterministic-evaluation verification of the resumed Hopper and Walker2d checkpoints, crossing their solved bars](../images/mujoco_resume_verification.png)

Hopper's panel in that second chart stays flat by design, not a charting
bug; see [README.md](README.md#what-solved-policies-look-like) for why.

## Trying It With Different Libraries

```python
# 1. Stable-Baselines3, through its own VecEnv API.
import stable_baselines3 as sb3
from gazebo_gymnasium_bridge.envs import make_inprocess

vec = make_inprocess("hopper", n_agents=16)
sb3.PPO("MlpPolicy", vec, n_steps=64).learn(total_timesteps=1_000_000)

# 2. Any Gymnasium tool (RLlib, CleanRL, Tianshou, TorchRL), a standard env.
import gazebo_gymnasium_bridge
import gymnasium as gym
env = gym.make("GazeboHopper-v0")

# 3. The native gymnasium vector env.
vec = gym.make_vec("GazeboHopper-v0", num_envs=16,
                   vectorization_mode="vector_entry_point")
```

[`docs/rl_libraries.md`](../rl_libraries.md) covers skrl, rl_games, and
rsl_rl too, each through its own `training_scripts/train_<library>.py
--agent hopper`, inside the `rl-libs` pixi environment. `hopper` is one of
the agents that page names explicitly as tested with all three.

## Version History

| Date | Change |
|---|---|
| 2026-07-18 | Ported via the planar-joint-root recipe; the solved bar (1000-step cap, reward at or above 1000) defined the same day |
| 2026-07-19 | [Porting Hopper, annotated](porting_hopper.md) published, narrating the port stage by stage |
| 2026-07-20 | Shipped in the `v0.1.0` first public release, one of 8 registered specs |
| Unreleased | Reached the solved bar (1002.2 reward, 999 of 1000 steps) by resuming the winning 300,000-step sweep configuration to 600,000 steps; not yet in a tagged release |
