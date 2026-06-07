# CartPole

Classic CartPole-v1 reproduced in Gazebo Harmonic. A cart slides along a rail
on the Y axis. A pole is hinged on top of the cart and rotates around the X
axis. The agent applies a velocity command to the cart and tries to keep the
pole upright.

This is the reference example. If you're adding a new environment, start by
reading this and copy the structure.

## What you get

- A continuously-running Gazebo world with the cart+pole+ground plane
- A sync-gate plugin (`cartpole_learner.py`) loaded into `gz sim` that proxies
  actions/observations between the agent and the world
- A plain `gym.Env` (`GazeboCartPoleEnv`) that any RL library can drive
- Two reference training scripts at the project root —
  `train_cartpole_sb3.py` and `train_cartpole_custom.py`

## Running it

Easiest path — one launch, one terminal:

```bash
ros2 launch gazebo_gymnasium_bringup cartpole_train.launch.py
```

That launches `gz sim` AND the SB3 PPO trainer. The trainer waits a few
seconds for the sim's services to come up, then starts learning.

Two-terminal version (lets you swap trainers without restarting the sim):

```bash
# Terminal 1 — simulator
ros2 launch gazebo_gymnasium_bringup cartpole.launch.py

# Terminal 2 — pick one
python training_scripts/train_cartpole_sb3.py
# or
python training_scripts/train_cartpole_custom.py
```

### Launch options

All toggleable from the CLI like `verbose:=false`:

| Arg | Default | What it does |
|-----|---------|--------------|
| `verbose` | `true` | Per-step debug print in the env. Throttle with `GAZEBO_GYM_VERBOSE_EVERY=N` env var. |
| `use_foxglove` | `false` | Launches `foxglove_bridge` (port 8765) + `ros_gz_bridge` so you can watch the run live in Foxglove Studio. |
| `foxglove_port` | `8765` | Override the Foxglove WebSocket port. |
| `record_rosbag` | `false` | Records every topic to `rosbags/cartpole_<timestamp>` in MCAP format. |
| `trainer` *(combined launch only)* | `sb3` | `sb3` or `custom` — which training script to run. |
| `timesteps` *(combined launch only)* | `200000` | total_timesteps for the trainer (via `GAZEBO_GYM_TIMESTEPS` env var). |

Example with everything on:

```bash
ros2 launch gazebo_gymnasium_bringup cartpole_train.launch.py \
    use_foxglove:=true record_rosbag:=true trainer:=sb3
```

After it runs, you can replay the MCAP file in Foxglove Studio for offline
analysis:

```
File → Open... → Open local file... → rosbags/cartpole_<timestamp>/<bag>.mcap
```

## The algorithm: PPO

We use **Proximal Policy Optimization** (PPO) ([Schulman et al. 2017](https://arxiv.org/abs/1707.06347)),
the same algorithm used by SB3's default CartPole baseline. PPO is an on-policy
actor-critic method:

- **Actor** (policy network): given the observation, outputs probabilities over
  the discrete actions. Here it's a 2-layer MLP (64×64 hidden).
- **Critic** (value network): estimates the expected return from the current
  state. Same architecture.

Each "rollout," the agent collects `n_steps` (256) transitions, then runs
`n_epochs` (10) passes of mini-batch SGD over them, optimizing a clipped
surrogate objective with `clip_range=0.2`. The clip keeps the new policy from
diverging too far from the old in one update — that's the "Proximal" part.

Hyperparameters in `train_cartpole_sb3.py`:

| Param | Value | Notes |
|-------|-------|-------|
| `n_steps` | 256 | Rollout size per update |
| `batch_size` | 64 | Mini-batch size for the SGD passes |
| `n_epochs` | 10 | SGD passes over each rollout |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE λ — bias/variance tradeoff in advantage estimates |
| `clip_range` | 0.2 | PPO surrogate clip |
| `ent_coef` | 0.001 | Entropy bonus — keeps policy stochastic. **Critical**: at the SB3 default of 0.0 the policy collapses too early on this task. |
| `learning_rate` | 3e-4 | Adam LR for both actor and critic |

The custom PPO in `train_cartpole_custom.py` uses the same hyperparameters with
the project's hand-rolled `PPOAgent` in
`gazebo_gymnasium_reinforcement_learning/deep_learning/PPO_agent.py`. Useful
when you want to look at every gradient yourself instead of treating PPO as a
black box.

## Observation space

`Box(4,) float32` — matches canonical Gymnasium CartPole-v1.

| Index | Field | Range | Source |
|-------|-------|-------|--------|
| 0 | Cart position (Y axis) | [-4.8, 4.8] m | `/world/cartpole/pose/info` → cart link |
| 1 | Cart velocity | unbounded | `/world/cartpole/.../joint_state` → slider_to_cart |
| 2 | Pole angle (around X) | [-0.4189, 0.4189] rad | `pose.orientation` → `2*atan2(q.x, q.w)` |
| 3 | Pole angular velocity | unbounded | `/.../joint_state` → cart_to_pole |

The plugin reads these from existing gz model plugins (`JointStatePublisher`,
`SceneBroadcaster`) on every physics tick and publishes them aggregated on
`/env/state` once per `frame_skip=5` ticks.

## Action space

`Discrete(2)`:

- `0` → publish `cmd_vel = -1.0 m/s` on the slider joint
- `1` → publish `cmd_vel = +1.0 m/s` on the slider joint

The model's `JointController` plugin reads `cmd_vel` and applies torque to
track the commanded velocity. A "grace period" in the sync-gate plugin sets
`cmd_vel = 0` once `frame_skip` ticks have passed since the last action —
this prevents the cart from running away during SB3's `model.train()` pause
between rollouts.

## Reward function

`+1.0` for every env step where neither termination condition is met. The
agent's job is to *survive* as long as possible.

- `terminated = True` when `|pole_angle| > 0.20944 rad` (≈ 12°) OR
  `|cart_position| > 2.4 m`
- `truncated = True` when `current_step >= max_episode_steps` (default 500)

Both conditions are checked **on the env side**, not in the plugin. The plugin
stays generic; only `GazeboCartPoleEnv.step()` knows the CartPole-specific
failure thresholds.

## Expected training trajectory

Tested with SB3 PPO + the hyperparameters above, on a typical desktop CPU:

| Approx episode | Behavior |
|----------------|----------|
| 0–50 | Random-equivalent. Episodes end in ~30 steps because the cart drifts to one side or the pole tips. |
| 50–300 | Mean episode length climbs steadily. PPO is finding the obvious bang-bang controller. |
| 300–700 | Mean episode length 150–250. Episodes routinely reach `max_steps_per_episode=500` (truncated). |
| 700+ | Solved. Most episodes hit the 500-step cap. Occasional pole-tips when the policy slips, but immediately recovers. |

`[EpisodeSummary]` lines tell you which condition ended each episode:

```
[EpisodeSummary] ep=863 steps=500 phys_ticks=2500 reward=500.0 reason=truncated max_pole_a=+0.203 max_cart_p=-0.350 final_pole_a=+0.065 final_cart_p=-0.140 mean_step_ms=48.8
```

`reason=truncated` + `max_pole_a` under 0.21 = the agent balanced for the
entire episode. `reason=terminated` + `max_pole_a` at exactly the threshold
= real pole-tip. `reason=timeout` = the env's step service wait hit its
5-second ceiling, usually meaning the sim has stalled.

## Watching in Foxglove

```bash
ros2 launch gazebo_gymnasium_bringup cartpole.launch.py use_foxglove:=true
```

In Foxglove Studio:

1. Connect to **WebSocket** with URL `ws://localhost:8765`.
2. Add a **3D** panel — drag `/tf` into the topics. You'll see the cart+pole
   pose updating live.
3. Add a **Plot** panel — add `/joint_states.position[0]` (slider position)
   and `/joint_states.position[1]` (pole angle).

For offline review of a recorded `.mcap` file: open Foxglove → "Open local
file" → pick the bag. Same panel layout works against the recorded data.

## Reference files

- World: [`gazebo_gymnasium_resources/worlds/cartpole.sdf`](../../gazebo_gymnasium_examples/gazebo_gymnasium_resources/worlds/cartpole.sdf)
- Model: [`gazebo_gymnasium_resources/models/cartpole/cartpole.sdf`](../../gazebo_gymnasium_examples/gazebo_gymnasium_resources/models/cartpole/cartpole.sdf)
- Sync-gate plugin: [`gazebo_gymnasium_resources/plugins/cartpole_learner.py`](../../gazebo_gymnasium_examples/gazebo_gymnasium_resources/plugins/cartpole_learner.py)
- Env class: [`gazebo_gymnasium_bridge/envs/cartpole.py`](../../gazebo_gymnasium_bridge/gazebo_gymnasium_bridge/envs/cartpole.py)
- SB3 trainer: [`train_cartpole_sb3.py`](../../training_scripts/train_cartpole_sb3.py)
- Custom PPO trainer: [`train_cartpole_custom.py`](../../training_scripts/train_cartpole_custom.py)
- Launch (sim only): [`cartpole.launch.py`](../../gazebo_gymnasium_examples/gazebo_gymnasium_bringup/launch/cartpole.launch.py)
- Launch (sim + trainer): [`cartpole_train.launch.py`](../../gazebo_gymnasium_examples/gazebo_gymnasium_bringup/launch/cartpole_train.launch.py)
