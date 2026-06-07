# InvertedPendulum

Continuous-control analog of CartPole. A cart slides along a horizontal rail,
a pole hinged on top of the cart can rotate freely, and the agent applies a
**continuous force** (not a discrete left/right push) to keep the pole
upright. This is the second working example and the first **continuous-action**
environment in the repo.

Model ported from Gymnasium's
[`InvertedPendulum-v5`](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/)
MuJoCo MJCF. Conversion pipeline:

```
mujoco_sources/inverted_pendulum.xml
        │  hand-written URDF
        ▼
models/inverted_pendulum/inverted_pendulum.urdf
        │  gz sdf -p
        ▼
models/inverted_pendulum/inverted_pendulum.sdf
```

See `models/inverted_pendulum/inverted_pendulum.urdf` for the hand-port commentary
(capsule → cylinder substitutions, why each link is shaped the way it is).

## What you get

- The InvertedPendulum model anchored to the world via a fixed joint
- A `gz-sim-apply-joint-force-system` plugin on the slider so the agent can
  apply scalar forces directly
- A sync-gate plugin (`inverted_pendulum_learner.py`) that publishes 4-tuple
  state and accepts a continuous action
- `GazeboInvertedPendulumEnv` — plain `gym.Env` with `Box(1,)` action space
- A reference SAC training script (`train_inverted_pendulum_sac.py`)

## Running it

Two-terminal flow:

```bash
# Terminal 1 — simulator
ros2 launch gazebo_gymnasium_bringup inverted_pendulum.launch.py

# Terminal 2 — SAC training
python training_scripts/train_inverted_pendulum_sac.py
```

Same launch flags as CartPole (`verbose`, `use_foxglove`, `foxglove_port`,
`record_rosbag`).

## The algorithm: SAC

**Soft Actor-Critic** ([Haarnoja et al. 2018](https://arxiv.org/abs/1801.01290))
is the standard MuJoCo/Gymnasium baseline for continuous-control tasks. Key
differences from the PPO we use on CartPole:

- **Off-policy:** maintains a replay buffer; reuses old experience for many
  gradient updates. Much more sample-efficient than PPO on continuous tasks.
- **Maximum entropy:** explicitly rewards the policy for staying stochastic.
  The "soft" in SAC. Removes the manual `ent_coef` tuning we had to do for
  PPO/CartPole.
- **Two Q-networks (twin critics):** combats overestimation bias on the
  value function. Standard practice in modern off-policy methods.
- **Automatic entropy temperature tuning:** `ent_coef="auto"` lets SAC pick
  the entropy coefficient itself based on a target entropy.

Hyperparameters in `train_inverted_pendulum_sac.py`:

| Param | Value | Notes |
|-------|-------|-------|
| `learning_rate` | 3e-4 | Adam LR for all networks |
| `buffer_size` | 100,000 | Replay buffer capacity |
| `learning_starts` | 100 | Random exploration before training kicks in |
| `batch_size` | 256 | Gradient batch size |
| `tau` | 0.005 | Target network polyak averaging |
| `gamma` | 0.99 | Discount factor |
| `train_freq` | 1 | One gradient step per env step |
| `gradient_steps` | 1 | One critic update per train_freq |
| `ent_coef` | `"auto"` | Auto-tuned target entropy |

SAC typically solves InvertedPendulum in **5,000–20,000 env steps** on
real-time-equivalent dynamics. With our `frame_skip=4` window per env step
(40 ms of sim time), expect 5–20 minutes of wall clock.

## Observation space

`Box(4,) float32` — matches canonical Gymnasium InvertedPendulum-v5
ordering (note: positions first, then velocities; **different from CartPole-v1's
interleaved layout**).

| Index | Field | Source |
|-------|-------|--------|
| 0 | Cart position (along X) | `pose.position.x` of the `cart` link |
| 1 | Pole angle (around Y) | `2 * atan2(q.y, q.w)` from pole's quaternion |
| 2 | Cart velocity | `slider` joint velocity |
| 3 | Pole angular velocity | `hinge` joint velocity |

The slider's joint limit (±1 m, from MJCF `range="-1 1"`) physically bounds
the cart, so unlike CartPole-v1 there's no cart-position termination
condition.

## Action space

`Box(low=-3, high=3, shape=(1,), dtype=float32)` — a scalar motor command.
The sync-gate plugin multiplies by **gear=100** (matching MJCF
`motor gear="100"`) before publishing to `/model/inverted_pendulum/joint/slider/cmd_force`,
so effective force on the slider is in [-300, 300] N.

SAC's policy outputs values in [-1, 1] and rescales internally to match the
declared action space — no normalization is needed on the env side.

## Reward function

`+1.0` per step where the pole is upright (`|pole_angle| ≤ 0.2 rad ≈ 11.5°`).
`reward = 0` is never returned — termination fires the moment the threshold
is exceeded.

- `terminated = True` when `|pole_angle| > 0.2 rad` OR any observation is
  non-finite (e.g. NaN, infinite — guards against numerical blowups)
- `truncated = True` when `current_step >= max_episode_steps` (default 1000)

Note: the canonical InvertedPendulum-v5 does NOT have a cart-position
termination. The joint limit handles that physically.

## Watching in Foxglove

```bash
ros2 launch gazebo_gymnasium_bringup inverted_pendulum.launch.py use_foxglove:=true
```

Connect Foxglove Studio to `ws://localhost:8765`. The launch bridges five
topics for visualization:

| Topic | Type | What it shows |
|-------|------|---------------|
| `/tf` | `tf2_msgs/TFMessage` | Live cart + pole pose (3D panel) |
| `/joint_states` | `sensor_msgs/JointState` | `position[1]` = pole angle, `velocity[1]` = pole angular velocity (Plot panel) |
| `/world/inverted_pendulum/stats` | `ros_gz_interfaces/WorldStatistics` | `real_time_factor` — **is the sim keeping up?** RTF=1.0 means real-time; <1.0 means the sim is slower than wall clock. Also exposes `iterations` (total physics ticks) |
| `/world/inverted_pendulum/clock` | `rosgraph_msgs/Clock` | Sim time |
| `/env/metrics` | `ros_gz_interfaces/Float32Array` | `data[0]` = env steps/sec, `data[1]` = mean step latency (ms), `data[2]` = current episode reward, `data[3]` = current episode step count, `data[4]` = **total training steps** (cumulative across all episodes — graph reward vs. this to track learning curve), `data[5]` = **current episode index** (1-based) |

The two FPS metrics measure different things:
- `stats.real_time_factor` is gz sim's view — how fast physics is advancing
- `metrics.data[0]` is the agent loop's view — how many env.step() calls
  per second the trainer is achieving (limited by Gazebo speed AND by the
  RL library's compute per update)

If `data[0]` is much lower than what `stats.real_time_factor` would
suggest, the RL update step itself is the bottleneck. If they're similar
and both low, Gazebo is the bottleneck.

If you also pass `record_rosbag:=true`, the MCAP at
`rosbags/inv_pendulum_<timestamp>/` loads directly into Foxglove for
replay.

## What's intentionally simple about this port

- **No initial-state randomization.** Canonical MuJoCo InvertedPendulum-v5
  adds uniform noise of ±0.01 to `qpos` and `qvel` at reset. We just reset
  the world to zero. The agent has to learn to recover from perfectly upright
  initial conditions, which it does fine — but with broader robustness if
  you eventually add the noise.
- **`max_episode_steps=1000` matches canonical.**
- **Reward shaping is unchanged from the MuJoCo source** (+1 per step alive).

## Reference files

- Source MJCF: [`mujoco_sources/inverted_pendulum.xml`](../../gazebo_gymnasium_examples/gazebo_gymnasium_resources/models/mujoco_sources/inverted_pendulum.xml)
- Hand-port URDF: [`models/inverted_pendulum/inverted_pendulum.urdf`](../../gazebo_gymnasium_examples/gazebo_gymnasium_resources/models/inverted_pendulum/inverted_pendulum.urdf)
- Generated SDF: [`models/inverted_pendulum/inverted_pendulum.sdf`](../../gazebo_gymnasium_examples/gazebo_gymnasium_resources/models/inverted_pendulum/inverted_pendulum.sdf)
- World: [`worlds/inverted_pendulum.sdf`](../../gazebo_gymnasium_examples/gazebo_gymnasium_resources/worlds/inverted_pendulum.sdf)
- Sync-gate plugin: [`plugins/inverted_pendulum_learner.py`](../../gazebo_gymnasium_examples/gazebo_gymnasium_resources/plugins/inverted_pendulum_learner.py)
- Env class: [`bridge/envs/inverted_pendulum.py`](../../gazebo_gymnasium_bridge/gazebo_gymnasium_bridge/envs/inverted_pendulum.py)
- Training script: [`train_inverted_pendulum_sac.py`](../../training_scripts/train_inverted_pendulum_sac.py)
- Launch: [`inverted_pendulum.launch.py`](../../gazebo_gymnasium_examples/gazebo_gymnasium_bringup/launch/inverted_pendulum.launch.py)
