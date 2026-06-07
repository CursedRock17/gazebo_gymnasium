# InvertedDoublePendulum

Two-link version of InvertedPendulum: a cart slides along a rail and balances
**a pole stacked on top of another pole**. The agent applies a continuous
force to the cart. Much harder than the single pendulum — the lower link
amplifies any error and you have to balance the upper link relative to the
lower one.

Model ported from Gymnasium's
[`InvertedDoublePendulum-v5`](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/)
MuJoCo MJCF via the batch
[`scripts/convert_mujoco_models.py`](../../scripts/convert_mujoco_models.py).
Post-conversion the SDF's phantom links were renamed to the conventional
`rail / cart / pole / pole2` names and the actuator + joint-state plugins
were attached by hand.

## What you get

- The two-link pendulum model
- A `gz-sim-apply-joint-force-system` plugin on the slider
- A `gz-sim-joint-state-publisher-system` for `slider`, `hinge`, `hinge2`
- Sync-gate plugin (`inverted_double_pendulum_learner.py`) that publishes a
  6-tuple of raw values and accepts a continuous action
- `GazeboInvertedDoublePendulumEnv` — `gym.Env` with `Box(1,)` action and
  `Box(8,)` observation (sin/cos expansion done in the env, not the plugin)
- A reference A2C training script (`train_inverted_double_pendulum_a2c.py`)

## Running it

```bash
# Terminal 1 — simulator
ros2 launch gazebo_gymnasium_bringup inverted_double_pendulum.launch.py

# Terminal 2 — A2C training
python training_scripts/train_inverted_double_pendulum_a2c.py
```

## The algorithm: A2C

**Advantage Actor-Critic** is the synchronous, on-policy baseline. We use it
here for algorithm variety — CartPole already has PPO, InvertedPendulum has
SAC, so A2C completes the on-policy / off-policy / sync-AC trio.

Differences from PPO (which we use on CartPole):

- **No clipping:** A2C uses the plain policy-gradient update. Simpler, but
  more prone to destructive large updates if the learning rate is too high.
- **Single rollout, single update:** PPO does multiple epochs over each
  rollout; A2C does one. This is why A2C tends to need smaller `n_steps` and
  more total interactions.
- **No KL constraint or trust region:** the only knob preventing big jumps
  is `max_grad_norm`.

Hyperparameters in `train_inverted_double_pendulum_a2c.py`:

| Param | Value | Notes |
|-------|-------|-------|
| `learning_rate` | 7e-4 | SB3 A2C default |
| `n_steps` | 16 | Rollout length per update |
| `gamma` | 0.99 | Discount |
| `gae_lambda` | 1.0 | Plain n-step returns (no GAE smoothing) |
| `ent_coef` | 0.01 | Entropy regularization |
| `vf_coef` | 0.5 | Value-function loss weight |
| `max_grad_norm` | 0.5 | Gradient clipping |

A2C is sample-inefficient — expect to need **hundreds of thousands of
steps** before stable behaviour. SAC would converge faster on this task; we
use A2C here for didactic comparison.

## Observation space

`Box(8,) float32`. Canonical Gymnasium InvertedDoublePendulum-v5 has 11 dims
(the last three being constraint forces). We drop the constraint forces
because Gazebo doesn't expose them as cleanly as MuJoCo does, leaving 8.

| Index | Field | Source |
|-------|-------|--------|
| 0 | Cart position (X) | `slider` joint position |
| 1 | sin(pole1 angle) | `hinge` joint position, sin |
| 2 | sin(pole2 angle, relative to pole1) | `hinge2` joint position, sin |
| 3 | cos(pole1 angle) | `hinge` joint position, cos |
| 4 | cos(pole2 angle) | `hinge2` joint position, cos |
| 5 | Cart velocity | `slider` joint velocity |
| 6 | Pole1 angular velocity | `hinge` joint velocity |
| 7 | Pole2 angular velocity | `hinge2` joint velocity |

The sync-gate plugin publishes the raw 6-tuple `[cart_pos, pole1_ang,
pole2_ang, cart_vel, pole1_vel, pole2_vel]`; the env expands sin/cos on the
agent side. This matches canonical Gymnasium conventions and avoids the
discontinuity at ±π.

## Action space

`Box(low=-1, high=1, shape=(1,), dtype=float32)`. Multiplied by **gear=500**
in the sync-gate plugin (matching MJCF `motor gear="500"`) before being
published to `/model/inverted_double_pendulum/joint/slider/cmd_force`.
Effective force on the slider: ±500 N.

## Reward function

`+10.0` per step alive (canonical `alive_bonus=10`). The canonical reward
also includes position/velocity penalty terms based on the tip site's Y
coordinate; we drop those because the tip site marker doesn't translate
directly to Gazebo. With alive-bonus only, training still converges, just
more slowly than the canonical reward shaping would allow.

- `terminated = True` when EITHER pole's joint angle exceeds 0.4 rad in
  magnitude, OR any observation is non-finite
- `truncated = True` when `current_step >= max_episode_steps` (default 1000)

## What's different from the canonical port

- **Dropped 3 constraint-force dims** in observation (Gazebo doesn't expose
  them cleanly).
- **Dropped tip-site reward shaping** — only the alive bonus survives.
- **Threshold-based termination** instead of tip-height termination
  (`|angle| > 0.4 rad` per pole).
- **No reset noise** (canonical adds uniform ±0.1 to qpos and qvel).

## Reference files

- Source MJCF:
  [`mujoco_sources/inverted_double_pendulum.xml`](../../gazebo_gymnasium_examples/gazebo_gymnasium_resources/models/mujoco_sources/inverted_double_pendulum.xml)
- Auto-converted SDF (with hand-renamed links):
  [`models/inverted_double_pendulum/inverted_double_pendulum.sdf`](../../gazebo_gymnasium_examples/gazebo_gymnasium_resources/models/inverted_double_pendulum/inverted_double_pendulum.sdf)
- World:
  [`worlds/inverted_double_pendulum.sdf`](../../gazebo_gymnasium_examples/gazebo_gymnasium_resources/worlds/inverted_double_pendulum.sdf)
- Sync-gate plugin:
  [`plugins/inverted_double_pendulum_learner.py`](../../gazebo_gymnasium_examples/gazebo_gymnasium_resources/plugins/inverted_double_pendulum_learner.py)
- Env class:
  [`bridge/envs/inverted_double_pendulum.py`](../../gazebo_gymnasium_bridge/gazebo_gymnasium_bridge/envs/inverted_double_pendulum.py)
- Training script:
  [`train_inverted_double_pendulum_a2c.py`](../../training_scripts/train_inverted_double_pendulum_a2c.py)
- Launch:
  [`inverted_double_pendulum.launch.py`](../../gazebo_gymnasium_examples/gazebo_gymnasium_bringup/launch/inverted_double_pendulum.launch.py)
