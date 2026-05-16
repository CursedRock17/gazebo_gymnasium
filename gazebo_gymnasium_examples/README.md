# gazebo_gymnasium_examples

Example environments built on `gazebo_gymnasium`. Each subdirectory is a self-contained example with its own model, world, environment script, training script, and launch files.

## Structure

```
gazebo_gymnasium_examples/
└── <example_name>/
    ├── models/<example_name>/
    │   ├── model.config       — Gazebo model metadata
    │   └── model.sdf          — Standalone SDF model (reusable asset)
    ├── worlds/
    │   └── <example_name>.sdf — Complete Gazebo world (model inlined)
    ├── scripts/
    │   ├── <example_name>_env.py  — GazeboEnv subclass (RL-library agnostic)
    │   └── train_sb3.py           — SB3 training script (or your own)
    └── launch/
        ├── <example_name>.launch.py            — Launch Gazebo only
        └── <example_name>_train_sb3.launch.py  — Launch Gazebo + training
```

## Examples

### CartPole

Classic CartPole-v1 problem simulated in Gazebo Harmonic.

- **Observation**: `[cart_position, cart_velocity, pole_angle, pole_angular_velocity]`
- **Action**: `Discrete(2)` — 0 = push left, 1 = push right
- **Reward**: +1 per step the pole stays upright
- **Terminated**: pole angle > 12° or cart > 2.4m from center
- **Truncated**: episode exceeds 500 steps

**Gazebo topics used:**

| Direction | Topic | Message | Purpose |
|---|---|---|---|
| Subscribe | `/world/cartpole/model/cartpole/joint_state` | `gz.msgs.Model` | Read cart + pole state |
| Publish | `/model/cartpole/joint/slider_to_cart/0/cmd_pos` | `gz.msgs.Double` | Set cart target position |

**Launch:**
```bash
# Gazebo only (then run your own training script)
ros2 launch gazebo_gymnasium_examples cartpole/launch/cartpole.launch.py

# Gazebo + SB3 PPO together
ros2 launch gazebo_gymnasium_examples cartpole/launch/cartpole_train_sb3.launch.py
```

**Train with SB3:**
```bash
python3 train_sb3.py                        # train 100k steps
python3 train_sb3.py --timesteps 200000     # custom step count
python3 train_sb3.py --check-only           # validate env with random actions
```

**Train with any other library:**
```python
from cartpole_env import CartPoleEnv

env = CartPoleEnv(steps_per_action=10)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Adding a New Example

See the root [README.md](../README.md#adding-a-new-environment) for the step-by-step guide.
