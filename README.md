# Gazebo Gymnasium

A library that connects [Farama Gymnasium](https://gymnasium.farama.org/) to [Gazebo Sim](https://gazebosim.org/docs/harmonic/getstarted/), enabling reinforcement learning agents to train directly inside a physics simulation.

The RL training loop runs as a **standalone Python process** and drives Gazebo externally over `gz.transport` — no custom Gazebo plugins required. Any Gymnasium-compatible RL library (SB3, RLlib, CleanRL, custom) works out of the box.

---

## Architecture

```
┌──────────────────────────────────┐       gz.transport (ZeroMQ)
│         Gazebo Server            │ ◄─────────────────────────────┐
│  JointPositionController         │ ◄── action commands (topics)  │
│  JointStatePublisher             │ ──► state observations        │
│  UserCommands (world control)    │ ◄── step / reset (services)   │
└──────────────────────────────────┘                               │
                                                                   │
┌──────────────────────────────────┐                               │
│   Your training script           │ ──────────────────────────────┘
│   (any RL library)               │
│     env = CartPoleEnv()          │
│     model.learn(env)             │
└──────────────────────────────────┘
```

Gazebo steps physics exactly N ticks per `env.step()` call via the `WorldControl` service, then pauses — giving the training loop deterministic control over simulation time.

---

## Packages

| Package | Type | Purpose |
|---|---|---|
| `gazebo_gymnasium` | ament_python | Core library: `GazeboEnv` base class + `WorldController` |
| `gazebo_gymnasium_examples` | ament_cmake | Example environments (CartPole, ...) |

---

## Prerequisites

| Dependency | Version | Install |
|---|---|---|
| ROS 2 | Jazzy | [docs.ros.org](https://docs.ros.org/en/jazzy/Installation.html) |
| Gazebo | Harmonic | [gazebosim.org](https://gazebosim.org/docs/harmonic/install/) |
| ros_gz | jazzy branch | [github.com/gazebosim/ros_gz](https://github.com/gazebosim/ros_gz/tree/jazzy) |
| Python | ≥ 3.10 | included with ROS 2 Jazzy |

Python package dependencies — install inside the virtual environment (see below):

```bash
source ~/gym_ws/venv/bin/activate
pip install gymnasium stable-baselines3
```

---

## Virtual Environment (recommended)

Using a `virtualenv` with `--system-site-packages` gives you an isolated space for RL libraries (gymnasium, stable-baselines3, etc.) while still inheriting ROS 2 and Gazebo Python bindings that live in the system Python.

```bash
# Install virtualenv if you don't have it
pip install virtualenv

# Create the venv inside the workspace (it is gitignored)
cd ~/gym_ws
virtualenv --system-site-packages venv

# Activate it (run this in every new terminal before using the workspace)
source ~/gym_ws/venv/bin/activate

# Install RL dependencies into the venv
pip install gymnasium stable-baselines3
```

> **Every terminal session:** activate the venv first, then source ROS 2:
> ```bash
> source ~/gym_ws/venv/bin/activate
> source /opt/ros/jazzy/setup.bash
> source ~/gym_ws/install/setup.bash
> ```

The `venv/` directory lives at the workspace root and is excluded from the repository via `.gitignore`.

---

## Build

```bash
# 1. Create a colcon workspace
mkdir -p ~/gym_ws/src
cd ~/gym_ws/src
git clone <repo-url> .

# 2. Activate the virtual environment (see above) and source ROS 2
source ~/gym_ws/venv/bin/activate
source /opt/ros/jazzy/setup.bash

# 3. Install system dependencies via rosdep
sudo rosdep init        # skip if already done
rosdep update
rosdep install --from-paths . --ignore-src -r -y

# 4. Build
cd ~/gym_ws
colcon build
source install/setup.bash
```

---

## Running the CartPole Example

**Terminal 1 — launch Gazebo:**
```bash
source ~/gym_ws/venv/bin/activate
source ~/gym_ws/install/setup.bash
ros2 launch gazebo_gymnasium_examples cartpole.launch.py
```

**Terminal 2 — run SB3 training:**
```bash
source ~/gym_ws/venv/bin/activate
source ~/gym_ws/install/setup.bash
python3 ~/gym_ws/install/gazebo_gymnasium_examples/lib/gazebo_gymnasium_examples/cartpole/train_sb3.py
```

Or launch both together (activate the venv first — the launch file inherits the calling shell's Python):
```bash
source ~/gym_ws/venv/bin/activate
source ~/gym_ws/install/setup.bash
ros2 launch gazebo_gymnasium_examples cartpole_train_sb3.launch.py
```

Validate the environment with random actions before training:
```bash
source ~/gym_ws/venv/bin/activate
source ~/gym_ws/install/setup.bash
python3 ~/gym_ws/install/gazebo_gymnasium_examples/lib/gazebo_gymnasium_examples/cartpole/train_sb3.py --check-only
```

---

## Adding a New Environment

1. Create a subdirectory under `gazebo_gymnasium_examples/`:
   ```
   gazebo_gymnasium_examples/
   └── my_robot/
       ├── models/my_robot/model.config + model.sdf
       ├── worlds/my_robot.sdf
       ├── scripts/my_robot_env.py      ← subclass GazeboEnv
       ├── scripts/train_sb3.py         ← or any RL library
       └── launch/my_robot.launch.py
   ```

2. Add three install blocks to `gazebo_gymnasium_examples/CMakeLists.txt` following the CartPole pattern.

3. Implement `MyRobotEnv(GazeboEnv)` by overriding the six abstract methods:

   | Method | Purpose |
   |---|---|
   | `apply_action(action)` | Publish command to Gazebo topic |
   | `get_observation()` | Read state from Gazebo topics |
   | `get_reward(action)` | Compute step reward |
   | `is_terminated()` | True if episode ended (failure/success) |
   | `is_truncated()` | True if episode hit time limit |
   | `set_default_observation()` | Reset internal state, return initial obs |

---

## Compatibility

`GazeboEnv` is a standard `gymnasium.Env`. Any library that accepts a Gymnasium environment works:

```python
# SB3
from stable_baselines3 import PPO, SAC, TD3
model = PPO("MlpPolicy", env, verbose=1)

# RLlib
from ray.rllib.algorithms.ppo import PPOConfig
algo = PPOConfig().environment(env=MyRobotEnv).build()

# CleanRL / custom
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```
