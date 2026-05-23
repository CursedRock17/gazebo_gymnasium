# Gazebo Gymnasium

A library that connects [Farama Gymnasium](https://gymnasium.farama.org/) to [Gazebo Sim](https://gazebosim.org/docs/harmonic/getstarted/), enabling reinforcement learning agents to train directly inside a physics simulation.

Two base classes are provided depending on your speed and setup requirements:

| Class | Approach | Speed | Requirements |
|---|---|---|---|
| `GazeboEnv` | External Gazebo process driven over `gz.transport` (DDS) | Good | ROS 2 + Gazebo + active Gazebo process |
| `FixtureEnv` | Gazebo **embedded in the training process** via `gz.sim8.TestFixture` | **Fastest** | Gazebo Harmonic only (no ROS 2, no separate process) |

Any Gymnasium-compatible RL library (SB3, RLlib, CleanRL, custom) works with either base class.

---

## Architecture

### GazeboEnv (IPC mode)

```
┌──────────────────────────────────┐       gz.transport (ZeroMQ/DDS)
│         Gazebo Server            │ ◄─────────────────────────────┐
│  JointPositionController         │ ◄── action commands (topics)  │
│  JointStatePublisher             │ ──► state observations        │
│  UserCommands (world control)    │ ◄── step / reset (services)   │
└──────────────────────────────────┘                               │
                                                                   │
┌──────────────────────────────────┐                               │
│   Your training script           │ ──────────────────────────────┘
│     env = CartPoleEnv()          │
│     model.learn(env)             │
└──────────────────────────────────┘
```

Physics steps N ticks per `env.step()` via the `WorldControl` service.  Requires launching a separate Gazebo process and sourcing ROS 2.

### FixtureEnv (in-process mode)

```
┌──────────────────────────────────────────────────────┐
│               Your training script                   │
│                                                      │
│   env = CartPoleFixtureEnv(sdf_path)                 │
│   # gz.sim8.TestFixture is embedded here             │
│   model.learn(env)   ──► env.step()                  │
│                              │                       │
│                    server.run(True, N, False)         │
│                              │                       │
│                   pre_update  ── apply force (ECM)   │
│                   post_update ── read state (ECM)    │
└──────────────────────────────────────────────────────┘
```

The training process IS the Gazebo server.  Zero IPC, zero DDS, zero threads.  `server.run(True, N, False)` blocks for exactly N physics steps — the fastest possible simulation loop.

---

## Packages

| Package | Type | Purpose |
|---|---|---|
| `gazebo_gymnasium` | ament_python | Core library: `GazeboEnv`, `FixtureEnv`, `WorldController`, `PPO` |
| `gazebo_gymnasium_examples` | ament_cmake | Example environments (CartPole) |

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

**Terminal 2 — run SB3 training (three equivalent ways):**

```bash
source ~/gym_ws/venv/bin/activate
source ~/gym_ws/install/setup.bash

# Option A: ros2 run (recommended after colcon build)
ros2 run gazebo_gymnasium_examples cartpole_train_sb3

# Option B: run directly from the source tree (no build needed)
python3 ~/gym_ws/src/gazebo_gymnasium/gazebo_gymnasium_examples/cartpole/scripts/train_sb3.py

# Option C: full install path (explicit)
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
ros2 run gazebo_gymnasium_examples cartpole_train_sb3 --check-only
```

View TensorBoard training curves (logs written to `./tb_logs` by default):
```bash
tensorboard --logdir ./tb_logs
```

---

## Running the CartPole Example — Fixture Mode (Fastest, No ROS 2)

`FixtureEnv` embeds Gazebo directly in the training process.  No separate Gazebo process, no ROS 2 sourcing, no `gz.transport` — just Python and Gazebo Harmonic.

**Requirements (fixture mode only):**
- Gazebo Harmonic installed (`sudo apt install gz-harmonic`)
- `gz.sim8` Python bindings on the path (included with Gazebo Harmonic at `/usr/local/lib/python`)
- Python 3.12 with `gymnasium` and `stable-baselines3` (or the built-in `gazebo_gymnasium.ppo`)

```bash
# Activate the venv (no ROS 2 source needed)
source ~/gym_ws/venv/bin/activate
source ~/gym_ws/install/setup.bash

# Run directly (SDF path auto-resolved relative to the script)
python3 gazebo_gymnasium_examples/cartpole/scripts/train_fixture.py

# Explicit options
python3 train_fixture.py \
    --sdf gazebo_gymnasium_examples/cartpole/worlds/cartpole_fixture.sdf \
    --timesteps 200000 \
    --steps-per-action 5 \
    --device cuda
```

The script tries SB3 PPO first; if SB3 is not installed, it falls back to the built-in `gazebo_gymnasium.ppo` (PyTorch only).

> **Note:** `LD_LIBRARY_PATH` must include `/usr/local/lib` for gz.sim8 shared libraries.
> The `train_fixture.py` script sets this automatically if not already set.

---

## Adding a New Environment

### Option A — GazeboEnv (IPC, requires running Gazebo process)

1. Create a subdirectory under `gazebo_gymnasium_examples/`:
   ```
   gazebo_gymnasium_examples/
   └── my_robot/
       ├── worlds/my_robot.sdf             ← include JointPositionController + JointStatePublisher
       ├── scripts/my_robot_env.py         ← subclass GazeboEnv
       ├── scripts/train_sb3.py            ← or any RL library
       └── launch/my_robot.launch.py
   ```

2. Add install blocks to `gazebo_gymnasium_examples/CMakeLists.txt` following the CartPole pattern.

3. Subclass `GazeboEnv` and implement:

   | Method | Purpose |
   |---|---|
   | `apply_action(action)` | Publish command to Gazebo topic |
   | `get_observation()` | Read state from Gazebo topics |
   | `get_reward(action)` | Compute step reward |
   | `is_terminated()` | True if episode ended (failure/success) |
   | `is_truncated()` | True if episode hit time limit |
   | `set_default_observation()` | Reset internal state, return initial obs |

### Option B — FixtureEnv (in-process, fastest, no ROS 2 needed)

1. Create a stripped SDF (no JointPositionController, no JointStatePublisher, only Physics plugin):
   ```
   gazebo_gymnasium_examples/
   └── my_robot/
       ├── worlds/my_robot_fixture.sdf     ← Physics plugin only
       └── scripts/my_robot_fixture_env.py ← subclass FixtureEnv
   ```

2. Subclass `FixtureEnv` and implement:

   | Method | Purpose |
   |---|---|
   | `configure(ecm)` | Look up joint entities; call `enable_position_check(ecm, True)` |
   | `apply_action_to_ecm(ecm, action)` | Call `joint.set_force(ecm, [f])` |
   | `apply_reset(ecm)` | Call `joint.reset_position(ecm, [0])` + `reset_velocity` |
   | `read_observation(ecm)` | Call `joint.position(ecm)` + `velocity(ecm)` → float32 array |
   | `get_reward(action)` | Compute step reward |
   | `is_terminated()` | True if episode ended |
   | `is_truncated()` | True if episode hit time limit |
   | `set_default_observation()` | Zero internal state, return initial obs |

   See `cartpole_fixture_env.py` for a complete reference implementation.

---

## PyPI Distribution

`gazebo_gymnasium` can be published to PyPI so users can `pip install gazebo-gymnasium`. Because the package is also an ament_python package, `pyproject.toml` cannot live inside the ament package directory (it conflicts with colcon's setup.py introspection). The publishing workflow uses a thin wrapper `pyproject.toml` placed at the **repo root** that points at the package source:

```toml
# pyproject.toml (repo root — not checked in; create when ready to publish)
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "gazebo-gymnasium"
version = "0.1.0"
description = "Gymnasium interface for Gazebo Sim (Harmonic)"
readme = "gazebo_gymnasium/README.md"
license = "Apache-2.0"
authors = [{name = "Lucas Wendland", email = "mtglucas1@gmail.com"}]
requires-python = ">=3.10"
keywords = ["robotics", "reinforcement-learning", "gazebo", "gymnasium", "ros2"]
dependencies = ["gymnasium>=0.29.0", "numpy>=1.21.0"]

[project.optional-dependencies]
sb3 = ["stable-baselines3>=2.0.0"]

[tool.setuptools.packages.find]
where = ["gazebo_gymnasium"]
include = ["gazebo_gymnasium*"]
```

> **Note:** `gz.transport13` and `gz.msgs10` are Gazebo system packages and are not available on PyPI. Users must install Gazebo Harmonic separately (`sudo apt install gz-harmonic`).

Publish:
```bash
pip install build twine
python -m build
twine upload dist/*
```

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
