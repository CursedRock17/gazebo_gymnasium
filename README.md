# Gazebo Gymnasium

**Train reinforcement-learning agents in [Gazebo](https://gazebosim.org/docs/latest/getstarted/),
with the standard [Gymnasium](https://gymnasium.farama.org/) API.**

Gazebo Gymnasium turns a Gazebo robot into a Gymnasium environment you can train
with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (or any
Gym-speaking library). You describe one agent as a single `AgentSpec` — its
model, observation, action, reward, and termination — and the framework runs
*N* copies of it in one Gazebo world as a vectorized environment. Because it's
built on real Gazebo physics and ROS 2, a policy you train in sim is a step
away from the same robot in the real world.

Eight environments ship ready to train (CartPole, InvertedDoublePendulum,
Hopper, Walker2d, HalfCheetah, Reacher, and a camera-based line follower), and
[adding your own](docs/creating_your_own_agent.md) is about 15 lines of Python
plus a model file.

```bash
pixi run train --agent hopper --n_agents 16    # after the two install steps below
```

**Standards & policies** (see `docs/ros2_reps_compliance.md` for the full mapping):

- [Quality Declaration](QUALITY_DECLARATION.md) — [REP-2004](https://ros.org/reps/rep-2004.html) Level 4 (Demos / Tutorials / Experiments)
- [Security Policy](SECURITY.md) — [REP-2006](https://ros.org/reps/rep-2006.html) vulnerability disclosure
- [Contributing Guide](CONTRIBUTING.md) — code style, dep conventions, lint workflow
- [ROS 2 REPs Compliance](docs/ros2_reps_compliance.md) — REP-by-REP status table
- Target platform: **Ubuntu Noble 24.04 + ROS 2 Jazzy + Gazebo Harmonic** (Tier 1 per [REP-2000](https://ros.org/reps/rep-2000.html))

## Installation
Tested on Ubuntu 24.04 (Noble) with ROS 2 Jazzy and Gazebo Harmonic.

### Pixi (recommended)

[Pixi](https://pixi.sh) installs ROS 2 Jazzy, Gazebo Harmonic (with its
`gz.sim` Python bindings) **and** the RL libraries into one locked
[conda-forge](https://conda-forge.org/) / [RoboStack](https://robostack.github.io)
environment — a **single Python for the whole stack**. That removes the
dual-Python split entirely: no `PYTHONPATH`/`LD_LIBRARY_PATH` lines, no separate
venv, no source-built Gazebo. Everything is pinned in `pixi.lock`, so
clone + `pixi install` reproduces the exact environment on any Linux machine,
CI, or robot.

**Install (one time):**

```bash
# 1. Install pixi
curl -fsSL https://pixi.sh/install.sh | bash

# 2. Clone, resolve the environment (ROS 2 + Gazebo + RL libs in one solve), build
git clone <repo-url> gazebo_gymnasium
cd gazebo_gymnasium
pixi install
pixi run build
```

**Train — one command, no launch needed.** The default backend hosts the
simulator inside the training process, so a single command trains any of the
built-in environments headlessly:

```bash
pixi run train                                 # CartPole, 4 agents (the default)
pixi run train --agent hopper --n_agents 16    # any environment, any agent count
pixi run train --agent walker2d --timesteps 400000
```

Then evaluate, sweep hyperparameters, or run the tests — every flag passes
straight through to the underlying script:

```bash
pixi run deploy --agent hopper --n_agents 4    # roll out a trained policy
pixi run sweep  --agent cartpole               # hyperparameter sweep (CSV + optional W&B)
pixi run test                                  # the full test suite (no simulator needed)
```

See the available agents any time with `pixi run train --help`. Task
definitions live in `pixi.toml`.

**Watch it live (optional).** Training is headless by default. To see an agent
in the Gazebo GUI, launch the simulator in one terminal and drive it from
another (CartPole ships a launch file today; other agents train headless):

```bash
pixi run sim                                   # terminal 1: Gazebo GUI
pixi run deploy --backend harness              # terminal 2: run the policy in it
```

### Next steps

- [**docs/examples/cartpole.md**](docs/examples/cartpole.md) — the reference
  environment, both backends, all launch/train arguments.
- [**docs/creating_your_own_agent.md**](docs/creating_your_own_agent.md) —
  add your own robot as one `AgentSpec` and train it.

### From apt + colcon (advanced)

The manual path if you'd rather use a system ROS 2 install. It uses
`scripts/env.sh` + the `Makefile` to wrap the environment; note that this
setup keeps the sim and training on **different Python builds** (see the
`server`/`client` note under Run), which Pixi avoids.

#### Prerequisites
Install the following:
- [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html)
- [Gazebo Harmonic](https://gazebosim.org/docs/harmonic/install_ubuntu/) from the OSRF apt repo
- `ros-jazzy-ros-gz` — the ROS&harr;Gazebo bridge

Once the OSRF and ROS 2 apt repos are configured, the Gazebo side is a one-liner:
```bash
sudo apt install ros-jazzy-ros-gz gz-harmonic
```

#### Create the workspace
```bash
mkdir -p ~/gym_ws/src
cd ~/gym_ws/src
git clone <repo-url> gazebo_gymnasium
```

#### Install ROS dependencies
```bash
sudo rosdep init      # first time only
rosdep update
rosdep install --from-paths gazebo_gymnasium --ignore-src -r -i -y --rosdistro jazzy
```

#### Build
```bash
cd ~/gym_ws
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash
```

#### Python environment for the training side

The training scripts (`training_scripts/train.py`, `deploy.py`) run on the
*agent* side and need a few Python packages. We use a project-local virtualenv against
Ubuntu's apt Python 3.12 so the `gz` bindings from `/usr/lib/python3/dist-packages`
remain importable. If you're a conda user, the conda env stays available for
everything else — just don't try to swap it in here, because the embedded
Python inside `libgz-sim8.so` is hard-linked to system 3.12.

```bash
# From the workspace root (~/gym_ws), alongside src/ / build/ / install/.
cd ~/gym_ws
virtualenv -p /usr/bin/python3.12 --system-site-packages ./venv
touch ./venv/COLCON_IGNORE   # keep colcon from scanning into it
source ./venv/bin/activate

pip install \
    stable_baselines3 \
    cleanrl \
    mjcf2urdf

# Sphinx for API docs (optional, only if you want to build them locally)
pip install sphinx sphinx_rtd_theme
```

Currently-pinned/known-good versions of the training-side deps:

| Package | Version | Purpose |
|---------|---------|---------|
| `stable_baselines3` | 2.7.0+ | Main RL library; PPO/SAC/etc. backends |
| `cleanrl` | 0.4.8+ | Reference single-file RL algorithm implementations (alternative to SB3) |
| `mjcf2urdf` | 0.0.1+ | Convert MuJoCo MJCF XMLs to URDF as the first step of porting Gymnasium MuJoCo envs |
| `torch` | 2.10+ | PyTorch (transitive via SB3 / CleanRL) |
| `gymnasium` | 1.2+ | Canonical Gym API |
| `sphinx`, `sphinx_rtd_theme` | latest | API docs build |

**Note for CleanRL:** `pip install cleanrl` pulls in the legacy `gym 0.26.2`
package as a transitive dep alongside `gymnasium 1.2`. They coexist as
separate Python packages without conflict — CleanRL's own scripts use the
older API. If you write new training code, prefer `import gymnasium as gym`.

#### Run

Everything goes through two terminals and the `Makefile`, which wraps the
environment setup — no more pasting `PYTHONPATH` / `LD_LIBRARY_PATH` lines, and
it strips conda/pyenv shims for you so you don't need a pristine shell:

```bash
# Terminal 1 — simulator
make sim   AGENT=cartpole N=4  HEADLESS=true

# Terminal 2 — training (single-agent is just N=1)
make train AGENT=cartpole N=16 ALGO=ppo TIMESTEPS=200000

make deploy AGENT=cartpole N=4      # evaluate a saved policy
make test                           # run the functional test suite
```

Under the hood each terminal sources `scripts/env.sh {server|client}` — the
`server` env is for the gz sim (its embedded Python needs the gz bindings),
the `client` env is the training venv. Run them directly if you prefer:

```bash
source scripts/env.sh server && \
  ros2 launch gazebo_gymnasium_bringup cartpole_multi.launch.py n_agents:=4 headless:=true
source scripts/env.sh client && \
  python3 training_scripts/train.py --agent cartpole --n_agents 16
```

If your Gazebo source workspace isn't at `~/harmonic_ws`, set
`GAZEBO_GYM_HARMONIC_WS` before sourcing (see `scripts/env.sh`). The dual
`server`/`client` split exists only because the sim and training run on
different Python builds today; collapsing that into one Python is planned.

Under Pixi none of this applies — one interpreter runs both the sim and
the training. The `server`/`client` split above is only for the apt path,
where the source-built Gazebo and the training venv are different Python
builds. (The launch files no longer pin `PYTHONHOME`; that was a
source-build-era workaround that breaks the homogeneous conda Python.)

## Troubleshooting

> Most fresh-machine issues are one of the two below. The rest of this section
> covers the manual apt path.

### Commands run, but use the wrong Gazebo/Python (system apt instead of pixi)

**Symptom:** inside `pixi run …` a launch "finishes cleanly" with no window, or
`gz`/`python` behave as if the pixi packages aren't installed. This happens on
machines that *also* have ROS 2 / Gazebo installed from apt: the shell resolves
`/usr/bin/gz` (or `/usr/bin/python3`), whose system Python can't import the
conda/pixi libraries.

**Fix:** let pixi own the whole command so its environment wins. The `pixi run`
tasks already do this (they `source install/setup.sh` inside the pixi env).
When running something by hand, wrap it the same way rather than calling `gz` /
`python` from a bare shell:

```bash
pixi run bash -c 'source install/setup.sh && which gz && gz sim --version'
```

If `which gz` points at `/usr/bin/gz` *inside* that command, your apt install is
shadowing pixi on `PATH` — start from a clean shell (no `source /opt/ros/...`
in your `.bashrc`) and use the `pixi run` tasks. A pixi-only machine never hits
this.

### `Failed to load system plugin [gz-sim-python-system-loader-system] : Could not find shared library.`

Two things this can mean:

1. **The plugin is installed but gz sim doesn't see it.** As of v0.1.0
   the launch files prepend the standard apt + source-build paths
   (`/usr/lib/x86_64-linux-gnu/gz-sim-8/plugins` and
   `/usr/local/lib/gz-sim-8/plugins`) to `GZ_SIM_SYSTEM_PLUGIN_PATH`,
   so this should "just work" out of a fresh shell. If you're seeing
   the error anyway, **rebuild bringup** (`colcon build --packages-select
   gazebo_gymnasium_bringup --symlink-install`) and `source
   install/setup.bash` again — older installs of the launch files
   didn't set this env var.

2. **You actually don't have python-system-loader installed.** Check
   for it:
   ```bash
   find /usr -name "libgz-sim-python-system-loader-system.so*" 2>/dev/null
   ```
   If nothing comes back, install `gz-harmonic` from the OSRF repo
   (instructions above). It includes the plugin.

### In-sim plugin imports fail (`ModuleNotFoundError` for `multi_agent_harness` / a controller)

The launches prepend
`<install>/gazebo_gymnasium_resources/share/gazebo_gymnasium_resources/plugins`
to `PYTHONPATH` so the gz server can import the world-level plugin. Same fix as
above: rebuild resources and re-source.

### gz sim segfaults inside `PyImport_ImportModule` / `PyUnicode_New`

The embedded libpython3.12 inside `libgz-sim8` is hard-linked to the
apt-installed Python 3.12 at `/usr`. The segfault means something else
in your shell is shadowing it. In order of frequency:

**1. Stale `install/` from a previous build with a different Python
active.** If you ever ran `colcon build` while pyenv pointed at a
non-3.12 version, the resulting `install/setup.bash` baked in paths
like `install/<pkg>/lib/python3.13/site-packages` that don't exist
now. Sourcing it adds those to `PYTHONPATH` and the embedded
interpreter crashes on import. Fix:

```bash
cd ~/gym_ws
conda deactivate 2>/dev/null || true       # if you use conda
pyenv shell system 2>/dev/null || true     # if you use pyenv
rm -rf build/ install/ log/
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash
```

**2. Another ROS workspace sourced on top of this one.** Check:

```bash
echo "$AMENT_PREFIX_PATH" | tr ':' '\n'
```

If you see entries from any workspace other than `~/gym_ws/install/*`
and `/opt/ros/jazzy`, those will expose stale `gazebo_gymnasium_*`
Python bindings whose ABI doesn't match the C++ libs being loaded.
Open a fresh terminal and source only `~/gym_ws/install/setup.bash`.

**3. Conda or pyenv active.** Conda's `libpython3.12.so` on
`LD_LIBRARY_PATH` will get loaded before the apt one. `conda
deactivate` and re-source the workspace.

To inspect any shell's relevant env in one shot, run
`scripts/diagnose_env.sh`. To diff a working terminal against a broken
one:

```bash
scripts/diagnose_env.sh > /tmp/working.env   # in the terminal that works
scripts/diagnose_env.sh > /tmp/broken.env    # in the terminal that doesn't
diff /tmp/working.env /tmp/broken.env
```

