# Gazebo Gymnasium

**Reinforcement-learning agents train in [Gazebo](https://gazebosim.org/docs/latest/getstarted/),
using the standard [Gymnasium](https://gymnasium.farama.org/) API.**

Gazebo Gymnasium turns a Gazebo robot into a Gymnasium environment,
trainable with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
or any Gym-speaking library. A single agent gets described as one
`AgentSpec`, covering its model, observation, action, reward, and
termination, and the framework runs N copies of it in one Gazebo world as
a vectorized environment. Because it builds on real Gazebo physics and
Robot Operating System 2 (ROS 2), a policy trained in simulation stays one
step away from running the same robot in the real world.

Four reinforcement-learning libraries are trained and verified against the
real engine, not just sketched:
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/),
[skrl](https://skrl.readthedocs.io/),
[rl_games](https://github.com/Denys88/rl_games), and
[rsl_rl](https://github.com/leggedrobotics/rsl_rl), the same four
[Isaac Lab](https://isaac-sim.github.io/IsaacLab/) supports.
`cartpole_continuous` gets solved through PPO (500 of 500) as the common
benchmark; see [docs/rl_libraries.md](docs/rl_libraries.md) for per-library
gotchas, supported observation and action spaces, and real reward, loss,
and timing comparison charts.

Eight environments ship ready to train, CartPole, InvertedDoublePendulum,
Hopper, Walker2d, HalfCheetah, Reacher, and a camera-based line follower
among them, and [adding your own](docs/creating_your_own_agent.md) takes
about 15 lines of Python plus a model file.

```bash
pixi run train --agent hopper --n_agents 16    # after the two install steps below
```

Trained policies push to and pull from the
[Hugging Face Hub](https://huggingface.co/) directly:

```bash
pixi run train --agent hopper --push-to-hub <user>/<repo>   # uploads model + eval-backed model card
pixi run deploy --agent hopper --from-hub <user>/<repo>      # downloads and runs it
```

**Standards and policies** (see `docs/ros2_reps_compliance.md` for the
full mapping):

- [Quality Declaration](QUALITY_DECLARATION.md), meeting
  [REP-2004](https://ros.org/reps/rep-2004.html) Level 4 (Demos, Tutorials,
  Experiments)
- [Security Policy](SECURITY.md), following
  [REP-2006](https://ros.org/reps/rep-2006.html) vulnerability disclosure
- [Contributing Guide](CONTRIBUTING.md), covering code style, dependency
  conventions, and the lint workflow
- [ROS 2 REP Compliance](docs/ros2_reps_compliance.md), a
  Robot Enhancement Proposal (REP) by REP status table
- Target platform: **Ubuntu Noble 24.04, ROS 2 Jazzy, Gazebo Harmonic**,
  Tier 1 per [REP-2000](https://ros.org/reps/rep-2000.html)

## Installation

**Platform: Linux x86-64 only.** `pixi.lock` is solved for `linux-64`, and
the underlying ROS 2 Jazzy and Gazebo Harmonic conda packages are built
for it, so `pixi install` fails to resolve on macOS or Windows. Ubuntu
24.04 (Noble) hosts development and testing directly; other x86-64 Linux
distributions should still work, since Pixi brings its own ROS and Gazebo
rather than relying on the system's. Windows needs WSL2, the Windows
Subsystem for Linux, version 2.

### Pixi Installation, Recommended

[Pixi](https://pixi.sh) installs ROS 2 Jazzy, Gazebo Harmonic (with its
`gz.sim` Python bindings), and the reinforcement-learning libraries into
one locked [conda-forge](https://conda-forge.org/) and
[RoboStack](https://robostack.github.io) environment, a single Python for
the whole stack. That removes the dual-Python split entirely: no
`PYTHONPATH` or `LD_LIBRARY_PATH` lines, no separate virtual environment,
no source-built Gazebo. Everything gets pinned in `pixi.lock`, so cloning
the repository and running `pixi install` reproduces the exact environment
on any Linux machine, continuous-integration runner, or robot.

**Installing, one time:**

```bash
# 1. Install pixi.
curl -fsSL https://pixi.sh/install.sh | bash

# Open a NEW terminal now (or run `exec $SHELL`) so the `pixi` command
# lands on PATH. The installer adds it to the shell profile, but the
# current shell will not see it until it reloads.

# 2. Clone the repository, resolve the environment (ROS 2, Gazebo, and the
#    RL libraries in one solve), and build.
git clone https://github.com/CursedRock17/gazebo_gymnasium.git gazebo_gymnasium
cd gazebo_gymnasium
pixi install
pixi run build
```

> **The first `pixi install` is a big, one-time download.** It fetches the
> entire ROS 2, Gazebo, and PyTorch stack (several gigabytes) and commonly
> takes 15 to 30 minutes on a fresh machine; after that it stays cached and
> near-instant. If a single package download stalls on a slow mirror
> (LLVM is a common culprit), pressing `Ctrl+C` and re-running `pixi
> install` resumes from the cached completed packages rather than starting
> over.

### Seeing It Run, CartPole In The Gazebo GUI

The quickest way to see what the library does is training the reference
CartPole, then watching the trained policy balance the poles live in
Gazebo, in three short steps.

```bash
# 1. Train CartPole (headless, roughly 1 to 2 minutes; it solves quickly).
pixi run train

# 2. Open the Gazebo GUI with 4 cartpoles.
pixi run sim

# 3. In a SECOND terminal, run the trained policy in that GUI.
pixi run deploy --backend harness
```

Four carts should nudge themselves left and right to keep their poles
upright, the same policy just trained, now driving the live simulator.

<!-- Screenshot placeholder. Once docs/images/cartpole_gui.png is captured
     (see docs/images/README.md), uncomment the next line to show it here.
     It stays commented out so the public README does not render a broken
     image.
![CartPole balancing in the Gazebo GUI](docs/images/cartpole_gui.png)
-->

> The GUI needs a display and a working graphics driver. On a headless
> server, skipping this and training or evaluating with the headless
> commands below works just as well.

**Training needs one command and no launch.** The default backend hosts
the simulator inside the training process, so a single command trains any
of the built-in environments headlessly.

```bash
pixi run train                                 # CartPole, 4 agents (the default)
pixi run train --agent hopper --n_agents 16    # any environment, any agent count
pixi run train --agent walker2d --timesteps 400000
```

Evaluating, sweeping hyperparameters, and running the tests all flow
straight through the same pass-through flags.

```bash
pixi run deploy --agent hopper --n_agents 4    # roll out a trained policy
pixi run sweep  --agent cartpole               # hyperparameter sweep (CSV + optional W&B)
pixi run benchmark --agent hopper --scale 1,4,16,32   # throughput / scaling
pixi run test                                  # the full test suite (no simulator needed)
```

`pixi run train --help` lists the available agents any time. Task
definitions live in `pixi.toml`. Only CartPole ships a GUI launch file
today; every other environment trains and evaluates headless.

### Next Steps

[**docs/examples/cartpole.md**](docs/examples/cartpole.md) is the
reference environment: both backends, and every launch and training
argument, worth reading next. [**docs/README.md**](docs/README.md) is
the full documentation index from there, in reading order rather than
alphabetical: training and reviewing results, building your own
environment, a page per environment, and framework internals.
[**CHANGELOG.md**](CHANGELOG.md) holds release notes and known
limitations.

### The apt and colcon Path, Advanced

A manual path exists for using a system ROS 2 install instead. It uses
`scripts/env.sh` and the `Makefile` to wrap the environment; note that
this setup keeps the simulation and training on different Python builds
(see the server and client note under Running It below), which Pixi
avoids entirely.

**Prerequisites** include the following:

- [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html)
- [Gazebo Harmonic](https://gazebosim.org/docs/harmonic/install_ubuntu/)
  from the OSRF apt repository
- `ros-jazzy-ros-gz`, the ROS-to-Gazebo bridge

Once the OSRF and ROS 2 apt repositories are configured, the Gazebo side
reduces to one line.

```bash
sudo apt install ros-jazzy-ros-gz gz-harmonic
```

**Creating the workspace:**

```bash
mkdir -p ~/gym_ws/src
cd ~/gym_ws/src
git clone https://github.com/CursedRock17/gazebo_gymnasium.git gazebo_gymnasium
```

**Installing ROS dependencies:**

```bash
sudo rosdep init      # first time only
rosdep update
rosdep install --from-paths gazebo_gymnasium --ignore-src -r -i -y --rosdistro jazzy
```

**Building:**

```bash
cd ~/gym_ws
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash
```

**Setting up the Python environment for training.** The training scripts,
`training_scripts/train.py` and `deploy.py`, run on the agent side and
need a few Python packages. A project-local virtual environment against
Ubuntu's apt Python 3.12 keeps the `gz` bindings from
`/usr/lib/python3/dist-packages` importable. A conda user's existing conda
environment stays available for everything else, but swapping it in here
does not work, since the embedded Python inside `libgz-sim8.so` is
hard-linked to system 3.12.

```bash
# From the workspace root (~/gym_ws), alongside src/, build/, and install/.
cd ~/gym_ws
virtualenv -p /usr/bin/python3.12 --system-site-packages ./venv
touch ./venv/COLCON_IGNORE   # keeps colcon from scanning into it
source ./venv/bin/activate

pip install \
    stable_baselines3 \
    cleanrl \
    mjcf2urdf

# Sphinx builds the API docs, optional, only needed to build them locally.
pip install sphinx sphinx_rtd_theme
```

The currently pinned, known-good versions of the training-side
dependencies appear below.

| Package | Version | Purpose |
|---------|---------|---------|
| `stable_baselines3` | 2.7.0+ | Main reinforcement-learning library; PPO, SAC, and other backends |
| `cleanrl` | 0.4.8+ | Reference single-file algorithm implementations, an alternative to SB3 |
| `mjcf2urdf` | 0.0.1+ | Converts MuJoCo MJCF XML files to URDF, the first step of porting Gymnasium MuJoCo environments |
| `torch` | 2.10+ | PyTorch, transitive through SB3 and CleanRL |
| `gymnasium` | 1.2+ | The canonical Gym API |
| `sphinx`, `sphinx_rtd_theme` | latest | Building the API documentation |

**A note for CleanRL:** `pip install cleanrl` pulls in the legacy `gym
0.26.2` package as a transitive dependency alongside `gymnasium 1.2`. They
coexist as separate Python packages without conflict, since CleanRL's own
scripts use the older API. New training code should prefer `import
gymnasium as gym`.

**Running it** goes through two terminals and the `Makefile`, which wraps
the environment setup, so no `PYTHONPATH` or `LD_LIBRARY_PATH` lines need
pasting, and it strips conda or pyenv shims automatically, so a pristine
shell is not required.

```bash
# Terminal 1 runs the simulator.
make sim   AGENT=cartpole N=4  HEADLESS=true

# Terminal 2 runs training (single-agent is just N=1).
make train AGENT=cartpole N=16 ALGO=ppo TIMESTEPS=200000

make deploy AGENT=cartpole N=4      # evaluates a saved policy
make test                           # runs the functional test suite
```

Under the hood, each terminal sources `scripts/env.sh {server|client}`:
the server environment serves gz sim, since its embedded Python needs the
gz bindings, while the client environment serves the training virtual
environment. Running them directly also works.

```bash
source scripts/env.sh server && \
  ros2 launch gazebo_gymnasium_bringup cartpole_multi.launch.py n_agents:=4 headless:=true
source scripts/env.sh client && \
  python3 training_scripts/train.py --agent cartpole --n_agents 16
```

Setting `GAZEBO_GYM_HARMONIC_WS` before sourcing matters if the Gazebo
source workspace is not at `~/harmonic_ws` (see `scripts/env.sh`). The
dual server and client split exists only because the simulator and
training run on different Python builds today; collapsing that into one
Python is planned.

Under Pixi, none of this applies, since one interpreter runs both the
simulator and the training. The server and client split above only
applies to the apt path, where the source-built Gazebo and the training
virtual environment are different Python builds. (The launch files no
longer pin `PYTHONHOME`; that was a source-build-era workaround that
breaks the homogeneous conda Python.)

## Troubleshooting

> Most fresh-machine issues reduce to one of the two below. The rest of
> this section covers the manual apt path.

### Commands Run But Use The Wrong Gazebo Or Python

**Symptom.** Inside `pixi run …`, a launch finishes cleanly with no
window, or `gz` and `python` behave as if the pixi packages are not
installed. This happens on machines that also have ROS 2 or Gazebo
installed from apt: the shell resolves `/usr/bin/gz` or
`/usr/bin/python3`, whose system Python cannot import the conda or pixi
libraries.

**Fix.** Letting pixi own the whole command lets its environment win. The
`pixi run` tasks already do this, since they source `install/setup.sh`
inside the pixi environment. Running something by hand should wrap it the
same way, rather than calling `gz` or `python` from a bare shell.

```bash
pixi run bash -c 'source install/setup.sh && which gz && gz sim --version'
```

If `which gz` points at `/usr/bin/gz` inside that command, the apt install
is shadowing pixi on PATH; starting from a clean shell, with no `source
/opt/ros/...` in `.bashrc`, and using the `pixi run` tasks resolves it. A
pixi-only machine never hits this.

### A System Plugin Fails To Load

The error reads `Failed to load system plugin
[gz-sim-python-system-loader-system]: Could not find shared library`, and
it can mean either of two things.

1. **The plugin is installed, but gz sim does not see it.** As of v0.1.0,
   the launch files prepend the standard apt and source-build paths
   (`/usr/lib/x86_64-linux-gnu/gz-sim-8/plugins` and
   `/usr/local/lib/gz-sim-8/plugins`) to `GZ_SIM_SYSTEM_PLUGIN_PATH`, so
   this should just work out of a fresh shell. Seeing the error anyway
   calls for rebuilding bringup (`colcon build --packages-select
   gazebo_gymnasium_bringup --symlink-install`) and sourcing
   `install/setup.bash` again, since older installs of the launch files
   did not set this environment variable.
2. **The python-system-loader plugin is simply not installed.** Checking
   for it:
   ```bash
   find /usr -name "libgz-sim-python-system-loader-system.so*" 2>/dev/null
   ```
   Nothing coming back means installing `gz-harmonic` from the OSRF
   repository (instructions above) resolves it, since that package
   includes the plugin.

### In-Sim Plugin Imports Fail

A `ModuleNotFoundError` for `multi_agent_harness` or a controller shows
this. The launches prepend
`<install>/gazebo_gymnasium_resources/share/gazebo_gymnasium_resources/plugins`
to `PYTHONPATH` so the gz server can import the world-level plugin. The
same fix as above, rebuilding resources and re-sourcing, resolves it.

### gz Sim Segfaults During Python Imports

A segfault inside `PyImport_ImportModule` or `PyUnicode_New` means the
embedded libpython3.12 inside `libgz-sim8`, hard-linked to the
apt-installed Python 3.12 at `/usr`, is getting shadowed by something else
in the shell. In order of frequency, three causes explain most cases.

**A stale `install/` directory from a previous build with a different
Python active** is the most common. Running `colcon build` while pyenv
pointed at a non-3.12 version bakes paths like
`install/<pkg>/lib/python3.13/site-packages` into the resulting
`install/setup.bash`, and those paths do not exist now. Sourcing it adds
them to `PYTHONPATH`, and the embedded interpreter crashes on import.

```bash
cd ~/gym_ws
conda deactivate 2>/dev/null || true       # if conda is in use
pyenv shell system 2>/dev/null || true     # if pyenv is in use
rm -rf build/ install/ log/
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash
```

**Another ROS workspace sourced on top of this one** is the second cause.
Checking:

```bash
echo "$AMENT_PREFIX_PATH" | tr ':' '\n'
```

Entries from any workspace other than `~/gym_ws/install/*` and
`/opt/ros/jazzy` expose stale `gazebo_gymnasium_*` Python bindings whose
Application Binary Interface (ABI) does not match the C++ libraries being
loaded. Opening a fresh terminal and sourcing only
`~/gym_ws/install/setup.bash` resolves it.

**Conda or pyenv staying active** is the third cause. Conda's
`libpython3.12.so` on `LD_LIBRARY_PATH` loads before the apt one;
`conda deactivate` followed by re-sourcing the workspace resolves it.

Inspecting any shell's relevant environment in one shot works through
`scripts/diagnose_env.sh`. Diffing a working terminal against a broken one
follows the same pattern.

```bash
scripts/diagnose_env.sh > /tmp/working.env   # in the terminal that works
scripts/diagnose_env.sh > /tmp/broken.env    # in the terminal that doesn't
diff /tmp/working.env /tmp/broken.env
```

## Glossary

| Term | Definition |
|---|---|
| **Robot Operating System 2 (ROS 2)** | The middleware Gazebo Gymnasium integrates with natively |
| **Robot Enhancement Proposal (REP)** | A ROS 2 ecosystem standards document, similar in spirit to Python's PEPs |
| **Windows Subsystem for Linux 2 (WSL2)** | The compatibility layer Windows users need, since this project is Linux-only |
| **Application Binary Interface (ABI)** | The compiled-code contract between Python bindings and their underlying C++ libraries |
| **AgentSpec** | The dataclass describing one agent's model, observation, action, reward, and DR configuration |
