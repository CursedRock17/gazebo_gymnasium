# Gazebo Gymnasium
------------------------
This repository reserves as integration between OpenAI's [Gymnasium](https://gymnasium.farama.org/)
package along with ROS 2 & [Gazebo](https://gazebosim.org/docs/latest/getstarted/).
It allows user to engage reinforcement learning through simulation, then port that data to real 
life, allowing for an easier integration process of a robot model.

**Standards & policies** (see `docs/ros2_reps_compliance.md` for the full mapping):

- [Quality Declaration](QUALITY_DECLARATION.md) — [REP-2004](https://ros.org/reps/rep-2004.html) Level 4 (Demos / Tutorials / Experiments)
- [Security Policy](SECURITY.md) — [REP-2006](https://ros.org/reps/rep-2006.html) vulnerability disclosure
- [Contributing Guide](CONTRIBUTING.md) — code style, dep conventions, lint workflow
- [ROS 2 REPs Compliance](docs/ros2_reps_compliance.md) — REP-by-REP status table
- Target platform: **Ubuntu Noble 24.04 + ROS 2 Jazzy + Gazebo Harmonic** (Tier 1 per [REP-2000](https://ros.org/reps/rep-2000.html))

## Installation
Tested on Ubuntu 24.04 (Noble) with ROS 2 Jazzy and Gazebo Harmonic.

### Source

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

The training scripts (`train_cartpole_*.py`) run on the *agent* side and
need a few Python packages. We use a project-local virtualenv against
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

#### Run an example

In a **fresh terminal** (so conda / pyenv / leftover workspaces don't
shadow the apt Python 3.12 that libgz-sim8 is hard-linked to):

```bash
cd ~/gym_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source venv/bin/activate           # only needed for the training-side scripts
ros2 launch gazebo_gymnasium_bringup cartpole.launch.py
```

Pass launch args the standard way:

```bash
ros2 launch gazebo_gymnasium_bringup cartpole.launch.py headless:=true
ros2 launch gazebo_gymnasium_bringup cartpole_multi.launch.py n_agents:=10 headless:=true
ros2 launch gazebo_gymnasium_bringup line_follower.launch.py use_foxglove:=true
```

The launch files themselves set `PYTHONHOME=/usr` and prepend the
gz-sim plugin path internally, so as long as the shell is clean (no
conda env active, no other ROS workspace sourced on top of this one),
everything just works. If you hit a segfault, see Troubleshooting
below — it's almost always stale state from a previous build.

## Troubleshooting

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

### Sync-gate plugin imports fail (`ModuleNotFoundError: No module named 'cartpole_learner'`)

The launches also prepend
`<install>/gazebo_gymnasium_resources/share/gazebo_gymnasium_resources/plugins`
to `PYTHONPATH`. Same fix as above: rebuild bringup and re-source.

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

