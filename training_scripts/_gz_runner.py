# Copyright 2026 Lucas Wendland
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared utilities for parallel + headless RL training runners.

Each training script that supports `--n_envs N` uses these helpers to:

  1. **Spawn a headless `gz sim -s` subprocess per env**, isolated from
     siblings via a unique `GZ_PARTITION`. Partitions sandbox gz-transport
     topics so N parallel envs can each publish on `/env/action` without
     colliding with each other.

  2. **Construct an env class** in the same Python process *after* the
     `GZ_PARTITION` env var is set — so the env's gz-transport nodes
     join the right partition.

  3. **Tear gz sim down cleanly** when the env closes (SubprocVecEnv
     terminates workers; the worker's atexit fires the gz sim kill).

Usage:

    from _gz_runner import make_headless_env_fn

    env_fn = make_headless_env_fn(
        env_cls=GazeboCartPoleEnv,
        env_kwargs={"world_name": "cartpole"},
        world_sdf_filename="cartpole.sdf",
        partition_id=0,
    )

    env = env_fn()             # single env path
    # or:
    vec_env = SubprocVecEnv([
        make_headless_env_fn(..., partition_id=i) for i in range(N)
    ])

`SubprocVecEnv` forks a Python subprocess per env_fn; each subprocess runs
its own `make_headless_env_fn(...)` which spawns gz sim then constructs
the env. Total processes: N gz sim + N Python workers + 1 trainer.
Recommended N <= cores - 2 to leave room for the trainer + OS.
"""

import atexit
import os
import signal
import subprocess
import time
from typing import Callable
from typing import Type


def _find_world_sdf(world_sdf_filename: str) -> str:
    """Resolve the installed share/worlds/<filename> path.

    Requires ROS 2 + the colcon workspace to be sourced.
    """
    from ament_index_python.packages import get_package_share_directory
    share_dir = get_package_share_directory("gazebo_gymnasium_resources")
    return os.path.join(share_dir, "worlds", world_sdf_filename)


def _spawn_gz_sim(world_path: str, partition: str,
                  startup_wait: float = 3.0) -> subprocess.Popen:
    """Start `gz sim -s -r <world>` with a custom GZ_PARTITION.

    server-only (`-s`) keeps it headless; `-r` starts the simulation
    immediately (matches what our launch files do). startup_wait gives gz
    sim time to load the world and advertise topics before the agent-side
    env starts subscribing.
    """
    gz_env = {
        **os.environ,
        "GZ_PARTITION": partition,
        # libgz-sim8 embeds Python 3.12; match the launch files' pin so
        # conda/pyenv on PATH doesn't redirect away from /usr/lib.
        "PYTHONHOME": "/usr",
    }
    proc = subprocess.Popen(
        ["gz", "sim", "-s", "-r", world_path],
        env=gz_env,
        preexec_fn=os.setsid,  # so we can SIGTERM the whole process group
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(startup_wait)
    return proc


def _terminate_gz_sim(proc: subprocess.Popen) -> None:
    """SIGTERM the gz sim process group; fall back to SIGKILL if it hangs."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            proc.kill()
        except Exception:  # noqa: BLE001 — process already gone, ignore
            pass


def make_headless_env_fn(
    env_cls: Type,
    env_kwargs: dict,
    world_sdf_filename: str,
    partition_id: int,
    startup_wait: float = 3.0,
) -> Callable:
    """Return a `() -> env` factory suitable for SubprocVecEnv or direct use.

    Each call to the returned factory:
      1. Sets `GZ_PARTITION` for the current process,
      2. Spawns a headless gz sim subprocess for the same partition,
      3. Constructs `env_cls(**env_kwargs)` (which picks up the partition
         from the env var when it builds its gz-transport nodes),
      4. Registers an atexit handler to kill the gz sim subprocess.

    The factory is intentionally a closure rather than a class — SB3's
    SubprocVecEnv pickles env_fns to send them across to worker processes,
    and closures over simple values pickle fine.
    """
    world_path = _find_world_sdf(world_sdf_filename)

    def _factory():
        partition = f"gym_env{partition_id}"
        os.environ["GZ_PARTITION"] = partition
        proc = _spawn_gz_sim(world_path, partition, startup_wait=startup_wait)
        atexit.register(_terminate_gz_sim, proc)
        env = env_cls(**env_kwargs)
        # Stash the gz proc on the env so callers can inspect / explicitly
        # close from outside if they want.
        env._gz_proc = proc  # type: ignore[attr-defined]
        return env

    return _factory
