#!/usr/bin/env python3
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
"""Spawn N line-follower rovers (+ their tracks) into the running `line_follower_harness` world.

Called by `line_follower_harness.launch.py` after gz sim is up. Unlike
spawn_multi_cartpoles.py, this needs to rewrite each rover's camera <topic>
to a per-agent one (/rl/camera_i) before spawning -- <include merge="true">
can't reach into a nested sensor element, so the whole model.sdf is read and
string-substituted, mirroring what inprocess_vec_env.py's world builder does
for the in-process backend. Agents are laid out along X at spec.spawn_y (a
line, not a grid -- each gets its own track loop, not shared floor space),
matching agent_spec._line_follower_spec()'s spawn_y/spawn_yaw/x_spacing.

Usage:
    python3 spawn_multi_line_followers.py --n-agents 4
"""

import argparse
import subprocess
import sys
import time

from ament_index_python.packages import get_package_share_directory

AGENT_NAME = "line_follower"
WORLD_NAME = "line_follower_harness"
X_SPACING = 6.0  # matches AgentSpec.x_spacing -- one ~2.4 m track per agent
SPAWN_Y = -1.0  # matches AgentSpec.spawn_y -- the known-good start pose
SPAWN_YAW = 1.5708  # matches AgentSpec.spawn_yaw
SPAWN_Z = 0.085  # matches AgentSpec.spawn_z
WAIT_FOR_WORLD_DEFAULT = 3.0
GAP_BETWEEN_SPAWNS = 0.3


def _resource_path(pkg: str, sub: str) -> str:
    import os

    return os.path.join(get_package_share_directory(pkg), sub)


def render_rover_sdf(index: int) -> str:
    """Read rover_bare/model.sdf, rewrite its name + camera topic per agent."""
    path = _resource_path("gazebo_gymnasium_resources", "models/rover_bare/model.sdf")
    with open(path) as f:
        content = f.read()
    content = content.replace("<model name='rover_bare'>", f'<model name="{AGENT_NAME}_{index}">')
    content = content.replace("<topic>camera</topic>", f"<topic>/rl/camera_{index}</topic>")
    return content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-agents", type=int, required=True)
    parser.add_argument(
        "--world",
        default=WORLD_NAME,
        help="name of the running gz world to spawn into "
        "(must match the launched world SDF's <world "
        "name=...>)",
    )
    parser.add_argument("--wait", type=float, default=WAIT_FOR_WORLD_DEFAULT)
    parser.add_argument("--gap", type=float, default=GAP_BETWEEN_SPAWNS)
    args = parser.parse_args()

    if args.n_agents < 1:
        print(f"ERROR: --n-agents must be >= 1, got {args.n_agents}", file=sys.stderr)
        return 2

    print(
        f"[spawn_multi_line_followers] Waiting {args.wait}s for gz sim "
        f"world {args.world!r} to come up..."
    )
    time.sleep(args.wait)

    n = args.n_agents
    offset = (n - 1) * X_SPACING / 2.0
    # ros_gz_sim create's -file loads a literal filesystem path, not a
    # package:// URI (that resolution only happens inside an SDF <include>,
    # parsed by gz itself) -- resolve it the same way render_rover_sdf does.
    track_path = _resource_path("gazebo_gymnasium_resources", "models/line_track/model.sdf")
    print(
        f"[spawn_multi_line_followers] Spawning {n} rovers along X "
        f"(spacing {X_SPACING} m), each with its own track..."
    )
    for i in range(n):
        x = i * X_SPACING - offset

        rover_cmd = [
            "ros2",
            "run",
            "ros_gz_sim",
            "create",
            "-world",
            args.world,
            "-string",
            render_rover_sdf(i),
            "-name",
            f"{AGENT_NAME}_{i}",
            "-x",
            f"{x:.3f}",
            "-y",
            f"{SPAWN_Y:.3f}",
            "-z",
            f"{SPAWN_Z:.3f}",
            "-Y",
            f"{SPAWN_YAW:.4f}",
        ]
        track_cmd = [
            "ros2",
            "run",
            "ros_gz_sim",
            "create",
            "-world",
            args.world,
            "-file",
            track_path,
            "-name",
            f"scenery_{i}",
            "-x",
            f"{x:.3f}",
            "-y",
            "0",
            "-z",
            "0",
        ]
        print(
            f"[spawn_multi_line_followers] {i + 1}/{n}: "
            f"{AGENT_NAME}_{i} + scenery_{i} at x={x:+.2f}"
        )
        for label, cmd in (("rover", rover_cmd), ("track", track_cmd)):
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(
                    f"[spawn_multi_line_followers] ERROR spawning {label} "
                    f"{i}: rc={result.returncode}\nstdout={result.stdout}\n"
                    f"stderr={result.stderr}",
                    file=sys.stderr,
                )
                return 1
        time.sleep(args.gap)

    print(f"[spawn_multi_line_followers] All {n} rovers + tracks spawned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
