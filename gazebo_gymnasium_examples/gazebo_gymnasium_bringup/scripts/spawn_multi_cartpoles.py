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
"""Spawn N cartpoles into the running `cartpole_multi` gz sim world.

Called by `cartpole_multi.launch.py` after gz sim is up. Sequential
spawning sidesteps the world-load race that capped pre-generated SDFs
at 4 cartpoles: each PythonSystemLoader plugin gets time to import +
configure before the next spawn fires.

Usage:
    python3 spawn_multi_cartpoles.py --n-agents 8
"""

import argparse
import math
import subprocess
import sys
import time

X_SPACING = 3.0
# Cartpole's slider rail is 8 m along Y; this clears it with a small
# gap so cartpoles in neighbouring grid rows don't collide.
Y_SPACING = 8.5
WORLD_NAME = "cartpole_multi"
WAIT_FOR_WORLD_DEFAULT = 3.0
GAP_BETWEEN_SPAWNS = 0.3


def grid_position(index: int, n: int) -> tuple:
    """Return (x, y) for cartpole #index in a roughly square N-grid.

    cols = ceil(sqrt(N)), row-major fill. Grid is centered at (0, 0).
    Y spacing is wider than X to clear the 8 m slider rail.
    """
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    col = index % cols
    row = index // cols
    x = (col - (cols - 1) / 2.0) * X_SPACING
    y = (row - (rows - 1) / 2.0) * Y_SPACING
    return x, y


_DEFAULT_URI = "package://gazebo_gymnasium_resources/models/cartpole"


def render_cartpole_sdf(index: int, model_uri: str = _DEFAULT_URI) -> str:
    """Build the SDF string sent to ros_gz_sim create for cartpole #index.

    `<include merge="true">` splices the chosen cartpole model into an outer
    `<model>` wrapper with a world-fixed joint pinning the slider rail. Pass
    ``model_uri=.../cartpole_bare`` for the batched-harness backend (bare
    geometry; the world-level harness plugin actuates + senses via the ECM).

    Position is NOT in the SDF — the EntityFactory service overrides the SDF
    <pose> with the create command's -x/-y/-z pose.
    """
    name = f"cartpole_{index}"
    return (
        '<sdf version="1.8">'
        f'<model name="{name}">'
        "<self_collide>true</self_collide>"
        '<include merge="true">'
        f"<uri>{model_uri}</uri>"
        "</include>"
        '<joint name="world_to_slider" type="fixed">'
        "<parent>world</parent><child>slider</child>"
        "</joint>"
        "</model>"
        "</sdf>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-agents", type=int, required=True, help="Number of cartpoles to spawn."
    )
    parser.add_argument(
        "--model-uri",
        default=_DEFAULT_URI,
        help="model to spawn (use .../cartpole_bare for the harness backend)",
    )
    parser.add_argument(
        "--world",
        default=WORLD_NAME,
        help="name of the running gz world to spawn into "
        "(must match the launched world SDF's <world "
        "name=...>)",
    )
    parser.add_argument(
        "--spawn-z",
        type=float,
        default=0.1,
        help="spawn height. Force-actuated models (the "
        "harness/ECM backends) must spawn CLEAR of the "
        "ground plane (e.g. 0.6) or contact friction "
        "pins the cart; match AgentSpec.spawn_z.",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=WAIT_FOR_WORLD_DEFAULT,
        help="Seconds to wait for gz sim before first spawn.",
    )
    parser.add_argument(
        "--gap", type=float, default=GAP_BETWEEN_SPAWNS, help="Seconds between successive spawns."
    )
    args = parser.parse_args()

    if args.n_agents < 1:
        print(f"ERROR: --n-agents must be >= 1, got {args.n_agents}", file=sys.stderr)
        return 2

    print(
        f"[spawn_multi_cartpoles] Waiting {args.wait}s for gz sim world "
        f"'{args.world}' to come up..."
    )
    time.sleep(args.wait)

    n = args.n_agents
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    print(
        f"[spawn_multi_cartpoles] Spawning {n} cartpoles in a "
        f"{rows}x{cols} grid (X spacing {X_SPACING} m, "
        f"Y spacing {Y_SPACING} m, gap {args.gap}s between spawns)..."
    )
    for i in range(n):
        x, y = grid_position(i, n)
        sdf = render_cartpole_sdf(i, args.model_uri)
        cmd = [
            "ros2",
            "run",
            "ros_gz_sim",
            "create",
            "-world",
            args.world,
            "-string",
            sdf,
            "-name",
            f"cartpole_{i}",
            "-x",
            f"{x:.3f}",
            "-y",
            f"{y:.3f}",
            "-z",
            f"{args.spawn_z:.3f}",
        ]
        print(f"[spawn_multi_cartpoles] {i + 1}/{n}: cartpole_{i} at ({x:+.2f}, {y:+.2f})")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(
                f"[spawn_multi_cartpoles] ERROR spawning cartpole_{i}: "
                f"rc={result.returncode}\n"
                f"stdout={result.stdout}\nstderr={result.stderr}",
                file=sys.stderr,
            )
            return 1
        time.sleep(args.gap)

    print(f"[spawn_multi_cartpoles] All {n} cartpoles spawned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
