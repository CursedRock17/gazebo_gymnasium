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

"""Add a `<visual>` sibling to every `<collision>` block in auto-converted SDFs.

Background:
  In MJCF a single `<geom>` element provides BOTH the rigid-body shape (for
  physics) AND the rendered geometry. URDF separates them into `<collision>`
  and `<visual>`. `mjcf2urdf` (the converter we use) only emits the
  collision side — Gymnasium's MuJoCo envs are headless by default so this
  goes unnoticed in the upstream tool.

  After `gz sdf -p` runs, that means our converted SDFs have collision
  geometry only. The bodies are physically there (you can collide with
  them, gravity affects them) but Gazebo's renderer has nothing to draw,
  so the model appears invisible.

This script:
  1. Finds every `<collision name='X'> ... </collision>` block
  2. Emits an immediately-following `<visual name='X_visual'>` with the
     same pose + geometry
  3. Attaches a default light-gray material so visuals aren't pure white

Idempotent: if a visual with the matching name already exists, skip.

Run from project root:
    ./venv/bin/python scripts/add_visuals_to_sdfs.py
"""

from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = (PROJECT_ROOT
              / "gazebo_gymnasium_examples"
              / "gazebo_gymnasium_resources"
              / "models")

# Capture a full <link>...</link>. We process per-link so we can skip links
# that already have a <visual> rather than blindly adding one per collision.
LINK_RE = re.compile(
    r"(?P<header>[ \t]*<link\s+name='[^']+'>)"
    r"(?P<body>.*?)"
    r"(?P<footer>[ \t]*</link>)",
    re.DOTALL,
)

# Capture an entire <collision name='...'> ... </collision> block within a
# given link body. Non-greedy so we don't span multiple collisions.
COLLISION_RE = re.compile(
    r"(?P<indent>[ \t]*)<collision\s+name='(?P<name>[^']+)'>"
    r"(?P<body>.*?)"
    r"</collision>",
    re.DOTALL,
)

# Light gray, slight specular — readable against the default ground.
DEFAULT_MATERIAL = (
    "<material>\n"
    "  <ambient>0.5 0.5 0.5 1</ambient>\n"
    "  <diffuse>0.7 0.7 0.7 1</diffuse>\n"
    "  <specular>0.1 0.1 0.1 1</specular>\n"
    "</material>"
)


def _make_visual(indent: str, name: str, body: str) -> str:
    """Build a `<visual>` block with the given pose+geometry body."""
    visual_name = f"{name.replace('_collision', '')}_visual"
    material_lines = DEFAULT_MATERIAL.splitlines()
    indented_material = "\n".join(
        f"{indent}  {line}" if line.strip() else line
        for line in material_lines
    )
    return (
        f"{indent}<visual name='{visual_name}'>"
        f"{body}"
        f"{indent}  {indented_material.lstrip()}\n"
        f"{indent}</visual>"
    )


def add_visuals(text: str) -> tuple[str, int]:
    """Return (new_text, n_added)."""
    additions = 0

    def process_link(link_match: re.Match) -> str:
        nonlocal additions
        body = link_match.group("body")

        # If this link already declares ANY <visual>, leave it alone — the
        # hand-ported models (diff_drive, cartpole, inverted_pendulum) fall
        # in this bucket and don't need our generic gray fill.
        if "<visual " in body or "<visual\n" in body:
            return link_match.group(0)

        def collision_to_pair(c_match: re.Match) -> str:
            nonlocal additions
            indent = c_match.group("indent")
            name = c_match.group("name")
            inner = c_match.group("body")
            visual_block = _make_visual(indent, name, inner)
            additions += 1
            return f"{c_match.group(0)}\n{visual_block}"

        new_body = COLLISION_RE.sub(collision_to_pair, body)
        return f"{link_match.group('header')}{new_body}{link_match.group('footer')}"

    new_text = LINK_RE.sub(process_link, text)
    return new_text, additions


def main() -> int:
    total_files_changed = 0
    total_visuals_added = 0

    for model_dir in sorted(MODELS_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "mujoco_sources":
            continue
        sdf_paths = list(model_dir.glob("*.sdf"))
        if not sdf_paths:
            continue
        for sdf_path in sdf_paths:
            text = sdf_path.read_text()
            new_text, n = add_visuals(text)
            if n == 0:
                print(f"  ok    {sdf_path.relative_to(PROJECT_ROOT)} "
                      f"(no visuals needed)")
                continue
            sdf_path.write_text(new_text)
            total_files_changed += 1
            total_visuals_added += n
            print(f"  fixed {sdf_path.relative_to(PROJECT_ROOT)} "
                  f"(+{n} visuals)")

    print()
    print(f"Added {total_visuals_added} visuals across "
          f"{total_files_changed} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
