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

"""Strip phantom <frame> declarations from auto-converted model SDFs.

Background:
  `gz sdf -p model.urdf` emits a bunch of helper `<frame>` declarations at
  the bottom of the produced SDF — one per joint, plus floor/world anchors.
  These come from the URDF→SDF graph-building pass. Two problems with them:

  1. They have NAMES that often collide with joint names (e.g. `bthigh` in
     half_cheetah, `pole` in inverted_double_pendulum). The collision means
     the SDF fails `gz sdf --check` with FrameAttachedToGraph cycle errors.
  2. They serve no purpose at runtime — link/joint pose resolution doesn't
     need them.

The fix: remove every `<frame ...> ... </frame>` block. Use regex because
the surrounding SDF structure is fragile (we don't want to re-emit it via
ET.tostring and lose formatting).

Idempotent: running it again on a clean SDF is a no-op.

Run from project root: ./venv/bin/python scripts/fix_sdf_phantom_frames.py
"""

from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = (PROJECT_ROOT
              / "gazebo_gymnasium_examples"
              / "gazebo_gymnasium_resources"
              / "models")

# Match `<frame name='X' attached_to='Y'> ... </frame>` even with whitespace
# and newlines inside. Use a non-greedy match for the body so we don't span
# multiple frame blocks.
FRAME_RE = re.compile(
    r"\s*<frame\b[^>]*>.*?</frame>\s*",
    re.DOTALL,
)


def fix_file(path: Path) -> tuple[int, str]:
    """Return (frames_removed, new_text)."""
    text = path.read_text()
    matches = FRAME_RE.findall(text)
    new_text = FRAME_RE.sub("\n", text)
    # Tidy up any triple-blank-lines the substitution left behind.
    new_text = re.sub(r"\n{3,}", "\n\n", new_text)
    return len(matches), new_text


def main() -> int:
    total_fixed = 0
    total_frames_removed = 0
    for model_dir in sorted(MODELS_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "mujoco_sources":
            continue
        sdf_path = model_dir / f"{model_dir.name}.sdf"
        if not sdf_path.exists():
            continue
        n, new_text = fix_file(sdf_path)
        if n == 0:
            print(f"  ok    {sdf_path.relative_to(PROJECT_ROOT)} (already clean)")
            continue
        sdf_path.write_text(new_text)
        total_fixed += 1
        total_frames_removed += n
        print(f"  fixed {sdf_path.relative_to(PROJECT_ROOT)} (-{n} phantom frames)")

    print()
    print(f"Stripped {total_frames_removed} phantom frames across "
          f"{total_fixed} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
