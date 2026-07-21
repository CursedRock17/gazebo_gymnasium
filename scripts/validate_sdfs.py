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

"""Headless validator for every model and world SDF in the project.

Runs `gz sdf --check` (semantic check, no GUI, no physics) against:
  * Every `<env>/<env>.sdf` model file
  * Every `worlds/*.sdf` file

Reports pass/fail per file with the actual error lines from gz sdf. Use this
as a regression gate before claiming an env is ready — if a model fails the
semantic check, the launch will explode when ApplyJointForce or
JointStatePublisher tries to resolve a link.

Exit code: 0 if everything validates, 1 otherwise.

Run from project root: ./venv/bin/python scripts/validate_sdfs.py
"""

from pathlib import Path
import re
import subprocess
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCES = PROJECT_ROOT / "gazebo_gymnasium_examples" / "gazebo_gymnasium_resources"
MODELS_DIR = RESOURCES / "models"
WORLDS_DIR = RESOURCES / "worlds"


def is_valid_sdf(stderr: str) -> bool:
    # `gz sdf --check` returns 0 even on errors, so parse stderr for "Error Code".
    return "Error Code" not in stderr


def check_one(path: Path) -> tuple[bool, str]:
    """Validate a model SDF or world SDF.

    World SDFs use `package://` URIs that gz sdf --check can't resolve; substitute them with the
    on-disk model dir before checking.
    """
    target = path
    is_world = path.parent == WORLDS_DIR
    tmp_file = None
    if is_world:
        text = path.read_text()
        # Replace package://gazebo_gymnasium_resources/models/<env> with the
        # actual model dir; that lets `gz sdf --check` resolve the include.
        text = re.sub(
            r"package://gazebo_gymnasium_resources/models/([^<\s]+)",
            lambda m: f"file://{MODELS_DIR}/{m.group(1)}",
            text,
        )
        tmp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".sdf", delete=False)
        tmp_file.write(text)
        tmp_file.close()
        target = Path(tmp_file.name)

    try:
        result = subprocess.run(
            ["gz", "sdf", "--check", str(target)],
            capture_output=True, text=True, timeout=30,
        )
        combined = result.stderr + result.stdout
    finally:
        if tmp_file is not None:
            Path(tmp_file.name).unlink(missing_ok=True)

    return is_valid_sdf(combined), combined


def collect_targets() -> list[Path]:
    targets = []
    # Model SDFs
    for model_dir in sorted(MODELS_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "mujoco_sources":
            continue
        sdf = model_dir / f"{model_dir.name}.sdf"
        if sdf.exists():
            targets.append(sdf)
        else:
            # diff_drive + rover use the canonical model.sdf naming.
            alt = model_dir / "model.sdf"
            if alt.exists():
                targets.append(alt)
    # World SDFs — but most worlds use `<include>` which gz sdf --check can't
    # resolve from package:// URIs unless GZ_SIM_RESOURCE_PATH points at the
    # share directory. We still validate them; failures from unresolved
    # includes will show up as "Could not find" errors and we'll know to
    # source the install before re-running.
    for world in sorted(WORLDS_DIR.glob("*.sdf")):
        targets.append(world)
    return targets


def main() -> int:
    targets = collect_targets()
    if not targets:
        print("No SDFs found.")
        return 1

    print(f"Validating {len(targets)} SDF files...\n")
    passed, failed = [], []
    for path in targets:
        ok, output = check_one(path)
        rel = path.relative_to(PROJECT_ROOT)
        if ok:
            print(f"  OK   {rel}")
            passed.append(path)
        else:
            print(f"  FAIL {rel}")
            # First 5 error lines is enough to fingerprint the issue.
            err_lines = [line for line in output.splitlines() if "Error Code" in line]
            for line in err_lines[:5]:
                print(f"       {line.strip()}")
            if len(err_lines) > 5:
                print(f"       ... and {len(err_lines) - 5} more errors")
            failed.append((path, output))

    print()
    print(f"Summary: {len(passed)} passed, {len(failed)} failed.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
