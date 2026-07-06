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

"""
Run `gz sdf --check` against every shipped model + world SDF.

This is the deeper sibling of `test_sdf_validity.py`. That one inspects the
XML statically and catches:
    * duplicate link/joint/frame names
    * frame-name collisions with links/joints
    * <link> with collisions but no visuals (the "invisible model" bug)

This file catches everything else — the semantic problems that only show
up once libsdformat parses the file:
    * FrameAttachedToGraph cycles (the original `PoseRelativeToGraph error:
      multiple incoming edges` failure)
    * unresolved joint parents/children
    * malformed <pose> tags
    * version mismatches between included models and the world

For world SDFs we substitute `package://gazebo_gymnasium_resources/models/X`
with the filesystem path under `models/` so `gz sdf --check` can resolve
the include without needing a colcon install to be sourced.

Skipped gracefully if `gz` isn't on PATH — same shape as the
`pytest.importorskip` pattern used in the other test files. This means the
test still passes in CI environments without Gazebo installed; it only
gates when Gazebo is actually available.

Run with: pytest gazebo_gymnasium_bridge/test/test_sdf_loads_in_gazebo.py -v
"""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import pytest


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
RESOURCES = (PROJECT_ROOT
             / "gazebo_gymnasium_examples"
             / "gazebo_gymnasium_resources")
MODELS_DIR = RESOURCES / "models"
WORLDS_DIR = RESOURCES / "worlds"


# Skip the whole file if `gz` isn't on PATH. Local devs without Gazebo
# installed shouldn't fail this — the static `test_sdf_validity.py` still
# runs for them and catches the most common regression class.
if shutil.which("gz") is None:
    pytest.skip(
        "`gz` CLI not found on PATH — skipping Gazebo-backed SDF checks.",
        allow_module_level=True,
    )


def _model_paths() -> list[Path]:
    if not MODELS_DIR.exists():
        return []
    out = []
    for d in sorted(MODELS_DIR.iterdir()):
        if not d.is_dir() or d.name == "mujoco_sources":
            continue
        primary = d / f"{d.name}.sdf"
        alt = d / "model.sdf"  # diff_drive convention
        if primary.exists():
            out.append(primary)
        elif alt.exists():
            out.append(alt)
    return out


def _world_paths() -> list[Path]:
    if not WORLDS_DIR.exists():
        return []
    return sorted(WORLDS_DIR.glob("*.sdf"))


def _resolve_package_uris(text: str) -> str:
    """Substitute ``package://`` model URIs with their on-disk paths.

    Lets ``gz sdf --check`` resolve ``<include>`` blocks.
    """
    return re.sub(
        r"package://gazebo_gymnasium_resources/models/([^<\s]+)",
        lambda m: f"file://{MODELS_DIR}/{m.group(1)}",
        text,
    )


def _gz_sdf_check(sdf_path: Path, resolve_uris: bool) -> tuple[bool, str]:
    target = sdf_path
    tmp = None
    try:
        if resolve_uris:
            text = _resolve_package_uris(sdf_path.read_text())
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".sdf", delete=False)
            tmp.write(text)
            tmp.close()
            target = Path(tmp.name)
        result = subprocess.run(
            ["gz", "sdf", "--check", str(target)],
            capture_output=True, text=True, timeout=30,
        )
        out = result.stderr + result.stdout
    finally:
        if tmp is not None:
            Path(tmp.name).unlink(missing_ok=True)
    return ("Error Code" not in out), out


@pytest.mark.parametrize("sdf_path", _model_paths(), ids=lambda p: p.parent.name)
def test_model_sdf_loads(sdf_path):
    """Each model SDF must pass `gz sdf --check` cleanly."""
    ok, output = _gz_sdf_check(sdf_path, resolve_uris=False)
    assert ok, (
        f"{sdf_path.relative_to(PROJECT_ROOT)} failed gz sdf --check:\n"
        f"{output}"
    )


@pytest.mark.parametrize("sdf_path", _world_paths(), ids=lambda p: p.stem)
def test_world_sdf_loads(sdf_path):
    """Check each world SDF passes ``gz sdf --check``.

    URIs (``package://``) are resolved to their on-disk paths first.
    """
    ok, output = _gz_sdf_check(sdf_path, resolve_uris=True)
    assert ok, (
        f"{sdf_path.relative_to(PROJECT_ROOT)} failed gz sdf --check:\n"
        f"{output}"
    )
