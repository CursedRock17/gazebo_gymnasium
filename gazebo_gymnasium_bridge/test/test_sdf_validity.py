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
Regression guard for SDF validity.

`gz sdf -p` (URDF→SDF) emits phantom `<frame>` blocks that reuse joint and
link names. Those collisions trigger `FrameAttachedToGraph` cycle errors at
launch time (`PoseRelativeToGraph error: multiple incoming edges to current
vertex [...]`). The fix is to strip those frames during conversion.

This test asserts the post-strip invariants on every shipped model SDF:
  1. No two link/joint/frame elements within the same model share a name.
  2. No `<frame>` block reuses a name already used by a link or joint.

We deliberately don't run `gz sdf --check` here — that would require Gazebo
to be installed in the test env. The static name-collision check catches
the same class of bug without any runtime deps.

Run with: pytest gazebo_gymnasium_bridge/test/test_sdf_validity.py -v
"""

from pathlib import Path
import re

import pytest

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
MODELS_DIR = PROJECT_ROOT / "gazebo_gymnasium_examples" / "gazebo_gymnasium_resources" / "models"


def model_sdf_paths():
    paths = []
    if not MODELS_DIR.exists():
        return paths
    for model_dir in sorted(MODELS_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "mujoco_sources":
            continue
        sdf = model_dir / f"{model_dir.name}.sdf"
        if sdf.exists():
            paths.append(sdf)
        else:
            # diff_drive uses model.sdf instead of <env>.sdf
            alt = model_dir / "model.sdf"
            if alt.exists():
                paths.append(alt)
    return paths


@pytest.mark.parametrize("sdf_path", model_sdf_paths(), ids=lambda p: p.parent.name)
class TestSdfValidity:
    def test_no_duplicate_named_elements(self, sdf_path):
        """No <link|joint|frame> share a name within the same model."""
        text = sdf_path.read_text()
        # Extract every named link/joint/frame at the top level of the model.
        # We don't fully parse — simple regex over the line-by-line declarations
        # is enough to catch the auto-converted-SDF failure mode.
        pattern = re.compile(r"<(link|joint|frame)\s+name='([^']+)'")
        seen = {}
        for kind, name in pattern.findall(text):
            seen.setdefault(name, []).append(kind)
        duplicates = {n: kinds for n, kinds in seen.items() if len(kinds) > 1}
        assert not duplicates, (
            f"Name collisions in {sdf_path.name}: {duplicates}. "
            f"Run scripts/fix_sdf_phantom_frames.py to strip the offending "
            f"<frame> blocks."
        )

    def test_no_frame_reuses_link_or_joint_name(self, sdf_path):
        """A <frame> name must not collide with any <link> or <joint> name."""
        text = sdf_path.read_text()
        link_names = set(re.findall(r"<link\s+name='([^']+)'", text))
        joint_names = set(re.findall(r"<joint\s+name='([^']+)'", text))
        frame_names = set(re.findall(r"<frame\s+name='([^']+)'", text))
        clashes = (link_names | joint_names) & frame_names
        assert not clashes, (
            f"{sdf_path.name}: frame names clash with link/joint names: {sorted(clashes)}"
        )

    def test_link_with_collision_also_has_visual(self, sdf_path):
        """Check every <link> with a <collision> also has a <visual>.

        Otherwise the model loads physically but renders invisible — the
        failure mode for all auto-converted MuJoCo SDFs before
        scripts/add_visuals_to_sdfs.py was run.
        """
        text = sdf_path.read_text()
        # Extract each <link>...</link> block and inspect its contents.
        link_blocks = re.findall(
            r"<link\s+name='([^']+)'>(.*?)</link>",
            text,
            re.DOTALL,
        )
        invisible_links = []
        for name, body in link_blocks:
            has_collision = "<collision " in body or "<collision\n" in body
            has_visual = "<visual " in body or "<visual\n" in body
            if has_collision and not has_visual:
                invisible_links.append(name)
        assert not invisible_links, (
            f"{sdf_path.name}: links with collisions but NO visual (will "
            f"render invisible in Gazebo): {invisible_links}. Run "
            f"scripts/add_visuals_to_sdfs.py."
        )
