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

"""Batch convert MuJoCo MJCF files into URDF + SDF + model.config.

Reads from `gazebo_gymnasium_resources/models/mujoco_sources/` and writes
each output into `gazebo_gymnasium_resources/models/<env_name>/`.

Pipeline per file:
  1. Run mjcf2urdf (splits worldbody geoms into separate URDF files)
  2. Merge split files into one URDF
  3. Post-process: capsule -> cylinder, sanitize robot name, fix xml version
  4. Run `gz sdf -p` to produce SDF
  5. Fix SDF version (1.12 -> 1.11 to match the rest of the project)
  6. Generate model.config

Failures are reported per-file; the script continues to the next env.
Run from project root: ./venv/bin/python scripts/convert_mujoco_models.py
"""

from pathlib import Path
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MUJOCO_SOURCES = (PROJECT_ROOT
                  / "gazebo_gymnasium_examples"
                  / "gazebo_gymnasium_resources"
                  / "models"
                  / "mujoco_sources")
MODELS_DIR = (PROJECT_ROOT
              / "gazebo_gymnasium_examples"
              / "gazebo_gymnasium_resources"
              / "models")
MJCF2URDF = PROJECT_ROOT / "venv" / "bin" / "mjcf2urdf"

# inverted_pendulum was hand-ported manually for higher fidelity; skip it here
# so we don't overwrite the curated URDF/SDF.
SKIP_FILES = {"inverted_pendulum.xml"}


def sanitize_name(name: str) -> str:
    """`inverted double pendulum` -> `inverted_double_pendulum`."""
    return re.sub(r"\s+", "_", name.strip()).lower()


def env_name_from_mjcf(mjcf_path: Path) -> str:
    """Return the output directory name (e.g. `half_cheetah.xml` -> `half_cheetah`)."""
    return mjcf_path.stem


def run_mjcf2urdf(mjcf_path: Path, out_dir: Path) -> list[Path]:
    """Return the list of .urdf files mjcf2urdf produced."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(MJCF2URDF), "--out", str(out_dir), str(mjcf_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"mjcf2urdf failed for {mjcf_path}:\n{result.stderr}"
        )
    return sorted(out_dir.glob("*.urdf"))


def capsule_to_cylinder(root: ET.Element) -> int:
    """In-place: replace every <capsule length=...

    radius=.../> with <cylinder length=... radius=.../>. URDF spec doesn't support capsules.
    Returns the count replaced.
    """
    n = 0
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "capsule":
                child.tag = "cylinder"
                n += 1
    return n


# Tiny non-zero mass + inertia for phantom links. Without these, `gz sdf -p`
# drops the link AND every link downstream in the chain, leaving us with an
# empty SDF. The values are small enough that the dynamics aren't affected.
PHANTOM_MASS = 1e-6
PHANTOM_INERTIA = 1e-9


def fix_zero_mass_links(root: ET.Element) -> int:
    """In-place: any <link> with mass=0 gets a tiny mass + inertia so gz sdf -p doesn't drop it.

    Returns the count fixed.
    """
    n = 0
    for link in root.iter("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_elem = inertial.find("mass")
        if mass_elem is None:
            continue
        try:
            value = float(mass_elem.get("value", "0"))
        except ValueError:
            value = 0.0
        if value <= 0.0:
            mass_elem.set("value", str(PHANTOM_MASS))
            ix = inertial.find("inertia")
            if ix is not None:
                for axis in ("ixx", "iyy", "izz"):
                    try:
                        if float(ix.get(axis, "0")) <= 0.0:
                            ix.set(axis, str(PHANTOM_INERTIA))
                    except ValueError:
                        ix.set(axis, str(PHANTOM_INERTIA))
            n += 1
    return n


def merge_urdfs(urdf_files: list[Path], target_name: str) -> str:
    """Combine multiple URDF files into one.

    Strategy: take the largest by link count (the kinematic chain) as the
    base; concatenate any standalone links from the other URDF files as
    fixed-to-world helpers (used for worldbody-level geoms like floors and
    rails).

    Returns the merged URDF as a serialized XML string.
    """
    if not urdf_files:
        raise RuntimeError("no URDF files produced")

    # Parse all
    parsed = []
    for p in urdf_files:
        tree = ET.parse(p)
        root = tree.getroot()
        link_count = len(root.findall("link"))
        parsed.append((p, root, link_count))

    # Sort so the largest is first (the actual kinematic chain).
    parsed.sort(key=lambda x: -x[2])
    main_root = parsed[0][1]

    # Rename main robot
    main_root.set("name", target_name)

    # Merge in extra links from the other URDF files. For each: add the
    # link, then attach it to the first link in main with a fixed joint.
    # The first link in `main` is treated as the model root.
    if len(parsed) > 1:
        root_link_name = main_root.find("link").get("name")
        for path, extra_root, _ in parsed[1:]:
            for extra_link in extra_root.findall("link"):
                # Avoid duplicate link names
                existing_names = {ln.get("name") for ln in main_root.findall("link")}
                lname = extra_link.get("name")
                while lname in existing_names:
                    lname = f"{lname}_extra"
                extra_link.set("name", lname)
                main_root.append(extra_link)

                # Add a fixed joint anchoring it.
                fj = ET.SubElement(main_root, "joint")
                fj.set("name", f"world_to_{lname}")
                fj.set("type", "fixed")
                ET.SubElement(fj, "parent", attrib={"link": root_link_name})
                ET.SubElement(fj, "child", attrib={"link": lname})

    return ET.tostring(main_root, encoding="unicode")


def clean_urdf_text(urdf_text: str, target_name: str) -> str:
    """Post-process the merged URDF text.

    Ensures a proper XML declaration, replaces any leftover capsule refs
    (defensive — already done on the parsed tree), and sanitizes whitespace.
    """
    # Re-parse so we can fix capsule, phantom-mass, robot name, version
    root = ET.fromstring(urdf_text)
    root.set("name", target_name)
    capsule_to_cylinder(root)
    fix_zero_mass_links(root)

    body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0"?>\n' + body + "\n"


def urdf_to_sdf(urdf_path: Path, sdf_path: Path) -> None:
    """Run `gz sdf -p` to convert URDF -> SDF and write to disk."""
    result = subprocess.run(
        ["gz", "sdf", "-p", str(urdf_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gz sdf -p failed for {urdf_path}:\n{result.stderr}"
        )
    sdf_text = result.stdout
    # Pin SDF version to 1.11 so it matches the rest of the project (gz sdf
    # emits 1.12 by default, which trips the version-conversion check when
    # included in our world SDFs declared as 1.8 -> internally upgraded to 1.11).
    sdf_text = re.sub(
        r"<sdf\s+version='1\.\d+'>",
        "<sdf version='1.11'>",
        sdf_text, count=1,
    )
    # Strip phantom <frame> blocks that gz sdf -p emits at the end of the
    # model. They reuse joint/link names which causes FrameAttachedToGraph
    # cycle errors during validation (Error Code 24/28). Without this step
    # the user's launch fails with errors like
    #   "PoseRelativeToGraph error: multiple incoming edges to current
    #    vertex [inverted_double_pendulum::pole]"
    sdf_text = re.sub(
        r"\s*<frame\b[^>]*>.*?</frame>\s*",
        "\n",
        sdf_text, flags=re.DOTALL,
    )
    sdf_text = re.sub(r"\n{3,}", "\n\n", sdf_text)
    # mjcf2urdf emits collision-only geometry — MJCF uses one <geom> for both
    # physics and rendering, but URDF separates them. Without an explicit
    # <visual> the model is physically there but Gazebo renders nothing.
    # Add a default-material visual mirroring each collision so spawned
    # models actually appear in the scene.
    sdf_text = _add_default_visuals(sdf_text)
    if not sdf_text.lstrip().startswith("<?xml"):
        sdf_text = '<?xml version="1.0"?>\n' + sdf_text
    sdf_path.write_text(sdf_text)


_LINK_RE = re.compile(
    r"(?P<header>[ \t]*<link\s+name='[^']+'>)(?P<body>.*?)(?P<footer>[ \t]*</link>)",
    re.DOTALL,
)
_COLLISION_RE = re.compile(
    r"(?P<indent>[ \t]*)<collision\s+name='(?P<name>[^']+)'>"
    r"(?P<body>.*?)</collision>",
    re.DOTALL,
)
_DEFAULT_MATERIAL_BLOCK = (
    "<material>\n"
    "          <ambient>0.5 0.5 0.5 1</ambient>\n"
    "          <diffuse>0.7 0.7 0.7 1</diffuse>\n"
    "          <specular>0.1 0.1 0.1 1</specular>\n"
    "        </material>"
)


def _add_default_visuals(sdf_text: str) -> str:
    """Append a default-material `<visual>` for each `<collision>` in links that lack one."""

    def process_link(link_match: re.Match) -> str:
        body = link_match.group("body")
        if "<visual " in body or "<visual\n" in body:
            return link_match.group(0)

        def collision_to_pair(c_match: re.Match) -> str:
            indent = c_match.group("indent")
            name = c_match.group("name")
            inner = c_match.group("body")
            visual_name = f"{name.replace('_collision', '')}_visual"
            return (
                f"{c_match.group(0)}\n"
                f"{indent}<visual name='{visual_name}'>"
                f"{inner}"
                f"{indent}  {_DEFAULT_MATERIAL_BLOCK}\n"
                f"{indent}</visual>"
            )

        new_body = _COLLISION_RE.sub(collision_to_pair, body)
        return f"{link_match.group('header')}{new_body}{link_match.group('footer')}"

    return _LINK_RE.sub(process_link, sdf_text)


def write_model_config(model_dir: Path, env_name: str, sdf_filename: str) -> None:
    (model_dir / "model.config").write_text(f"""<?xml version="1.0"?>
<model>
  <name>{env_name}</name>
  <version>1.0</version>
  <sdf version="1.8">{sdf_filename}</sdf>

  <author>
    <name>Lucas Wendland</name>
    <email>lwendlan@umd.edu</email>
  </author>

  <description>
    Auto-converted from Gymnasium MuJoCo MJCF source:
    models/mujoco_sources/{env_name}.xml
  </description>
</model>
""")


def convert_one(mjcf_path: Path) -> tuple[bool, str]:
    """Return (success, summary message)."""
    env_name = env_name_from_mjcf(mjcf_path)
    model_dir = MODELS_DIR / env_name
    urdf_path = model_dir / f"{env_name}.urdf"
    sdf_path = model_dir / f"{env_name}.sdf"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        try:
            urdf_files = run_mjcf2urdf(mjcf_path, tmp_dir)
        except Exception as e:
            return False, f"mjcf2urdf step failed: {e}"

        try:
            merged = merge_urdfs(urdf_files, env_name)
            cleaned = clean_urdf_text(merged, env_name)
        except Exception as e:
            return False, f"URDF merge/clean failed: {e}"

    model_dir.mkdir(parents=True, exist_ok=True)
    urdf_path.write_text(cleaned)

    try:
        urdf_to_sdf(urdf_path, sdf_path)
    except Exception as e:
        return False, f"gz sdf step failed: {e}"

    write_model_config(model_dir, env_name, sdf_path.name)

    return True, (f"{env_name}: URDF + SDF + model.config "
                  f"({len(cleaned)} URDF bytes, {sdf_path.stat().st_size} SDF bytes)")


def main():
    if not MJCF2URDF.exists():
        print(f"FATAL: {MJCF2URDF} not found. Activate the venv with mjcf2urdf installed.")
        sys.exit(1)

    print(f"Source: {MUJOCO_SOURCES}")
    print(f"Output: {MODELS_DIR}")
    print()

    successes = []
    failures = []

    for mjcf_path in sorted(MUJOCO_SOURCES.glob("*.xml")):
        if mjcf_path.name in SKIP_FILES:
            print(f"  SKIP {mjcf_path.name} (hand-ported)")
            continue
        ok, msg = convert_one(mjcf_path)
        if ok:
            print(f"  OK   {msg}")
            successes.append(mjcf_path.name)
        else:
            print(f"  FAIL {mjcf_path.name}: {msg}")
            failures.append((mjcf_path.name, msg))

    print()
    print(f"Done. {len(successes)} succeeded, {len(failures)} failed.")
    if failures:
        print()
        print("Failures (need manual porting):")
        for name, msg in failures:
            print(f"  {name}: {msg}")


if __name__ == "__main__":
    main()
