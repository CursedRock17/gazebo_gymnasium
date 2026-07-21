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

"""One-shot converter for the Onshape-exported rover URDF.

Produces `models/rover/model.sdf` from `models/rover/rover.urdf`. Does the
ROS-2-specific cleanup the converted SDF needs to actually work in Gazebo:

  1. Rewrites the broken `package://assets/meshes/X.stl` URIs to point at
     the real `package://gazebo_gymnasium_resources/models/rover/assets/X.stl`
     locations (the meshes/ subdir doesn't exist on disk; the onshape-to-
     robot exporter assumes a sub-folder that we don't have).
  2. Runs `gz sdf -p` to convert URDF -> SDF.
  3. Pins the SDF version to 1.11 (matches the rest of the project).
  4. Strips phantom <frame> blocks (URDF->SDF conversion emits these and
     they break the frame-attached-to graph).
  5. Drops the collisions for all the internal chassis parts (motors,
     brackets, battery, esp32, motor_controller, voltage_regulator). They
     are *inside* the base_plate enclosure and don't participate in any
     real-world contact resolution.
  6. Also drops the front_wheel_lower_joint collision (structural caster
     mount, never touches the ground).
  7. Appends a DiffDrive system plugin so the rover responds to /cmd_vel,
     plus a JointStatePublisher and an OdometryPublisher.
  8. Appends a forward-facing camera link + sensor at the front of the
     chassis.

Run from project root:
    ./venv/bin/python scripts/convert_rover_urdf.py
"""

from pathlib import Path
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROVER_DIR = (PROJECT_ROOT
             / "gazebo_gymnasium_examples" / "gazebo_gymnasium_resources"
             / "models" / "rover")
URDF_PATH = ROVER_DIR / "rover.urdf"
SDF_PATH = ROVER_DIR / "model.sdf"
CONFIG_PATH = ROVER_DIR / "model.config"


# Names (without prefix) of collision blocks that should be DROPPED. These
# are all the internal chassis parts whose collision contributes nothing
# meaningful to physics — they live inside the base_plate enclosure.
# Drop every mesh collision from the converted SDF. Only the three wheel
# spheres (added back below) participate in contact resolution.
COLLISION_PART_NAMES_TO_DROP = {
    # Internal-chassis parts (live inside the base_plate enclosure):
    "left_motor",
    "right_motor",
    "left_motor_bracket",
    "right_motor_bracket",
    "battery",
    "esp32",
    "motor_controller",
    "voltage_regulator",
    "front_wheel_upper_joint",
    "front_wheel_bracket",
    "front_wheel_lower_joint",
    # The chassis itself — base_plate's mesh collision is the single most
    # expensive contact in the rover. Replaced with no collision at all,
    # which is fine because nothing should be hitting the chassis at the
    # speeds we run.
    "base_plate",
    # Wheel mesh collisions — we replace these with sphere primitives in
    # the post-conversion pass so wheel-ground contact stays simple.
    "left_wheel",
    "right_wheel",
    "front_wheel",
}


# Wheel sphere radii. Drive-wheel radius derived from the URDF's left_wheel
# inertia (mass + izz) using a thin-disk approximation. The front caster
# is smaller in real life; 0.025 m matches eyeballing the STL and keeps
# the rover at a stable forward pitch.
DRIVE_WHEEL_RADIUS = 0.034
CASTER_WHEEL_RADIUS = 0.025


WHEEL_SPHERE_COLLISIONS = {
    # link name -> (radius, friction overrides). Friction is intentionally
    # high on the drive wheels so they don't slip when accelerating; the
    # caster gets low friction so it doesn't fight the swivel.
    "left_wheel":  (DRIVE_WHEEL_RADIUS, 1.2, 0.8),
    "right_wheel": (DRIVE_WHEEL_RADIUS, 1.2, 0.8),
    "front_wheel": (CASTER_WHEEL_RADIUS, 0.1, 0.1),
}


# After gz sdf -p, collisions on a single link are auto-named
# `<link>_collision`, `<link>_collision_1`, `<link>_collision_2`, ...
# corresponding to the order parts appear in the URDF. We identify which
# index to drop by matching the embedded mesh URI rather than by index —
# that way the script keeps working even if the URDF order shifts.
COLLISION_RE = re.compile(
    r"      <collision name='[^']+'>\n"
    r".*?"
    r"      </collision>\n",
    re.DOTALL,
)


# Strips phantom <frame> blocks gz sdf -p emits at the bottom of each
# model. Mirrors the regex in scripts/fix_sdf_phantom_frames.py.
FRAME_RE = re.compile(
    r"\s*<frame\b[^>]*>.*?</frame>\s*",
    re.DOTALL,
)


# DiffDrive + JointState + Odometry plugins go at the END of the <model>
# block. Wheel separation is ~0.163 m (lateral distance between left and
# right wheel attachment points in the URDF). Wheel radius ~0.034 m
# computed from the wheel's mass + izz inertia (thin-disk approximation).
DRIVETRAIN_PLUGIN_BLOCK = """
    <!-- DiffDrive system: drives left + right wheel joints from /cmd_vel.
         Wheel separation is the lateral distance between the wheel
         attachment points on base_link (from rover.urdf joint origins).
         Wheel radius derived from the wheel's mass + izz inertia
         (thin-disk approximation: r = sqrt(2*Izz/m) ≈ 0.034 m). -->
    <plugin
      filename="gz-sim-diff-drive-system"
      name="gz::sim::systems::DiffDrive">
      <left_joint>left_axle</left_joint>
      <right_joint>right_axle</right_joint>
      <wheel_separation>0.163</wheel_separation>
      <wheel_radius>0.034</wheel_radius>
      <odom_publish_frequency>30</odom_publish_frequency>
      <!-- No explicit <topic> tag: DiffDrive defaults to
           /model/<model_name>/cmd_vel, which is what the sync-gate
           plugin publishes to. Setting <topic>cmd_vel</topic> here
           would override that and silently lose commands. -->
      <!-- Physical caster-pop limit is around 1 m/s forward; clamp here
           so even a misbehaving policy can't blow past it. Yaw limit
           matches the env's MAX_Z_VEL spec. -->
      <max_linear_velocity>1.0</max_linear_velocity>
      <min_linear_velocity>-1.0</min_linear_velocity>
      <max_angular_velocity>1.25</max_angular_velocity>
      <min_angular_velocity>-1.25</min_angular_velocity>
    </plugin>

    <plugin
      filename="gz-sim-joint-state-publisher-system"
      name="gz::sim::systems::JointStatePublisher">
      <joint_name>left_axle</joint_name>
      <joint_name>right_axle</joint_name>
      <joint_name>front_swivel</joint_name>
      <joint_name>front_axle</joint_name>
    </plugin>
"""


# Forward-facing camera link. The Onshape-exported rover uses -Y as its
# forward direction (the front caster is at y ≈ -0.098 in base_link's
# frame). The camera sits a bit ahead of the front caster, raised above
# the chassis, pitched 0.5 rad (~28°) down so it sees the line track on
# the ground a short distance ahead.
#
# Pose rotation derivation (RPY = roll, pitch, yaw, applied extrinsically):
#   yaw=-π/2 rotates the camera's optical axis (+X) from rover-frame +X
#     to rover-frame -Y, which is the rover's forward direction.
#   pitch=+0.5 rad tilts the camera down by ~28° from the horizontal so
#     the ground a short distance ahead falls within the field of view.
#   roll=0 keeps the camera upright (no banking).
CAMERA_LINK_BLOCK = """
    <!-- Forward-facing camera for vision-based control. Mounted ahead of
         the front caster (rover forward = -Y in base_link frame). The
         pose rotates the camera's +X optical axis to point along rover
         -Y, then pitches it ~28° down so it sees the ground a short
         distance ahead — where the line track lives. -->
    <link name="camera_link">
      <pose relative_to="base_link">0 -0.15 0.05 0 0.5 -1.5708</pose>
      <inertial>
        <mass>0.005</mass>
        <inertia>
          <ixx>1e-7</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>1e-7</iyy>
          <iyz>0</iyz>
          <izz>1e-7</izz>
        </inertia>
      </inertial>
      <visual name="visual">
        <geometry>
          <box>
            <size>0.02 0.04 0.02</size>
          </box>
        </geometry>
        <material>
          <ambient>0.1 0.1 0.1 1</ambient>
          <diffuse>0.1 0.1 0.1 1</diffuse>
          <specular>0.05 0.05 0.05 1</specular>
        </material>
      </visual>
      <sensor name="camera" type="camera">
        <pose>0 0 0 0 0 0</pose>
        <topic>camera</topic>
        <gz_frame_id>rover/camera_link</gz_frame_id>
        <!-- 15 Hz is plenty for line following at the rover's 1 m/s top
             speed (that's ~7 cm of travel per frame). Lower than 30 Hz
             halves the rendering cost, which is the dominant per-step
             expense once a camera enters the world. -->
        <update_rate>15</update_rate>
        <always_on>1</always_on>
        <visualize>true</visualize>
        <camera>
          <horizontal_fov>1.047</horizontal_fov>
          <image>
            <!-- 160x120 is plenty of resolution for line detection
                 (~0.13 m of ground per pixel at the rover's camera
                 distance). Quarters the per-frame render + threshold
                 cost vs 320x240. -->
            <width>160</width>
            <height>120</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.05</near>
            <far>50</far>
          </clip>
        </camera>
      </sensor>
    </link>

    <joint name="camera_joint" type="fixed">
      <parent>base_link</parent>
      <child>camera_link</child>
    </joint>
"""


MODEL_CONFIG = """<?xml version="1.0"?>
<model>
  <name>rover</name>
  <version>1.0</version>
  <sdf version="1.8">model.sdf</sdf>

  <author>
    <name>Lucas Wendland</name>
    <email>lwendlan@umd.edu</email>
  </author>

  <description>
    Differential-drive rover exported from Onshape via onshape-to-robot.
    Original meshes live in `assets/`. The SDF is generated by
    `scripts/convert_rover_urdf.py` from `rover.urdf`. Cleaned-up: only
    the chassis baseplate + the four wheels carry collisions (internal
    motors/electronics are visual-only). Forward-facing camera bolted to
    the front of the chassis for vision-based control.
  </description>
</model>
"""


def fix_urdf_mesh_paths(urdf_text: str) -> str:
    return urdf_text.replace(
        "package://assets/meshes/",
        "package://gazebo_gymnasium_resources/models/rover/assets/",
    )


def run_gz_sdf(urdf_path: Path) -> str:
    result = subprocess.run(
        ["gz", "sdf", "-p", str(urdf_path)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gz sdf -p failed:\n{result.stderr}")
    return result.stdout


def pin_sdf_version(sdf_text: str) -> str:
    return re.sub(
        r"<sdf\s+version='1\.\d+'>",
        "<sdf version='1.11'>",
        sdf_text, count=1,
    )


def drop_collisions_referencing(sdf_text: str, part_basenames: set[str]) -> tuple[str, int]:
    """Drop collision blocks whose mesh URI matches any of the named parts.

    `part_basenames` is the set of mesh filename stems we want to drop
    (e.g. "left_motor" matches `assets/left_motor.stl`).
    """
    n_dropped = 0

    def _replace(m: re.Match) -> str:
        nonlocal n_dropped
        block = m.group(0)
        for name in part_basenames:
            if f"/{name}.stl" in block:
                n_dropped += 1
                return ""
        return block

    new_text = COLLISION_RE.sub(_replace, sdf_text)
    return new_text, n_dropped


def strip_phantom_frames(sdf_text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", FRAME_RE.sub("\n", sdf_text))


def rename_model(sdf_text: str, new_name: str) -> str:
    return re.sub(
        r"<model name='[^']+'>",
        f"<model name='{new_name}'>",
        sdf_text, count=1,
    )


def inject_extras_before_model_close(sdf_text: str, extras: str) -> str:
    """Place plugin + camera blocks just before `</model>`."""
    marker = "  </model>"
    idx = sdf_text.rfind(marker)
    if idx == -1:
        raise RuntimeError("Could not find </model> close tag.")
    return sdf_text[:idx] + extras + "\n" + sdf_text[idx:]


def flip_right_axle_axis(sdf_text: str) -> str:
    """Negate `right_axle`'s rotation axis so both wheels spin the same way.

    The Onshape export gives `left_axle` and `right_axle` opposite-handed
    joint frames (left's effective axis points -X in model frame, right's
    points +X). DiffDrive issues `+vel` to both wheels for forward motion;
    with mismatched axes, the left wheel rolls forward but the right wheel
    rolls backward, so the rover spins in place instead of driving.

    Flipping right's axis (0, 0, 1) → (0, 0, -1) aligns the effective
    rotation direction for both wheels.
    """
    pattern = re.compile(
        r"(<joint name='right_axle'[^>]*>.*?<axis>\s*<xyz>)0 0 1(</xyz>)",
        re.DOTALL,
    )
    new_text, n = pattern.subn(r"\g<1>0 0 -1\g<2>", sdf_text, count=1)
    if n == 0:
        raise RuntimeError("Could not find right_axle <axis> block to flip.")
    return new_text


def inject_wheel_sphere_collisions(sdf_text: str) -> str:
    """Add a sphere `<collision>` to each wheel link.

    After `drop_collisions_referencing`, the wheel links contain only a
    `<visual>` (their mesh `<collision>` is gone). We splice a small
    sphere collision in before `</link>` for each of the three wheels.
    Sphere primitives are orders of magnitude cheaper than mesh-vs-plane
    contact resolution, which was the dominant cost in the old SDF.
    """
    for link_name, (radius, mu, mu2) in WHEEL_SPHERE_COLLISIONS.items():
        collision_block = (
            f"      <collision name='{link_name}_collision'>\n"
            f"        <geometry>\n"
            f"          <sphere>\n"
            f"            <radius>{radius}</radius>\n"
            f"          </sphere>\n"
            f"        </geometry>\n"
            f"        <surface>\n"
            f"          <friction>\n"
            f"            <ode>\n"
            f"              <mu>{mu}</mu>\n"
            f"              <mu2>{mu2}</mu2>\n"
            f"            </ode>\n"
            f"          </friction>\n"
            f"        </surface>\n"
            f"      </collision>\n"
        )
        # Insert right before `</link>` of this specific link. Match the
        # link header, then a non-greedy body, then the closing tag.
        pattern = re.compile(
            rf"(<link name='{re.escape(link_name)}'>.*?)(\s*</link>)",
            re.DOTALL,
        )
        sdf_text, n = pattern.subn(
            lambda m: m.group(1) + "\n" + collision_block + m.group(2),
            sdf_text,
            count=1,
        )
        if n == 0:
            raise RuntimeError(
                f"Could not find <link name='{link_name}'> to inject collision.")
    return sdf_text


def main() -> int:
    if not URDF_PATH.exists():
        print(f"ERROR: URDF not found at {URDF_PATH}", file=sys.stderr)
        return 1

    urdf_text = URDF_PATH.read_text()
    urdf_text = fix_urdf_mesh_paths(urdf_text)

    # gz sdf -p reads from a file path, not stdin. Write the fixed URDF
    # to a tempfile, then convert.
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(urdf_text)
        tmp_urdf = Path(f.name)
    try:
        sdf_text = run_gz_sdf(tmp_urdf)
    finally:
        tmp_urdf.unlink(missing_ok=True)

    sdf_text = pin_sdf_version(sdf_text)
    sdf_text = strip_phantom_frames(sdf_text)
    sdf_text, dropped = drop_collisions_referencing(
        sdf_text, COLLISION_PART_NAMES_TO_DROP)
    sdf_text = inject_wheel_sphere_collisions(sdf_text)
    sdf_text = flip_right_axle_axis(sdf_text)
    sdf_text = rename_model(sdf_text, "rover")
    sdf_text = inject_extras_before_model_close(
        sdf_text, DRIVETRAIN_PLUGIN_BLOCK + CAMERA_LINK_BLOCK)

    if not sdf_text.lstrip().startswith("<?xml"):
        sdf_text = '<?xml version="1.0"?>\n' + sdf_text

    SDF_PATH.write_text(sdf_text)
    CONFIG_PATH.write_text(MODEL_CONFIG)

    print(f"Wrote {SDF_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {CONFIG_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Dropped {dropped} unnecessary collisions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
