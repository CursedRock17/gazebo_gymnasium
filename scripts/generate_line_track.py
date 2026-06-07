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

"""Generate `models/line_track/model.sdf` as a closed rectangular loop with rounded corners.

SDF has no torus / annulus primitive, so the curved corners are approximated
by a fan of small rotated `<box>` segments. The straight edges are four
long boxes. Every piece is a separate `<visual>` inside one static link —
no collision, the line is purely a vision target.

Rover-scale defaults (the rover is small — ~16 cm between wheel centers):
    outer track footprint:    3 m × 2 m
    corner radius:            0.4 m
    track width:              0.08 m (about 0.4× rover width)
    height:                   0.002 m (thin strip, sits at z=0.001 m)
    segments per corner:      12 (7.5°/step — gaps invisible at strip width)

Tweak the constants at the top of the file and re-run to retune.

Run: ./venv/bin/python scripts/generate_line_track.py
"""

import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACK_DIR = (PROJECT_ROOT / "gazebo_gymnasium_examples" / "gazebo_gymnasium_resources"
             / "models" / "line_track")
SDF_PATH = TRACK_DIR / "model.sdf"


# Track geometry — all in meters.
HALF_LEN_X = 1.5     # half of the outer X extent (so full loop spans 3 m)
HALF_LEN_Y = 1.0     # half of the outer Y extent (so full loop spans 2 m)
CORNER_RADIUS = 0.4
TRACK_WIDTH = 0.08
TRACK_HEIGHT = 0.002
TRACK_Z = 0.001      # bottom face sits just above z=0 to dodge z-fighting
SEGMENTS_PER_CORNER = 12


# Black material — high contrast against the line_follower world's light
# gray ground plane (which uses ambient/diffuse 0.85).
MATERIAL = """      <material>
        <ambient>0.02 0.02 0.02 1</ambient>
        <diffuse>0.02 0.02 0.02 1</diffuse>
        <specular>0.0 0.0 0.0 1</specular>
      </material>"""


def _visual_box(name: str, pose: tuple[float, float, float, float, float, float],
                size: tuple[float, float, float]) -> str:
    x, y, z, roll, pitch, yaw = pose
    sx, sy, sz = size
    return (
        f"      <visual name='{name}'>\n"
        f"        <pose>{x:.6f} {y:.6f} {z:.6f} {roll:.6f} {pitch:.6f} {yaw:.6f}</pose>\n"
        f"        <geometry>\n"
        f"          <box>\n"
        f"            <size>{sx:.6f} {sy:.6f} {sz:.6f}</size>\n"
        f"          </box>\n"
        f"        </geometry>\n"
        f"{MATERIAL}\n"
        f"      </visual>"
    )


def build_straights() -> list[str]:
    """Build the four straight edges of the loop.

    The straights stop at the corner-arc tangent points (±(HALF_LEN_X-r)
    and ±(HALF_LEN_Y-r)). Each is a single long box.
    """
    visuals = []

    straight_x_len = 2 * (HALF_LEN_X - CORNER_RADIUS)
    straight_y_len = 2 * (HALF_LEN_Y - CORNER_RADIUS)

    # Top + bottom: long axis along X.
    visuals.append(_visual_box(
        name="straight_top",
        pose=(0.0, HALF_LEN_Y, TRACK_Z, 0.0, 0.0, 0.0),
        size=(straight_x_len, TRACK_WIDTH, TRACK_HEIGHT),
    ))
    visuals.append(_visual_box(
        name="straight_bottom",
        pose=(0.0, -HALF_LEN_Y, TRACK_Z, 0.0, 0.0, 0.0),
        size=(straight_x_len, TRACK_WIDTH, TRACK_HEIGHT),
    ))
    # Left + right: long axis along Y (swap sx/sy of the box).
    visuals.append(_visual_box(
        name="straight_right",
        pose=(HALF_LEN_X, 0.0, TRACK_Z, 0.0, 0.0, 0.0),
        size=(TRACK_WIDTH, straight_y_len, TRACK_HEIGHT),
    ))
    visuals.append(_visual_box(
        name="straight_left",
        pose=(-HALF_LEN_X, 0.0, TRACK_Z, 0.0, 0.0, 0.0),
        size=(TRACK_WIDTH, straight_y_len, TRACK_HEIGHT),
    ))
    return visuals


def build_corner(name_prefix: str,
                 center: tuple[float, float],
                 start_angle: float) -> list[str]:
    """Build a 90° arc made of small rotated boxes tangent to the circle.

    Each segment is a thin box centered at angle θ_i on a circle of
    radius CORNER_RADIUS around `center`. Box-local +X is rotated to align
    with the tangent at θ_i (which is θ_i + π/2). Segment length is the
    arc length per step, scaled up slightly so adjacent segments overlap
    instead of leaving visible gaps.
    """
    visuals = []
    cx, cy = center
    step = (math.pi / 2) / SEGMENTS_PER_CORNER
    # Each segment's length = arc-length per step + a 5% overlap fudge.
    seg_length = CORNER_RADIUS * step * 1.05

    for i in range(SEGMENTS_PER_CORNER):
        theta = start_angle + (i + 0.5) * step
        px = cx + CORNER_RADIUS * math.cos(theta)
        py = cy + CORNER_RADIUS * math.sin(theta)
        yaw = theta + math.pi / 2  # tangent angle
        visuals.append(_visual_box(
            name=f"{name_prefix}_{i:02d}",
            pose=(px, py, TRACK_Z, 0.0, 0.0, yaw),
            size=(seg_length, TRACK_WIDTH, TRACK_HEIGHT),
        ))
    return visuals


def build_all_visuals() -> list[str]:
    visuals = build_straights()

    # Corner centers are inset by CORNER_RADIUS from the outer rectangle's
    # corners. Each corner's arc spans 90° starting at the angle below
    # (measured from the arc center, +X = 0°, CCW positive).
    inset_x = HALF_LEN_X - CORNER_RADIUS
    inset_y = HALF_LEN_Y - CORNER_RADIUS
    corners = [
        # (name, center_xy, start_angle_radians)
        ("corner_tr", (inset_x, inset_y), 0.0),
        ("corner_tl", (-inset_x, inset_y), math.pi / 2),
        ("corner_bl", (-inset_x, -inset_y), math.pi),
        ("corner_br", (inset_x, -inset_y), 3 * math.pi / 2),
    ]
    for name, center, start_angle in corners:
        visuals.extend(build_corner(name, center, start_angle))
    return visuals


def main() -> int:
    visuals = build_all_visuals()
    visual_block = "\n".join(visuals)

    sdf = f"""<?xml version="1.0"?>
<sdf version="1.8">
  <!-- Auto-generated by scripts/generate_line_track.py. Hand edits will be
       lost on regeneration; tune the constants at the top of the script
       and re-run instead. -->
  <model name="line_track">
    <static>true</static>
    <link name="track">
      <!-- One static link holds every track segment as a <visual>. No
           <collision> blocks: the line is a vision target only — the
           rover should never feel friction against it. -->
{visual_block}
    </link>
  </model>
</sdf>
"""

    SDF_PATH.write_text(sdf)
    n_corners = SEGMENTS_PER_CORNER * 4
    print(f"Wrote {SDF_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  4 straights + {n_corners} corner segments = {4 + n_corners} visuals.")
    print(f"  Outer footprint: {HALF_LEN_X * 2} m × {HALF_LEN_Y * 2} m, "
          f"corner radius {CORNER_RADIUS} m, track width {TRACK_WIDTH} m.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
