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
"""Line-follower track-shape geometry, shared between build time and runtime.

`scripts/generate_line_tracks.py` imports the builder functions here to
write the actual `<visual>` box SDFs once, ahead of time. `inprocess_vec_env.py`
imports `PRESETS` again at world-build time to pick a valid spawn point (a
random segment's own pose, tangent-aligned by construction) for whichever
shape a given agent draws -- the same geometry, used for two different
purposes, kept in one place rather than duplicated.

Segment format: (pose, size) tuples, pose=(x, y, z, roll, pitch, yaw), size=
(sx, sy, sz), in the shape's own LOCAL frame (centered at the origin) --
`_build_world` offsets by each agent's own x_spacing position separately.
"""

import math

import numpy as np

TRACK_WIDTH = 0.08
TRACK_HEIGHT = 0.002
TRACK_Z = 0.001


def _arc_segments(center, radius, start_angle, span, n_segments, width=TRACK_WIDTH):
    """A fan of small rotated boxes approximating an arc of any span."""
    cx, cy = center
    step = span / n_segments
    seg_length = abs(radius * step) * 1.08
    segs = []
    for i in range(n_segments):
        theta = start_angle + (i + 0.5) * step
        px = cx + radius * math.cos(theta)
        py = cy + radius * math.sin(theta)
        yaw = theta + math.pi / 2 * (1 if span > 0 else -1)
        segs.append(((px, py, TRACK_Z, 0.0, 0.0, yaw), (seg_length, width, TRACK_HEIGHT)))
    return segs


def _straight_segment(p1, p2, width=TRACK_WIDTH, overlap=1.03):
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    mid = (p1 + p2) / 2
    length = float(np.linalg.norm(p2 - p1)) * overlap
    yaw = float(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    return ((mid[0], mid[1], TRACK_Z, 0.0, 0.0, yaw), (length, width, TRACK_HEIGHT))


def build_rounded_polygon(vertices, radius, segments_per_corner=10, width=TRACK_WIDTH):
    """Closed loop through `vertices`, each corner rounded to `radius` (convex only)."""
    n = len(vertices)

    def tangent_point(i):
        v = np.asarray(vertices[i], dtype=float)
        prev_v = np.asarray(vertices[(i - 1) % n], dtype=float)
        next_v = np.asarray(vertices[(i + 1) % n], dtype=float)
        d1n = (prev_v - v) / np.linalg.norm(prev_v - v)
        d2n = (next_v - v) / np.linalg.norm(next_v - v)
        interior = math.acos(np.clip(np.dot(d1n, d2n), -1.0, 1.0))
        t = radius / math.tan(interior / 2)
        return v, d1n, d2n, interior, v + d1n * t, v + d2n * t

    segs = []
    for i in range(n):
        v, d1n, d2n, interior, tangent_to_prev, tangent_to_next = tangent_point(i)
        bisector = d1n + d2n
        bisector = bisector / np.linalg.norm(bisector)
        arc_center = v + bisector * (radius / math.sin(interior / 2))
        start_angle = math.atan2(
            tangent_to_next[1] - arc_center[1], tangent_to_next[0] - arc_center[0]
        )
        end_angle = math.atan2(
            tangent_to_prev[1] - arc_center[1], tangent_to_prev[0] - arc_center[0]
        )
        span = (end_angle - start_angle + math.pi) % (2 * math.pi) - math.pi
        segs.extend(
            _arc_segments(arc_center, radius, start_angle, span, segments_per_corner, width)
        )
        _, _, _, _, tangent2_to_prev, _ = tangent_point((i + 1) % n)
        segs.append(_straight_segment(tangent_to_next, tangent2_to_prev, width))
    return segs


def build_circle(radius, n_segments=32, width=TRACK_WIDTH):
    return _arc_segments((0.0, 0.0), radius, 0.0, 2 * math.pi, n_segments, width)


def build_racetrack(straight_length, radius, n_segments_per_end=12, width=TRACK_WIDTH):
    """Stadium/obround shape: two straights + two 180-degree semicircle ends."""
    half = straight_length / 2
    segs = [
        _straight_segment((-half, radius), (half, radius), width),
        _straight_segment((half, -radius), (-half, -radius), width),
    ]
    segs.extend(
        _arc_segments((half, 0.0), radius, -math.pi / 2, math.pi, n_segments_per_end, width)
    )
    segs.extend(
        _arc_segments((-half, 0.0), radius, math.pi / 2, math.pi, n_segments_per_end, width)
    )
    return segs


def preset_rectangle():
    """Matches models/line_track/model.sdf (generate_line_track.py) exactly."""
    return build_rounded_polygon(
        [(1.5, 1.0), (-1.5, 1.0), (-1.5, -1.0), (1.5, -1.0)], radius=0.4, segments_per_corner=12
    )


def preset_square():
    s = 1.2
    return build_rounded_polygon([(s, s), (-s, s), (-s, -s), (s, -s)], radius=0.3)


def preset_triangle():
    r = 1.4
    verts = [
        (r * math.cos(a), r * math.sin(a))
        for a in (math.pi / 2, math.pi / 2 + 2 * math.pi / 3, math.pi / 2 + 4 * math.pi / 3)
    ]
    return build_rounded_polygon(verts, radius=0.28)


def preset_pentagon():
    r = 1.35
    verts = [
        (
            r * math.cos(math.pi / 2 + i * 2 * math.pi / 5),
            r * math.sin(math.pi / 2 + i * 2 * math.pi / 5),
        )
        for i in range(5)
    ]
    return build_rounded_polygon(verts, radius=0.3)


def preset_zigzag():
    # Two real bugs found here via an actual functional smoke test (not
    # just the visual preview, which looked fine at a glance both times):
    #   1. radius=0.12 is smaller than the rover's own ~0.16m wheelbase --
    #      a physically-too-tight turn; the camera (tuned around the
    #      original 0.4m corners) couldn't see the line from a spawn point
    #      right on that corner, producing a blank frame.
    #   2. Naively bumping the radius to 0.2 then broke geometrically: the
    #      original vertex layout's bottom-left transition (zigzag pattern
    #      meeting the rectangle's closing edge) had an unexpectedly sharp
    #      ~15-degree interior angle -- no radius rounds that cleanly, the
    #      tangent-point inset overshoots the adjacent vertex and produces
    #      a self-intersecting loop.
    # Redesigned so the zigzag pattern meets the rectangle's corners at
    # gentle angles (~145-157 degrees) and the sharpest interior angles
    # (the zigzag peaks/valleys themselves, ~71 degrees) stay well clear of
    # overshooting their much-longer adjacent edges at radius=0.2.
    verts = [
        (-1.4, -0.9),
        (1.4, -0.9),
        (1.4, 0.2),
        (0.9, 0.9),
        (0.4, 0.2),
        (-0.1, 0.9),
        (-0.6, 0.2),
        (-1.1, 0.9),
        (-1.4, 0.2),
    ]
    return build_rounded_polygon(verts, radius=0.2, segments_per_corner=6)


def preset_circle():
    return build_circle(radius=1.3)


def preset_racetrack():
    return build_racetrack(straight_length=1.8, radius=0.75)


# name -> (segment-builder, model-directory-name). "rectangle" reuses the
# pre-existing models/line_track/ directory (generate_line_track.py), not a
# line_track_rectangle/ duplicate; the other six are line_track_<name>/,
# written by scripts/generate_line_tracks.py.
def centerline(segments, spacing=0.02):
    """Ordered centreline polyline through a track's own segment boxes.

    Every builder here lays the track as boxes end to end along the path, so
    the centreline is already implicit in the geometry and does not have to be
    recovered from a mesh: each box contributes points along its own axis, and
    the boxes are then chained end to end.

    Chaining is by nearest endpoint rather than by build order, because build
    order is not path order for every shape -- ``build_racetrack`` emits both
    straights before either arc. Adjacent boxes share a joint by construction,
    so the nearest unused endpoint is unambiguous.

    :param segments: what a build_* function returned: ((x, y, z, r, p, yaw),
        (length, width, height)) per box.
    :param spacing: target spacing of the returned points, in metres.
    :return: (N, 2) float array of ordered points, first != last (closed
        implicitly), spaced evenly along the loop.
    """
    strips = []
    for (px, py, _z, _roll, _pitch, yaw), (length, _w, _h) in segments:
        # Interior of each box only: the builders overlap neighbours slightly
        # (1.03x on straights, 1.08x on arcs) so the joints do not gap, and
        # sampling to the very end would zigzag back at every seam. Even
        # resampling below bridges the small holes this leaves.
        half = 0.4 * length
        n = max(2, int(math.ceil(2 * half / spacing)) + 1)
        ts = np.linspace(-half, half, n)
        direction = np.array([math.cos(yaw), math.sin(yaw)])
        strips.append(np.array([px, py]) + ts[:, None] * direction)

    # Greedy chain: from the free end of the path so far, take whichever
    # unused strip starts or ends nearest, flipping it if needed.
    remaining = list(range(1, len(strips)))
    path = [strips[0]]
    tail = strips[0][-1]
    while remaining:
        best, best_d, best_flip = None, np.inf, False
        for idx in remaining:
            for flip in (False, True):
                strip = strips[idx][::-1] if flip else strips[idx]
                d = float(np.linalg.norm(strip[0] - tail))
                if d < best_d:
                    best, best_d, best_flip = idx, d, flip
        strip = strips[best][::-1] if best_flip else strips[best]
        path.append(strip)
        tail = strip[-1]
        remaining.remove(best)

    pts = np.vstack(path)
    # Resample evenly around the closed loop, so arc length is proportional to
    # index and a projection onto it is well conditioned.
    closed = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    count = max(8, int(round(cum[-1] / spacing)))
    target = np.linspace(0.0, cum[-1], count, endpoint=False)
    return np.stack([np.interp(target, cum, closed[:, i]) for i in range(2)], axis=1)


def perimeter(points):
    """Closed-loop length of an ordered centreline, in metres."""
    closed = np.vstack([points, points[0]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


PRESETS = {
    "rectangle": (preset_rectangle, "line_track"),
    "square": (preset_square, "line_track_square"),
    "triangle": (preset_triangle, "line_track_triangle"),
    "pentagon": (preset_pentagon, "line_track_pentagon"),
    "zigzag": (preset_zigzag, "line_track_zigzag"),
    "circle": (preset_circle, "line_track_circle"),
    "racetrack": (preset_racetrack, "line_track_racetrack"),
}
