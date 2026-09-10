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
"""The deployment contract: what a `rover_line` policy sees, and what it commands.

Kept in one file because both sides depend on it. The simulator builds
observations from this module and the rover's runtime must build the same
thing; a mismatch does not fail loudly at the boundary, it fails on hardware,
and it fails in a way that still looks plausible.

**This file is vendored, not imported, by the rover repo.** The deployment
script has to run on a laptop with only numpy, opencv and SB3, so it cannot
import the simulator package. An identical copy lives at
`results/for_rover_repo/rover_line_contract.py`, and
`test_rover_line.py::test_vendored_contract_is_identical` asserts the two are
byte-for-byte the same, which makes drift a test failure rather than something
a `--self-check` catches only when somebody remembers to run it.

Nothing simulator-specific belongs here. The drive-sign negation is a Gazebo
frame convention, not a property of the rover, so it stays in `rover_line.py`:
this module speaks the hardware's units throughout.

Depends on numpy alone, deliberately.
"""

import numpy as np

SCHEMA_VERSION = 2
# Bump when the observation or action layout changes. A runtime that knows only
# an older API must refuse a newer policy rather than feed it a wrongly shaped
# array -- silently mis-slicing floats is the failure this guards.
#
# v2 rebuilt the contract on three full-height camera bands carrying a centroid
# in BOTH axes, a stop-capable speed command, and jerk priced in the reward
# instead of a low-pass filter on the command. v1's four half-frame bands with
# derived heading/curvature, its forward-only speed floor, and its filter-state
# observation are gone.
MODEL_API = 2

# --- The physical rover. ---
WHEEL_RADIUS = 0.035  # m
WHEEL_SEP = 0.099  # m, between the two wheel link poses
CONTROL_HZ = 10.0  # must match the firmware's DEFAULT_CMD_RATE_HZ

# Speed command. `forward` spans 0 to 2*CRUISE_RAD_S, so the midpoint of the
# action range IS the cruise speed and a=-1 is a full stop -- unlike the
# previous forward-only map, which could not stop at all. `steering` is added
# to one wheel and subtracted from the other, so a wheel may reverse and the
# rover can pivot.
CRUISE_RAD_S = 3.0
MAX_STEERING_RAD_S = 4.0
# Nothing may command a wheel past this; it is also the observation's scale.
MAX_WHEEL_RAD_S = 10.0

IMAGE = (48, 64, 3)  # H, W, C
DARK = 60  # a pixel is "line" when every channel is below this
# A band needs at least this many line pixels to count as seeing the track.
# Without it a single dark pixel -- a caster edge, a shadow, sensor noise --
# defines that band's centroid, which is a confident wrong answer rather than
# an honest "nothing here". Measured on real frames: a genuine band carries
# 154-199 dark pixels of 384, so this rejects noise with a 25x margin.
MIN_LINE_PIXELS = 6

# Bands as fractions of FULL frame height, nearest first. The near band is the
# one a proportional controller would steer on; the upper two are lookahead,
# and exist so the policy can see a corner before it arrives.
BANDS = ((0.85, 1.0), (0.55, 0.85), (0.25, 0.55))
N_BANDS = len(BANDS)

N_FEATURES = 3 * N_BANDS  # per band: centroid x, centroid y, visible
OBS_LEN = N_FEATURES + 2  # + left/right wheel speed
ACTION_LEN = 2  # [forward, steering], both in [-1, 1]

OBS_LAYOUT = (
    "0-2 near band (centroid x, centroid y, visible); 3-5 middle band; "
    "6-8 far band; 9-10 left/right wheel speed / MAX_WHEEL_RAD_S"
)
ACTION_LAYOUT = (
    f"0 forward in [-1, 1] -> 0..{2 * CRUISE_RAD_S} rad/s (-1 stops); "
    f"1 steering in [-1, 1] -> +/-{MAX_STEERING_RAD_S} rad/s"
)

# --- Reward weights. ---
ALIGNMENT_FLOOR = 0.25  # share of the motion reward paid regardless of centring
LINE_LOST_PENALTY = -1.0  # per step, while no line is visible
SMOOTHNESS_COST = 0.02  # per unit squared change in action
STEP_COST = 0.01  # per step, so dawdling is never free
FAILURE_PENALTY = -5.0  # once, when the episode ends in failure

# Real encoder resolution (wmala2/rover-firmware, 500 RPM profile). Counts are
# integers, so reported speed comes in steps of one count per control step.
ENCODER_CPR_WHEEL = 680
WHEEL_CIRCUM_M = 0.2199
ENCODER_QUANTUM_RAD_S = (WHEEL_CIRCUM_M / ENCODER_CPR_WHEEL) / WHEEL_RADIUS / (1.0 / CONTROL_HZ)


def band_centroid(band_rows):
    """Centroid of the line within one horizontal band.

    :param band_rows: the band's slice of the frame, HxWx3 uint8.
    :return: (x in [-1, 1] about the frame centre, mean row index, visible),
        or (0.0, 0.0, False) when the band holds no line.
    """
    mask = band_rows.max(axis=2) < DARK
    weights = mask.sum(axis=0)
    if weights.sum() < MIN_LINE_PIXELS:
        return 0.0, 0.0, False  # noise, not tape
    width = band_rows.shape[1]
    centre = (width - 1) / 2.0
    x = (float(np.arange(width) @ weights) / float(weights.sum()) - centre) / centre
    rows = np.flatnonzero(mask.any(axis=1))
    return float(np.clip(x, -1.0, 1.0)), float(rows.mean()), True


def line_features(img):
    """Three bands of (centroid x, centroid y, visible), nearest band first.

    Both axes are carried, not just the horizontal one: where the line sits
    vertically inside a band says how far ahead it is, which separates a
    corner arriving from a corner already under the wheels.

    :param img: the policy-visible camera frame, HxWx3 uint8.
    :return: float32 array of shape (N_BANDS, 3), every element in [-1, 1].
    """
    height = img.shape[0]
    out = np.zeros((N_BANDS, 3), dtype=np.float32)
    for i, (start, end) in enumerate(BANDS):
        top, bottom = int(start * height), int(end * height)
        x, row, visible = band_centroid(img[top:bottom])
        if visible:
            y = 2.0 * (top + row) / (height - 1) - 1.0
            out[i] = [x, np.clip(y, -1.0, 1.0), 1.0]
    return out


def wheel_speeds(wheel_rad_s):
    """Measured wheel speeds (rad/s) -> the normalized pair the policy observes.

    Snapped to the encoder's resolution first: a real reading is quantized
    because counts are integers, but the wall-clock interval it is divided by
    is never exactly the control period, so the raw value drifts off the grid
    the policy trained on.

    :param wheel_rad_s: measured (left, right) in rad/s, positive forward.
    :return: float32 pair in [-1, 1].
    """
    rad_s = np.asarray(wheel_rad_s, dtype=np.float32)
    rad_s = np.round(rad_s / ENCODER_QUANTUM_RAD_S) * ENCODER_QUANTUM_RAD_S
    return np.clip(rad_s / MAX_WHEEL_RAD_S, -1.0, 1.0).astype(np.float32)


def observation_from_sensors(img, wheel_rad_s=None):
    """The whole policy observation: OBS_LEN floats, every one in [-1, 1].

    The same function runs on a rendered frame and on a real JPEG, which is
    the point of extracting features rather than learning them.

    :param img: the policy-visible camera frame, HxWx3 uint8.
    :param wheel_rad_s: measured (left, right) wheel speed in rad/s; None reads
        as stopped wheels, which is what a caller with no encoder link sees.
    :return: float32 vector of length OBS_LEN.
    """
    speeds = wheel_speeds(wheel_rad_s) if wheel_rad_s is not None else np.zeros(2, np.float32)
    return np.concatenate([line_features(img).ravel(), speeds]).astype(np.float32)


def wheel_targets(action):
    """Action -> (left, right) wheel speed in rad/s, positive forward on both.

    ``np.resize`` rather than plain indexing because Gazebo's HarnessCore
    discovers a spec's actuated joints by probing this with the SCALARS 0 and
    1, and swallows whatever it raises. Indexing ``a[1]`` on a one-element
    probe raises IndexError, the wheels are then never registered as actuated,
    and every command is silently dropped -- a rover that reports perfect
    commands and does not move.
    """
    a = np.resize(np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0), 2)
    forward = CRUISE_RAD_S * (a[0] + 1.0)
    steering = MAX_STEERING_RAD_S * a[1]
    return float(forward - steering), float(forward + steering)


def line_visible(img):
    """True when the nearest band holds enough line pixels to be track."""
    height = img.shape[0]
    top, bottom = int(BANDS[0][0] * height), int(BANDS[0][1] * height)
    return band_centroid(img[top:bottom])[2]


def step_reward(img, action, wheel_rad_s, prev_action):
    """Forward progress, scaled by how well the line is centred.

    Progress is measured, not commanded: it is the mean of the two encoder
    readings against the cruise speed, so a command the motors did not execute
    earns nothing. It is normalized to 1.0 at cruise and clipped to [-2, 1],
    which caps the gain from simply going faster and leaves centring as the
    way to earn the rest.

    Losing the line pays a flat penalty rather than zero, because zero is also
    what standing still on a visible line pays, and those should not be worth
    the same. Jerk is charged directly, which is why this task needs no
    low-pass filter on the command.

    :param img: the raw camera frame.
    :param action: the action just applied.
    :param wheel_rad_s: measured (left, right) wheel speed in rad/s.
    :param prev_action: the action applied on the previous step.
    :return: this step's reward.
    """
    action = np.resize(np.asarray(action, dtype=np.float32).ravel(), ACTION_LEN)
    prev_action = np.resize(np.asarray(prev_action, dtype=np.float32).ravel(), ACTION_LEN)
    feats = line_features(img)
    visible = bool(feats[0][2])

    measured = float(np.mean(np.asarray(wheel_rad_s, dtype=np.float32)))
    progress = float(np.clip(measured / CRUISE_RAD_S, -2.0, 1.0))
    if visible:
        alignment = 1.0 - abs(float(feats[0][0]))
        motion = progress * (ALIGNMENT_FLOOR + (1.0 - ALIGNMENT_FLOOR) * alignment)
    else:
        motion = LINE_LOST_PENALTY

    smoothness = SMOOTHNESS_COST * float(np.sum((action - prev_action) ** 2))
    return motion - smoothness - STEP_COST
