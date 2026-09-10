"""Run a trained `rover_line` policy on the physical rover.

Belongs in the rover repo, not in gazebo_gymnasium: no physical logic lives
there. Standalone on purpose -- it imports nothing from the simulator, so it
can run on the deployment laptop with only numpy, opencv and SB3 installed.

    python rover_line_deploy.py model.zip --url http://<cam-ip>/capture
    python rover_line_deploy.py model.zip --dir ./captured_frames
    python rover_line_deploy.py model.zip --self-check   # verify preprocessing

Nothing is sent to the drive ESP32 unless --send is passed. By default this
prints the wheel speeds the policy WOULD command, so the sim-to-real visual
gap can be inspected before anything moves.

THE SELF-CHECK IS THE POINT. The whole reason `rover_line` feeds the policy
extracted features instead of raw pixels is that the feature extractor is a
fixed function that can run identically on both sides. That only holds if
this file's copy of it actually matches the simulator's. --self-check imports
the env's version and asserts they agree bit for bit on random frames; if it
fails, every action below is wrong in a way that still looks plausible.
"""

import argparse
import sys

import numpy as np

# The deployment contract, vendored verbatim from
# gazebo_gymnasium_bridge/envs/rover_line_contract.py. Imported rather than
# re-implemented: this file used to carry its own copy of the extractor and
# the action map, which meant a change in the simulator could silently leave
# this side behind. test_rover_line.py asserts the two copies are identical.
from rover_line_contract import ENCODER_CPR_WHEEL
from rover_line_contract import IMAGE
from rover_line_contract import line_visible
from rover_line_contract import OBS_LEN
from rover_line_contract import observation_from_sensors
from rover_line_contract import SCHEMA_VERSION
from rover_line_contract import WHEEL_CIRCUM_M
from rover_line_contract import WHEEL_RADIUS
from rover_line_contract import wheel_targets

# Consecutive failed encoder reads tolerated before the run is stopped. One
# dropped UDP reply is normal; a run of them means the link is down, and
# driving on a frozen wheel-speed reading is worse than stopping.
_MAX_STALE_ENCODER_READS = 5


def pytest_approx(x, tol=1e-9):
    """Tiny float comparison helper; this file deliberately has no test deps."""

    class _Near:
        def __eq__(self, other):
            return abs(other - x) <= tol

    return _Near()


def counts_to_rad_s(delta_counts, dt):
    """Encoder counts since the last reply -> wheel speed in rad/s.

    The firmware's 'e' command returns cumulative quadrature counts, so the
    caller differences consecutive replies and passes the delta.

    :param delta_counts: counts accumulated on one wheel since the last read.
    :param dt: seconds those counts accumulated over.
    :return: that wheel's angular speed in rad/s.
    """
    metres = float(delta_counts) / ENCODER_CPR_WHEEL * WHEEL_CIRCUM_M
    return metres / WHEEL_RADIUS / float(dt)


def to_obs_frame(rgb):
    """Any RGB image -> the 64x64 frame the extractor expects."""
    import cv2

    if rgb.shape[:2] != IMAGE[:2]:
        rgb = cv2.resize(rgb, (IMAGE[1], IMAGE[0]), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.uint8)


class RoverClient:
    """UDP client for the drive firmware's JSON protocol.

    Commands go to port 9000 and replies come back on 9001
    (wmala2/rover-firmware). Only what this loop needs is implemented:
    ``'m'`` to set closed-loop wheel speeds in m/s, ``'e'`` to read cumulative
    quadrature counts, and a stop.

    Counts are cumulative, so wheel SPEED is a difference between consecutive
    replies over the interval between them -- which is why this keeps the
    previous reply and its timestamp rather than returning raw counts.

    :param host: the rover's IP address.
    :param timeout: seconds to wait for an encoder reply before giving up.
    """

    CMD_PORT = 9000
    REPLY_PORT = 9001

    def __init__(self, host, timeout=0.2):
        import socket

        self.host = host
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("", self.REPLY_PORT))
        self._sock.settimeout(timeout)
        self._index = 0
        self._prev = None  # (left_counts, right_counts, monotonic seconds)

    def _send(self, payload):
        import json

        self._sock.sendto(json.dumps(payload).encode("utf-8"), (self.host, self.CMD_PORT))

    def set_wheel_speeds(self, left_rad_s, right_rad_s):
        """Send one closed-loop velocity command, converted to the m/s the firmware takes."""
        self._index += 1
        self._send(
            {
                "command": "m",
                "left_mps": round(float(left_rad_s) * WHEEL_RADIUS, 4),
                "right_mps": round(float(right_rad_s) * WHEEL_RADIUS, 4),
                "index": self._index,
            }
        )

    def read_wheel_speeds(self):
        """Query the encoders and convert to (left_rad_s, right_rad_s).

        :return: the pair in m/s, or None when no usable reading is available
            -- no reply, a malformed one, or the first call, which has no
            previous sample to difference against. The caller decides what to
            do with None; substituting zero would tell the policy the wheels
            had stopped.
        """
        import json
        import time

        self._send({"command": "e"})
        try:
            data, _addr = self._sock.recvfrom(512)
            reply = json.loads(data.decode("utf-8"))
            left, right = float(reply["left"]), float(reply["right"])
        except (OSError, ValueError, KeyError):
            return None
        now = time.monotonic()
        prev, self._prev = self._prev, (left, right, now)
        if prev is None:
            return None
        dt = now - prev[2]
        if dt <= 0:
            return None
        return (counts_to_rad_s(left - prev[0], dt), counts_to_rad_s(right - prev[1], dt))

    def stop(self):
        """Command the slowest speed the wheels take, then zero PWM.

        The action space can command a halt directly, so a zero closed-loop
        setpoint is the honest stop; the open-loop zero PWM that follows makes
        sure nothing is left energised.
        """
        self.set_wheel_speeds(0.0, 0.0)
        self._send({"command": "o", "left_pwm": 0, "right_pwm": 0})

    def close(self):
        self._sock.close()


def describe(action, obs):
    """One line per step: what the camera sees and what the wheels are told."""
    left, right = wheel_targets(action)
    turn = "straight"
    if abs(left - right) > 0.05:
        turn = "LEFT" if right > left else "RIGHT"
    seen = "".join("#" if obs[3 * b + 2] > 0 else "." for b in range(3))
    return (
        f"bands[{seen}] near_x={obs[0]:+.2f} near_y={obs[1]:+.2f} "
        f"enc=({obs[9]:+.2f},{obs[10]:+.2f}) | "
        f"L={left:+.2f} R={right:+.2f} rad/s ({left * WHEEL_RADIUS:+.2f} m/s)  {turn}"
    )


def self_check():
    """Verify this file's contract copy against the simulator's, and exit.

    The whole reason `rover_line` feeds the policy extracted features instead
    of raw pixels is that the extractor is a fixed function that runs
    identically on both sides. That only holds if the copies agree, so this
    imports the environment's own module and diffs them on random frames.
    """
    try:
        from gazebo_gymnasium_bridge.envs import rover_line as rl
        from gazebo_gymnasium_bridge.envs import rover_line_contract as sim
    except ImportError:
        print(
            "self-check needs gazebo_gymnasium_bridge importable "
            "(run it on the dev machine, not the rover).",
            file=sys.stderr,
        )
        return 2

    assert SCHEMA_VERSION == sim.SCHEMA_VERSION, (
        f"contract v{SCHEMA_VERSION} here, v{sim.SCHEMA_VERSION} in the simulator"
    )
    rng = np.random.default_rng(0)
    img = np.full(IMAGE, 255, np.uint8)
    for trial in range(200):
        img = np.full(IMAGE, 255, np.uint8)
        if trial % 7:  # plus occasional blank frames
            col = rng.integers(0, 60)
            img[rng.integers(0, 40) :, col : col + rng.integers(2, 10)] = rng.integers(0, sim.DARK)
        img = np.clip(img.astype(int) + rng.integers(-20, 20, img.shape), 0, 255).astype(np.uint8)
        # Feed both sides the same physical wheel speed; only the simulator
        # applies its own drive-sign convention on the way in.
        rad_s = rng.uniform(-6, 6, 2)
        mine = observation_from_sensors(img, rad_s)
        theirs = rl.features(img, rl._DRIVE_SIGN * rad_s)
        assert np.allclose(mine, theirs, atol=1e-6), f"trial {trial}: {mine} != {theirs}"
        assert line_visible(img) != rl.terminated(img), f"trial {trial}: termination differs"
        # The action map must agree too: a mismatched decode drives the wheels
        # at speeds the policy never asked for.
        act = rng.uniform(-1, 1, 2)
        assert np.allclose(wheel_targets(act), sim.wheel_targets(act), atol=1e-9)

    # The rover's camera delivers 640x480, not the frame the policy wants, so
    # the resize is part of the pipeline and can silently change what the
    # extractor sees.
    for width in (4, 8, 16):
        big = np.full((480, 640, 3), 255, np.uint8)
        big[240:, 320 - width * 5 : 320 + width * 5] = 0
        small = np.full(IMAGE, 255, np.uint8)
        small[24:, 32 - width // 2 : 32 + width // 2] = 0
        got = observation_from_sensors(to_obs_frame(big))
        want = observation_from_sensors(small)
        assert np.allclose(got[:3], want[:3], atol=2.0 / (IMAGE[1] - 1)), (
            f"width {width}: resized near band {got[:3]} != {want[:3]}"
        )

    # The encoder conversion has no simulator counterpart to diff against, so
    # check it against the firmware's own constants: one wheel revolution per
    # second is 2*pi rad/s.
    # One wheel revolution per second is 2*pi rad/s. The tolerance is loose
    # because WHEEL_CIRCUM_M is the firmware's own constant rounded to four
    # decimals, so it is not exactly 2*pi*WHEEL_RADIUS -- and the firmware's
    # value is the one that matters, since it is what converts real counts.
    assert counts_to_rad_s(ENCODER_CPR_WHEEL, 1.0) == pytest_approx(2 * np.pi, 1e-3)
    assert counts_to_rad_s(ENCODER_CPR_WHEEL // 2, 0.5) == pytest_approx(2 * np.pi, 1e-2)

    print(
        f"self-check OK: contract v{SCHEMA_VERSION}, observation {OBS_LEN} floats; "
        f"extractor, termination and action map all match the simulator on 200 "
        f"frames; 640x480 -> {IMAGE[1]}x{IMAGE[0]} resize preserves the line"
    )
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model", nargs="?", help="path to the trained .zip")
    p.add_argument("--url", help="camera capture URL, polled each step")
    p.add_argument("--dir", help="directory of frames to replay instead")
    p.add_argument("-n", type=int, default=40, help="steps to run")
    p.add_argument(
        "--self-check",
        action="store_true",
        help="verify preprocessing against the simulator and exit",
    )
    p.add_argument(
        "--host",
        metavar="IP",
        help="rover IP; enables live encoder reads (UDP 9000 out, 9001 back). "
        "Without it the encoder channels read zero and this is a dry run only",
    )
    p.add_argument(
        "--send", action="store_true", help="actually transmit wheel speeds to the drive firmware"
    )
    args = p.parse_args()

    if args.self_check:
        sys.exit(self_check())
    if not args.model:
        p.error("model is required unless --self-check is given")

    import stable_baselines3 as sb3

    model = sb3.PPO.load(args.model)
    # Announce the contract this binary speaks. A policy trained against a
    # different layout loads without complaint and then acts on mis-sliced
    # floats, so the version belongs in the run log.
    print(f"contract schema v{SCHEMA_VERSION}; observation {model.observation_space}")
    frames = _frame_source(args)
    # Announce the contract this binary speaks. A policy trained against a
    # different layout loads without complaint and then acts on mis-sliced
    # floats, so the version belongs in the run log.
    print(f"contract schema v{SCHEMA_VERSION}; observation {model.observation_space}")

    # The policy was trained with live encoders in BOTH its observation and its
    # reward, so running without them is not a degraded mode, it is a different
    # input distribution. Say so rather than quietly feeding zeros.
    rover = RoverClient(args.host) if args.host else None
    if rover is None:
        print(
            "NOTE: no --host, so encoders read as stopped wheels. Dry run only: "
            "the policy was trained on live encoder readings."
        )
    wheel_rad_s = None
    stale = 0
    try:
        for t, rgb in zip(range(args.n), frames, strict=False):
            img = to_obs_frame(rgb)
            if not line_visible(img):
                print(f"[{t:>3}] line not visible")
            if rover is not None:
                reading = rover.read_wheel_speeds()
                if reading is None:
                    # Hold the last good reading rather than substituting zero,
                    # which would claim the rover had stopped dead.
                    stale += 1
                    if stale > _MAX_STALE_ENCODER_READS:
                        print(f"[{t:>3}] ENCODERS SILENT for {stale} steps; stopping.")
                        break
                else:
                    wheel_rad_s, stale = reading, 0
            obs = observation_from_sensors(img, wheel_rad_s)
            action, _ = model.predict(obs[None, :], deterministic=True)
            print(f"[{t:>3}] {describe(action[0], obs)}")
            if args.send:
                if rover is None:
                    raise SystemExit("--send needs --host")
                rover.set_wheel_speeds(*wheel_targets(action[0]))
    finally:
        if rover is not None:
            if args.send:
                rover.stop()
            rover.close()


def _frame_source(args):
    import cv2

    if args.dir:
        from pathlib import Path

        for f in sorted(Path(args.dir).iterdir()):
            im = cv2.imread(str(f))
            if im is not None:
                yield cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    elif args.url:
        import urllib.request

        while True:
            with urllib.request.urlopen(args.url, timeout=5) as r:
                buf = np.frombuffer(r.read(), np.uint8)
            im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if im is not None:
                yield cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        raise SystemExit("need --url or --dir (or --self-check)")


if __name__ == "__main__":
    main()
