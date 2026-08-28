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
"""Validate that the rover in a run is the rover we think it is, and drives.

Three checks, all measured against the simulation that actually ran rather
than against the source tree:

1. **camera** -- every agent's camera is pitched 45 degrees down.
2. **forward** -- the rover advances, instead of creeping or reversing.
3. **speed** -- it does so inside the band the physical rover can execute,
   neither stalled nor moving faster than its own wheels could carry it.

Episode return cannot see any of this. The centering term pays out whether or
not the rover moves; the forward term is computed from the *commanded* action,
not from motion that happened; and nothing in the reward looks at the camera
mount at all. So this reads ground truth: each agent's model carries a gz-sim
OdometryPublisher (GAZEBO_GYM_ODOM=1, see inprocess_vec_env), and the camera
pose comes from the world SDF the server was handed.

    python training_scripts/validate_rover.py --model model.zip --n-agents 4

Exits non-zero when any check fails on any agent, so it works as a gate in a
sweep and not only as a report.
"""

import argparse
import os
from pathlib import Path
import sys
import threading

import numpy as np

# Must be set before the env builds its world SDF.
os.environ.setdefault("GAZEBO_GYM_ODOM", "1")

from gazebo_gymnasium_bridge.envs import make_inprocess  # noqa: E402
from gazebo_gymnasium_bridge.envs import wrap_for_observations  # noqa: E402
from gazebo_gymnasium_bridge.envs.agent_spec import _LF_WHEEL_RADIUS  # noqa: E402
from gazebo_gymnasium_bridge.envs.agent_spec import _LF_WHEEL_SPEED  # noqa: E402
from gazebo_gymnasium_bridge.envs.agent_spec import _LF_WHEEL_SPEED_MIN  # noqa: E402
from gazebo_gymnasium_bridge.envs.agent_spec import _LF_WHEEL_SPEED_POP  # noqa: E402
from gazebo_gymnasium_bridge.envs.agent_spec import register_track_randomized  # noqa: E402
from gazebo_gymnasium_bridge.envs.agent_spec import TRACK_SHAPE_MODES  # noqa: E402
from gazebo_gymnasium_bridge.envs.inprocess_vec_env import camera_pitch_degrees  # noqa: E402
from gazebo_gymnasium_bridge.envs.inprocess_vec_env import odom_enabled  # noqa: E402

V_MAX_MPS = _LF_WHEEL_SPEED * _LF_WHEEL_RADIUS  # 0.51
V_MIN_MPS = _LF_WHEEL_SPEED_MIN * _LF_WHEEL_RADIUS  # 0.10
V_POP_MPS = _LF_WHEEL_SPEED_POP * _LF_WHEEL_RADIUS  # 1.00

# An episode reset teleports the chassis back to its spawn pose, which the
# odometry stream reports as one enormous displacement between consecutive
# samples -- metres in ~16 ms, i.e. hundreds of m/s. Differencing straight
# through that inflates both path length and peak speed (measured: a peak of
# 88 m/s and a mean ground speed above the wheels' own 0.51 m/s full scale).
# Anything implying more than this is a discontinuity, not motion, and gets
# skipped and counted. Well clear of any real over-speed, which tops out
# around the 1.0 m/s caster-pop limit.
_TELEPORT_MPS = 3.0 * V_POP_MPS
# Samples closer together than half the nominal odometry period carry too
# little displacement to divide by.
_MIN_SAMPLE_DT_S = 0.5 / 60.0


# --------------------------------------------------------------------------- #
# Checks 2 and 3: motion, from the simulator's own odometry
# --------------------------------------------------------------------------- #


class OdomTap:
    """Accumulate per-agent odometry off the gz-transport callback threads."""

    #: Body-frame axis the chassis advances along, and its sign. rover_bare's
    #: base_link points along Y, not the X a robot_base_frame twist is usually
    #: read on -- measured, not assumed: full throttle gives a mean twist of
    #: (x=0.003, y=0.481, z=0) m/s.
    FORWARD_AXES = {"x": (0, 1.0), "-x": (0, -1.0), "y": (1, 1.0), "-y": (1, -1.0)}

    def __init__(self, prefix, n_agents, forward_axis="y"):
        from gz.msgs10.odometry_pb2 import Odometry
        from gz.transport13 import Node

        self.n_agents = n_agents
        self._axis, self._sign = self.FORWARD_AXES[forward_axis]
        self._lock = threading.Lock()
        self._first = [None] * n_agents
        self._last = [None] * n_agents
        self._t0 = [None] * n_agents
        self._t1 = [None] * n_agents
        self.path_len = np.zeros(n_agents)  # metres of ground actually covered
        self.fwd = [[] for _ in range(n_agents)]  # body-frame forward speed samples
        self.peak_speed = np.zeros(n_agents)  # fastest instantaneous ground speed
        self.fast_samples = np.zeros(n_agents, dtype=int)  # samples above V_POP
        self.jumps = np.zeros(n_agents, dtype=int)  # reset teleports, skipped
        self.samples = np.zeros(n_agents, dtype=int)
        self._node = Node()
        self._topics = [f"{prefix}/odom_{i}" for i in range(n_agents)]
        for i, topic in enumerate(self._topics):
            if not self._node.subscribe(Odometry, topic, self._make_cb(i)):
                raise RuntimeError(f"could not subscribe to {topic!r}")

    def _make_cb(self, i):
        def _cb(msg):
            try:
                p = msg.pose.position
                xy = (p.x, p.y)
                t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
                lin = msg.twist.linear
                v = self._sign * float((lin.x, lin.y, lin.z)[self._axis])
                with self._lock:
                    if self._first[i] is None:
                        self._first[i], self._t0[i] = xy, t
                    else:
                        prev, dt = self._last[i], t - self._t1[i]
                        step = float(np.hypot(xy[0] - prev[0], xy[1] - prev[1]))
                        if dt >= _MIN_SAMPLE_DT_S:
                            inst = step / dt
                            if inst > _TELEPORT_MPS:
                                # A reset, not motion. The teleport corrupts
                                # the twist for that sample too, not just the
                                # pose difference, so it is dropped from EVERY
                                # accumulator rather than only from the
                                # pose-derived ones.
                                self.jumps[i] += 1
                            else:
                                self.path_len[i] += step
                                self.peak_speed[i] = max(self.peak_speed[i], inst)
                                self.fast_samples[i] += inst > V_POP_MPS
                                # Twist is in the robot base frame, so this is
                                # signed forward speed: negative is reversing.
                                self.fwd[i].append(v)
                                self.samples[i] += 1
                    self._last[i], self._t1[i] = xy, t
            except Exception:  # noqa: B902 - never raise into a transport thread
                pass

        return _cb

    def snapshot(self):
        """Per-agent (net, path, ground, forward, peak, fast%, rev%, msgs)."""
        with self._lock:
            n = self.n_agents
            # `net` spans the whole run including any resets, so it only
            # means "distance from start" for a run that never reset.
            net, elapsed = np.zeros(n), np.zeros(n)
            for i in range(n):
                if self._first[i] is not None and self._last[i] is not None:
                    a, b = self._first[i], self._last[i]
                    net[i] = float(np.hypot(b[0] - a[0], b[1] - a[1]))
                    elapsed[i] = max(self._t1[i] - self._t0[i], 0.0)
            seen = np.maximum(self.samples, 1).astype(float)

            def _per(x, d):
                return np.divide(x, d, out=np.zeros(n), where=d > 0)

            # Median, not mean: a single residual spike at a reset boundary
            # moves a mean of ~900 samples by more than the whole signal
            # (measured: median 0.505 m/s against a mean of 0.033 on the same
            # run). The median says what the rover does almost all the time,
            # which is the question being asked.
            fwd_med = np.array([np.median(f) if f else 0.0 for f in self.fwd])
            rev = np.array([float(np.mean(np.asarray(f) < 0.0)) if f else 0.0 for f in self.fwd])
            # Ground speed assumes nothing about which way the chassis faces;
            # forward speed does, and is what separates driving the track from
            # being dragged around it backwards.
            return {
                "net": net,
                "path": self.path_len.copy(),
                "ground": _per(self.path_len, elapsed),
                "forward": fwd_med,
                "peak": self.peak_speed.copy(),
                "fast_frac": _per(self.fast_samples.astype(float), seen),
                "rev_frac": rev,
                "jumps": self.jumps.copy(),
                "samples": self.samples.copy(),
            }

    def close(self):
        for topic in self._topics:
            try:
                self._node.unsubscribe(topic)
            except Exception:  # noqa: B902
                pass
        self._node = None


# --------------------------------------------------------------------------- #


def _build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", default="line_follower")
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--model", default=None, help="policy .zip; omitted = full-throttle forward")
    ap.add_argument("--frame-stack", type=int, default=4, help="must match training")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--track-shapes",
        default="off",
        choices=TRACK_SHAPE_MODES,
        help="track-shape/spawn randomization, matching train.py/sweep.py",
    )
    ap.add_argument("--camera-pitch", type=float, default=45.0, help="expected, degrees down")
    ap.add_argument("--camera-tol", type=float, default=0.5, help="degrees")
    ap.add_argument(
        "--forward-axis",
        default="y",
        choices=sorted(OdomTap.FORWARD_AXES),
        help="base_link axis the chassis advances along (rover_bare: +y)",
    )
    ap.add_argument("--min-path", type=float, default=1.0, help="metres each agent must cover")
    ap.add_argument(
        "--min-forward",
        type=float,
        default=0.5 * V_MIN_MPS,
        help="median body-frame forward speed (m/s) each agent must hold",
    )
    ap.add_argument(
        "--max-reversing",
        type=float,
        default=0.10,
        help="largest share of samples an agent may spend going backwards",
    )
    ap.add_argument(
        "--max-speed",
        type=float,
        default=V_POP_MPS,
        help="instantaneous ground speed (m/s) above which motion is unphysical",
    )
    ap.add_argument(
        "--json-out", default=None, help="also write the per-agent numbers here as JSON"
    )
    return ap


def _drive(args, agent):
    """Roll out `--steps` and return (odometry snapshot, camera pitches).

    The camera read happens here, before ``close()``: the env deletes its
    generated world SDF on teardown, so there is nothing left to parse
    afterwards.
    """
    vec = make_inprocess(agent, n_agents=args.n_agents, seed=args.seed)
    pitches = camera_pitch_degrees(vec._world_path, f"{agent}_")
    tap = None
    try:
        tap = OdomTap(vec._topic_prefix, args.n_agents, args.forward_axis)
        policy = None
        if args.model:
            from stable_baselines3 import PPO

            policy = PPO.load(str(Path(args.model)))
            vec, _ = wrap_for_observations(vec, frame_stack=args.frame_stack)
        obs = vec.reset()
        for _ in range(args.steps):
            if policy is None:
                # No policy: command both wheels flat out. The control case --
                # if THIS doesn't move the rover, the fault is in the actuation
                # path rather than in anything a policy learned.
                action = np.ones((args.n_agents, 2), dtype=np.float32)
            else:
                action, _ = policy.predict(obs, deterministic=True)
            obs, _rew, _dones, _info = vec.step(action)
        return tap.snapshot(), pitches
    finally:
        if tap is not None:
            tap.close()
        vec.close()


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if not odom_enabled():
        print("GAZEBO_GYM_ODOM is not enabled; nothing to subscribe to.", file=sys.stderr)
        return 2

    agent = register_track_randomized(args.agent, args.track_shapes)
    snap, pitches = _drive(args, agent)
    if not snap["samples"].any():
        print("no odometry messages received on any agent topic", file=sys.stderr)
        return 2

    n = args.n_agents
    src = Path(args.model).name if args.model else "full-throttle (no policy)"
    print(f"\n{agent}: {args.steps} steps, {n} agents, driven by {src}")

    # ---- check 1: camera -------------------------------------------------- #
    cam_ok = np.zeros(n, dtype=bool)
    for i in range(n):
        deg = pitches.get(f"{agent}_{i}")
        cam_ok[i] = deg is not None and abs(deg - args.camera_pitch) <= args.camera_tol
    seen = sorted({round(v, 3) for v in pitches.values()})
    print(
        f"\ncamera  pitch {seen} deg down "
        f"(want {args.camera_pitch} +/- {args.camera_tol}) -> "
        f"{int(cam_ok.sum())}/{n} {'PASS' if cam_ok.all() else 'FAIL'}"
    )

    # ---- checks 2 and 3: motion ------------------------------------------- #
    fwd_ok = (
        (snap["path"] >= args.min_path)
        & (snap["forward"] >= args.min_forward)
        & (snap["rev_frac"] <= args.max_reversing)
    )
    # Too fast is a physical-plausibility bar, not a preference: above the
    # caster-pop speed the wheels leave the ground and odometry stops meaning
    # anything. Too slow is the creeping failure this spec exists to prevent.
    speed_ok = (snap["peak"] <= args.max_speed) & (snap["ground"] >= args.min_forward)

    print(
        f"\n{'agent':>5} {'cam deg':>8} {'path m':>8} {'net m':>8} {'ground':>7} "
        f"{'fwd m/s':>8} {'peak':>7} {'rev %':>6} {'resets':>7} {'msgs':>6}"
        f"  camera forward speed"
    )
    for i in range(n):
        deg = pitches.get(f"{agent}_{i}")
        print(
            f"{i:>5} {('--' if deg is None else f'{deg:.2f}'):>8} "
            f"{snap['path'][i]:>8.3f} {snap['net'][i]:>8.3f} {snap['ground'][i]:>7.3f} "
            f"{snap['forward'][i]:>8.3f} {snap['peak'][i]:>7.3f} "
            f"{100 * snap['rev_frac'][i]:>6.1f} {snap['jumps'][i]:>7} "
            f"{snap['samples'][i]:>6}  "
            f"{'PASS  ' if cam_ok[i] else 'FAIL  '} "
            f"{'PASS   ' if fwd_ok[i] else 'FAIL   '} "
            f"{'PASS' if speed_ok[i] else 'FAIL'}"
        )

    print(
        f"\nbands: forward >= {args.min_forward:.3f} m/s, path >= {args.min_path} m, "
        f"reversing <= {100 * args.max_reversing:.0f}%, peak <= {args.max_speed:.2f} m/s"
    )
    print(
        f"       (wheels can deliver {V_MIN_MPS:.2f} to {V_MAX_MPS:.2f} m/s; "
        f"{V_POP_MPS:.2f} m/s pops the caster)"
    )
    checks = {"camera": cam_ok, "forward": fwd_ok, "speed": speed_ok}
    for label, arr in checks.items():
        print(f"{label:>8}: {int(arr.sum())}/{n} agents {'PASS' if arr.all() else 'FAIL'}")

    if args.json_out:
        import json

        payload = {k: np.asarray(v).tolist() for k, v in snap.items()}
        payload["camera_pitch_deg"] = [pitches.get(f"{agent}_{i}") for i in range(n)]
        payload["checks"] = {k: v.tolist() for k, v in checks.items()}
        payload["model"] = args.model
        Path(args.json_out).write_text(json.dumps(payload, indent=2))

    return 0 if all(a.all() for a in checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
