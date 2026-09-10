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
"""Did the rover actually drive a lap? Measured by odometry, not by commands.

`dr_eval.py --solved-distance` derives distance from the spec's own action
map times the timestep, which is open-loop arithmetic: it reports ~18 m for a
rover that never moved, because that is simply what the commands would have
covered. This reads the simulator's ground-truth pose instead.

    python training_scripts/eval_rover_line.py models/rover_line_multi/final_ppo_n4.zip

An episode counts as a lap when the chassis covers LAP_METERS of real ground
without losing the line. Reported alongside it: net displacement, which
should be SMALL on a closed loop -- a rover that drove 9 m in a straight line
covered the distance without driving the track.
"""

import argparse
import os
import threading

import numpy as np

os.environ.setdefault("GAZEBO_GYM_ODOM", "1")

from gazebo_gymnasium_bridge.envs import get_spec  # noqa: E402
from gazebo_gymnasium_bridge.envs import make_inprocess  # noqa: E402
from gazebo_gymnasium_bridge.envs import rover_line as rl  # noqa: E402
from gazebo_gymnasium_bridge.envs import wrap_for_observations  # noqa: E402

_TELEPORT_M = 0.5  # a per-sample jump this large is a reset, not motion


class _Odom:
    """Ground-truth path length, net displacement and SIGNED progress, per agent.

    `path` is arc length and carries no direction, so a rover driving the loop
    backwards accumulates it exactly as fast as one driving it forwards --
    which is how a reversing policy can clear a lap bar. `forward` fixes that:
    each step's displacement is projected onto the direction the camera looks
    (body -Y, see rover_line._DRIVE_SIGN) and summed with its sign, so
    reversing subtracts.
    """

    def __init__(self, prefix, n):
        from gz.msgs10.odometry_pb2 import Odometry
        from gz.transport13 import Node

        self.n = n
        self._lock = threading.Lock()
        self._last = [None] * n
        self._start = [None] * n
        self.path = np.zeros(n)
        self.net = np.zeros(n)
        self.forward = np.zeros(n)
        self.max_tilt = np.zeros(n)  # worst |roll| or |pitch| seen, degrees
        self._node = Node()
        for i in range(n):
            if not self._node.subscribe(Odometry, f"{prefix}/odom_{i}", self._cb(i)):
                raise RuntimeError(f"could not subscribe to {prefix}/odom_{i}")

    def _cb(self, i):
        def cb(msg):
            p, q = msg.pose.position, msg.pose.orientation
            # Yaw from the quaternion; the camera axis is body -Y.
            yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            cam = -np.array([np.cos(yaw + np.pi / 2), np.sin(yaw + np.pi / 2)])
            # Roll and pitch, to catch a rover that ended up on its side. A
            # tipped chassis can still log odometry motion and clear a
            # distance bar, so distance alone cannot see this.
            roll = np.arctan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
            pitch = np.arcsin(np.clip(2.0 * (q.w * q.y - q.z * q.x), -1.0, 1.0))
            tilt = np.degrees(max(abs(roll), abs(pitch)))
            with self._lock:
                self.max_tilt[i] = max(self.max_tilt[i], tilt)
                if self._last[i] is not None:
                    step = np.array([p.x - self._last[i][0], p.y - self._last[i][1]])
                    d = float(np.hypot(step[0], step[1]))
                    if d < _TELEPORT_M:
                        self.path[i] += d
                        # Signed: driving away from the camera view subtracts.
                        self.forward[i] += float(step @ cam)
                if self._start[i] is not None:
                    self.net[i] = float(np.hypot(p.x - self._start[i][0], p.y - self._start[i][1]))
                self._last[i] = (p.x, p.y)

        return cb

    def restart(self):
        with self._lock:
            self.path[:] = 0.0
            self.net[:] = 0.0
            self.forward[:] = 0.0
            self.max_tilt[:] = 0.0
            self._start = list(self._last)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model")
    p.add_argument("--agent", default="rover_line")
    p.add_argument("--n-agents", type=int, default=4)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lap-meters", type=float, default=rl.LAP_METERS)
    p.add_argument(
        "--max-deviation",
        type=float,
        default=0.6,
        help="worst tolerated |near-band cross-track offset| during an episode, "
        "in observation units where 1.0 is the frame edge. Surviving the cap "
        "says a run never lost the line; this says it never nearly did",
    )
    p.add_argument(
        "--max-tilt-deg",
        type=float,
        default=20.0,
        help="worst tolerated chassis roll/pitch. A tipped rover still logs "
        "odometry motion, so a distance bar alone cannot see it",
    )
    p.add_argument(
        "--random", action="store_true", help="ignore the model and act randomly (baseline)"
    )
    args = p.parse_args()

    import stable_baselines3 as sb3

    spec = get_spec(args.agent)
    cap = spec.max_episode_steps
    n = args.n_agents
    env = make_inprocess(args.agent, n_agents=n, seed=args.seed)
    env, _policy = wrap_for_observations(env, frame_stack=1)
    model = None if args.random else sb3.PPO.load(args.model)
    rng = np.random.default_rng(args.seed)

    odom = _Odom(env.venv._topic_prefix if hasattr(env, "venv") else env._topic_prefix, n)
    laps = alive_to_cap = total = 0
    paths, lens, fwds, devs = [], [], [], []

    for r in range(args.rounds):
        obs = env.reset()
        odom.restart()
        alive = np.ones(n, dtype=bool)
        died = np.full(n, cap, dtype=int)
        # Worst centring error each agent reached, from the observation the
        # policy itself saw -- band 0 is the nearest scan band.
        worst_dev = np.zeros(n)
        for t in range(cap):
            if model is None:
                action = rng.uniform(-1, 1, (n, 2)).astype(np.float32)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _rew, dones, _info = env.step(action)
            near = np.abs(np.asarray(obs)[:, 0])
            worst_dev = np.where(alive, np.maximum(worst_dev, near), worst_dev)
            died[alive & dones] = t + 1
            alive &= ~dones
            if not alive.any():
                break
        path, net = odom.path.copy(), odom.net.copy()
        fwd = odom.forward.copy()
        lasted = died >= cap
        # A lap must be driven FORWARD. Without this the bar is direction-blind
        # and a rover reversing around the loop passes it.
        # Every criterion must hold, not just distance: the run has to last
        # the cap, cover a lap FORWARD, stay centred, and stay upright.
        lap = (
            lasted
            & (fwd >= args.lap_meters)
            & (worst_dev <= args.max_deviation)
            & (odom.max_tilt <= args.max_tilt_deg)
        )
        laps += int(lap.sum())
        alive_to_cap += int(lasted.sum())
        total += n
        paths.extend(path.tolist())
        fwds.extend(fwd.tolist())
        devs.extend(worst_dev.tolist())
        lens.extend(died.tolist())
        print(
            f"  round {r}: laps {int(lap.sum())}/{n}  "
            f"ep_len={died.tolist()}  path_m={np.round(path, 2).tolist()}  "
            f"fwd_m={np.round(fwd, 2).tolist()}  "
            f"net_m={np.round(net, 2).tolist()}  "
            f"worst_dev={np.round(worst_dev, 2).tolist()}  "
            f"max_tilt={np.round(odom.max_tilt, 1).tolist()}"
        )

    print(
        f"\nEVAL {args.agent}: laps {laps}/{total} "
        f"({100.0 * laps / max(total, 1):.1f}%)  "
        f"survived-to-cap {alive_to_cap}/{total}"
    )
    print(
        f"  median ep_len {np.median(lens):.0f}/{cap}   "
        f"median actual path {np.median(paths):.2f} m   "
        f"median FORWARD {np.median(fwds):.2f} m   "
        f"(a lap is {args.lap_meters:.2f} m)"
    )
    print(
        f"  worst centring deviation {np.max(devs):.2f} "
        f"(bar {args.max_deviation:.2f})   "
        f"worst tilt {np.max(odom.max_tilt):.1f} deg (bar {args.max_tilt_deg:.1f})"
    )
    if np.median(fwds) < 0.5 * np.median(paths):
        print(
            "  WARNING: forward progress is far below path length -- the rover "
            "is covering ground without driving where its camera looks."
        )
    env.close()


if __name__ == "__main__":
    main()
