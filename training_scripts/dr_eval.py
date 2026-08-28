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
"""Per-agent solve-rate eval, precise enough to trust for a go/no-go call.

``deploy.py`` breaks its episode loop on ``dones.all()``, which reports
imprecisely once agents desync -- fine for watching a policy, not for deciding
whether a checkpoint is solved. This tracks each agent's own termination step
independently and reports the fraction that reached the full step cap.

    python training_scripts/dr_eval.py MODEL.zip --n-agents 8 --rounds 5
    python training_scripts/dr_eval.py MODEL.zip --track-shapes reset
"""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import VecMonitor

from gazebo_gymnasium_bridge.envs import make_inprocess
from gazebo_gymnasium_bridge.envs import wrap_for_observations
from gazebo_gymnasium_bridge.envs.agent_spec import _LF_WHEEL_RADIUS
from gazebo_gymnasium_bridge.envs.agent_spec import get_spec
from gazebo_gymnasium_bridge.envs.agent_spec import register_spec
from gazebo_gymnasium_bridge.envs.agent_spec import register_track_randomized
from gazebo_gymnasium_bridge.envs.agent_spec import TRACK_SHAPE_MODES

_ALGOS = {"ppo": sb3.PPO, "a2c": sb3.A2C, "ddpg": sb3.DDPG, "sac": sb3.SAC}

# Progress here is ESTIMATED from the commanded wheel velocities, not read
# back from the simulator: HarnessCore exposes no world-pose read path for
# this spec. The estimate is what the policy asked the wheels to do, which is
# the quantity a reward can be held to. For measured ground truth, read the
# per-agent odometry topics instead -- training_scripts/verify_forward_motion.py.
#
# The action -> velocity mapping comes from the spec's own action_to_commands
# rather than being reimplemented here. It used to be reimplemented as
# `mean(action) * wheel_speed`, which silently became wrong the moment
# line_follower moved to a forward-only clamped mapping on 2026-08-27 (there,
# action -1 is the SLOWEST FORWARD speed, not full reverse, so the old
# arithmetic reported reversing that never happened).
_PHYSICS_STEP_S = 0.01  # inprocess_vec_env._PHYSICS max_step_size


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model", help="path to the .zip checkpoint to evaluate")
    p.add_argument("--agent", default="line_follower")
    p.add_argument("--algo", default="ppo", choices=sorted(_ALGOS))
    p.add_argument("--n-agents", type=int, default=8)
    p.add_argument("--rounds", type=int, default=3, help="episodes per agent")
    p.add_argument("--track-shapes", default="reset", choices=TRACK_SHAPE_MODES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frame-stack", type=int, default=4, help="must match training")
    p.add_argument(
        "--wheel-radius",
        type=float,
        default=_LF_WHEEL_RADIUS,
        help="metres (default: the spec's own, from the firmware's WHEEL_DIAMETER_M)",
    )
    p.add_argument(
        "--solved-distance",
        type=float,
        default=0.0,
        help="metres of ground an episode must ALSO cover to count as solved "
        "(0 = off, the old steps-only bar). Surviving the step cap says the "
        "line stayed in frame; it says nothing about the rover going "
        "anywhere, and a creeping policy wins on the steps-only bar. The "
        "single-track loop is ~10 m, so 8.0 is most of a lap",
    )
    p.add_argument(
        "--spawn-redraw",
        type=int,
        default=0,
        help="redraw an already-terminal spawn up to N times (see "
        "AgentSpec.spawn_redraw_attempts)",
    )
    args = p.parse_args()

    name = register_track_randomized(args.agent, args.track_shapes)
    if args.spawn_redraw:
        spec = replace(
            get_spec(name),
            name=f"{name}_redraw",
            spawn_redraw_attempts=args.spawn_redraw,
        )
        register_spec(spec.name, lambda: spec)
        name = spec.name
    cap = get_spec(name).max_episode_steps
    env = VecMonitor(make_inprocess(name, n_agents=args.n_agents, seed=args.seed))
    env, _policy = wrap_for_observations(env, args.frame_stack)
    model = _ALGOS[args.algo].load(args.model)

    solved = total = 0
    solved_steps_only = 0
    term_steps = []
    nets, paths, reversing = [], [], []
    spec = get_spec(name)
    dt = spec.frame_skip * _PHYSICS_STEP_S
    to_cmds = spec.action_to_commands

    def _forward_mps(actions):
        """Mean wheel speed in m/s per agent, via the spec's own mapping."""
        out = np.empty(len(actions))
        for i, a in enumerate(actions):
            rads = [v for _joint, _mode, v in to_cmds(a)]
            out[i] = float(np.mean(rads)) * args.wheel_radius
        return out

    try:
        for r in range(args.rounds):
            obs = env.reset()
            alive = np.ones(args.n_agents, dtype=bool)
            died_at = np.full(args.n_agents, cap, dtype=int)
            net = np.zeros(args.n_agents)
            path = np.zeros(args.n_agents)
            rev = np.zeros(args.n_agents, dtype=int)
            for t in range(cap):
                action, _ = model.predict(obs, deterministic=True)
                # Signed forward speed: same-sign wheel commands drive the
                # base forward, opposite-sign ones spin it in place.
                v = _forward_mps(np.asarray(action, dtype=float).reshape(args.n_agents, -1))
                net += np.where(alive, v * dt, 0.0)
                path += np.where(alive, np.abs(v) * dt, 0.0)
                rev += (alive & (v < 0)).astype(int)
                obs, _rew, dones, _info = env.step(action)
                died_at[alive & dones] = t + 1
                alive &= ~dones
                if not alive.any():
                    break
            nets.extend(net.tolist())
            paths.extend(path.tolist())
            reversing.extend((rev / np.maximum(died_at, 1)).tolist())
            # Two bars. Steps-only is the historical one: the line stayed in
            # frame to the cap. It says nothing about the rover covering
            # ground, and a creeping policy passes it, so --solved-distance
            # adds the distance the episode actually travelled.
            lasted = died_at >= cap
            round_steps_only = int(lasted.sum())
            round_solved = int((lasted & (path >= args.solved_distance)).sum())
            solved_steps_only += round_steps_only
            solved += round_solved
            total += args.n_agents
            term_steps.extend(died_at.tolist())
            extra = ""
            if args.solved_distance > 0:
                extra = (
                    f" (steps-only would be {round_steps_only}) path={np.round(path, 2).tolist()}"
                )
            print(
                f"  round {r}: solved {round_solved}/{args.n_agents} "
                f"term_steps={died_at.tolist()}{extra}",
                flush=True,
            )
    finally:
        env.close()

    pct = 100.0 * solved / total
    med_net = float(np.median(nets))
    med_path = float(np.median(paths))
    eff = 100.0 * med_net / med_path if med_path > 1e-9 else 0.0
    bar = f"cap+{args.solved_distance:g}m" if args.solved_distance > 0 else "cap-only"
    print(
        f"EVAL model={Path(args.model).name} track_shapes={args.track_shapes} "
        f"n={args.n_agents} rounds={args.rounds} bar={bar} "
        f"solved={solved}/{total} ({pct:.1f}%) "
        f"steps_only={solved_steps_only}/{total} "
        f"({100.0 * solved_steps_only / total:.1f}%) "
        f"median_term_step={int(np.median(term_steps))} "
        f"median_net_m={med_net:.2f} median_path_m={med_path:.2f} "
        f"forward_efficiency={eff:.0f}% reversing_frac={float(np.mean(reversing)):.2f}"
    )
    # Surviving the step cap is NOT the same as driving the track: termination
    # only fires on line loss, so a policy that keeps the line centred while
    # inching back and forth scores a perfect steps_only rate while going
    # nowhere. That is what --solved-distance exists to catch, and the gap
    # between `solved` and `steps_only` is the size of the illusion.
    return 0 if solved == total else 1


if __name__ == "__main__":
    sys.exit(main())
