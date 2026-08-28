"""Run a trained line_follower policy on REAL camera frames, wheels stopped.

Belongs in the rover repo, not in gazebo_gymnasium (no physical logic lives
there). Nothing is ever sent to the drive ESP32: this prints what the policy
WOULD command, so the sim-to-real visual gap can be checked before anything
moves.

    python real_camera_test.py model.zip --url http://<cam-ip>/capture -n 40
    python real_camera_test.py model.zip --dir ./captured_frames
    python real_camera_test.py model.zip --self-check      # validate preprocessing

The self-check matters. If preprocessing here does not match what the policy
was trained on, every action below is garbage in a way that still looks
plausible. It renders frames from the sim, pushes them through BOTH this
pipeline and the env's own, and asserts the actions agree.
"""

import argparse
from collections import deque
import sys

import cv2
import numpy as np
import stable_baselines3 as sb3

OBS_HW = (64, 64)  # AgentSpec._LF_IMAGE
N_STACK = 4  # train.py --frame-stack default
WHEEL_SPEED = 15.0  # _LF_WHEEL_SPEED, rad/s at |action| = 1
WHEEL_RADIUS_M = 0.035  # real rover (sim derived ~0.034)
CONTROL_HZ = 20.0  # frame_skip=5 x 10 ms physics step
DARK_THRESH = 60  # agent_spec._LF_DARK


def to_obs_frame(rgb):
    """Any RGB image -> the 64x64 RGB uint8 frame the policy was trained on."""
    if rgb.shape[:2] != OBS_HW:
        rgb = cv2.resize(rgb, (OBS_HW[1], OBS_HW[0]), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.uint8)


def new_stack():
    """Empty frame stack, matching SB3's VecFrameStack at episode start.

    SB3 zero-fills and lets real frames push the zeros out, so the first three
    decisions of an episode are made against a partly-empty stack. Priming
    with copies of the first frame instead looks more sensible and is WRONG:
    it feeds the policy a history it never saw in training. Verified
    byte-identical to the env from step 3 on, and identical from step 0 with
    this zero-fill.
    """
    return deque([np.zeros((*OBS_HW, 3), np.uint8)] * N_STACK, maxlen=N_STACK)


def stack_to_obs(frames):
    """Deque of N_STACK HWC frames -> (1, 12, 64, 64), oldest channels first.

    Mirrors VecFrameStack (concatenate on the channel axis, oldest first) then
    VecTransposeImage (HWC -> CHW), which is the order wrap_for_observations
    applies.
    """
    hwc = np.concatenate(list(frames), axis=2)
    return np.transpose(hwc, (2, 0, 1))[None, ...]


def line_centroid_x(rgb):
    """Horizontal centroid of dark pixels, or None. Mirrors _lf_line_centroid."""
    mask = rgb.max(axis=2) < DARK_THRESH
    if not mask.any():
        return None
    return float(np.argwhere(mask)[:, 1].mean() / rgb.shape[1])


def describe(action):
    left, right = float(action[0]), float(action[1])
    l_mps = left * WHEEL_SPEED * WHEEL_RADIUS_M
    r_mps = right * WHEEL_SPEED * WHEEL_RADIUS_M
    turn = "straight"
    if abs(l_mps - r_mps) > 0.05:
        turn = "LEFT" if r_mps > l_mps else "RIGHT"
    return f"action=({left:+.2f},{right:+.2f}) -> {l_mps:+.3f}/{r_mps:+.3f} m/s  {turn}"


def self_check(model_path):
    """Assert this preprocessing agrees with the environment's own."""
    # DR OFF for this check. In training the policy sees DR-AUGMENTED frames
    # (exposure, noise, JPEG, track colour), and that augmentation is random
    # per call, so raw frames can never reproduce it exactly. Feeding real
    # camera frames unaugmented is the CORRECT deployment behaviour -- DR
    # exists to cover the real distribution, not to be replayed on top of it.
    # Turning DR off here isolates what this check is actually for: that the
    # resize, stacking order and channel transpose match training.
    from dataclasses import replace

    from stable_baselines3.common.vec_env import VecMonitor

    from gazebo_gymnasium_bridge.envs import make_inprocess
    from gazebo_gymnasium_bridge.envs import wrap_for_observations
    from gazebo_gymnasium_bridge.envs.agent_spec import get_spec
    from gazebo_gymnasium_bridge.envs.agent_spec import register_spec
    from gazebo_gymnasium_bridge.envs.agent_spec import register_track_randomized

    base = get_spec(register_track_randomized("line_follower", "reset"))
    spec = replace(
        base,
        name="lf_selfcheck",
        visual_randomization=0.0,
        track_color_randomization=0.0,
        action_noise_randomization=0.0,
        battery_discharge_randomization=0.0,
        action_gain_randomization=0.0,
    )
    register_spec(spec.name, lambda: spec)
    name = spec.name
    raw_env = make_inprocess(name, n_agents=1, seed=0)
    env, _ = wrap_for_observations(VecMonitor(raw_env), N_STACK)
    model = sb3.PPO.load(model_path)

    obs = env.reset()
    frames = new_stack()
    frames.append(to_obs_frame(raw_env._latest_obs[0]))
    worst = 0.0
    for _step in range(20):
        env_action, _ = model.predict(obs, deterministic=True)
        mine, _ = model.predict(stack_to_obs(frames), deterministic=True)
        worst = max(worst, float(np.abs(np.asarray(env_action[0]) - np.asarray(mine[0])).max()))
        obs, _, _, _ = env.step(env_action)
        frames.append(to_obs_frame(raw_env._latest_obs[0]))
    env.close()
    print(f"self-check: max action difference over 20 steps = {worst:.6f}")
    if worst > 1e-5:
        print("MISMATCH -- preprocessing here does NOT match training. Fix before trusting.")
        return 1
    print("OK -- preprocessing matches the training pipeline.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model")
    p.add_argument("--url", help="camera /capture endpoint (single JPEG per request)")
    p.add_argument("--dir", help="directory of saved frames instead of a live camera")
    p.add_argument("-n", type=int, default=30, help="frames to pull from --url")
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--save", default=None, help="write an annotated montage here")
    args = p.parse_args()

    if args.self_check:
        return self_check(args.model)
    if not args.url and not args.dir:
        p.error("need --url, --dir, or --self-check")

    model = sb3.PPO.load(args.model)
    print(
        f"control rate the policy expects: {CONTROL_HZ:.0f} Hz "
        f"({1000 / CONTROL_HZ:.0f} ms per decision)"
    )

    if args.dir:
        import pathlib

        paths = sorted(pathlib.Path(args.dir).glob("*"))
        sources = [cv2.cvtColor(cv2.imread(str(q)), cv2.COLOR_BGR2RGB) for q in paths]
        labels = [q.name for q in paths]
    else:
        import urllib.request

        sources, labels = [], []
        for k in range(args.n):
            with urllib.request.urlopen(args.url, timeout=5) as r:
                buf = np.frombuffer(r.read(), np.uint8)
            sources.append(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
            labels.append(f"frame {k}")

    if not sources:
        print("no frames")
        return 1

    frames = new_stack()
    seen = 0
    tiles = []
    for label, rgb in zip(labels, sources, strict=False):
        small = to_obs_frame(rgb)
        frames.append(small)
        action, _ = model.predict(stack_to_obs(frames), deterministic=True)
        cx = line_centroid_x(small)
        seen += cx is not None
        where = "no line detected" if cx is None else f"line at x={cx:.2f}"
        print(f"{label:>14}: {where:<20} {describe(action[0])}")
        if args.save:
            tile = cv2.resize(small, (192, 192), interpolation=cv2.INTER_NEAREST)
            tiles.append(cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))

    print(f"\nline visible in {seen}/{len(sources)} frames")
    if seen < len(sources):
        print(
            "Frames with no detectable dark line are the sim-to-real gap to close "
            "first: check exposure, lighting, and that the track reads darker than "
            f"{DARK_THRESH}/255 to the real camera."
        )
    if args.save and tiles:
        rows = [np.hstack(tiles[i : i + 6]) for i in range(0, len(tiles) - len(tiles) % 6, 6)]
        if rows:
            cv2.imwrite(args.save, np.vstack(rows))
            print(f"wrote {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
