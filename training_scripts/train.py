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

r"""Generalized training entry point — N agents of any registered spec in one sim.

This is the single training script. Single-agent is just ``--n_agents 1``;
"MultiAnt" etc. are ``--agent ant``. It drives the generalized
``MultiAgentGazeboVecEnv`` via ``make_multi`` and trains with Stable-Baselines3.

Prereq: the matching world must already be running, e.g.
    ros2 launch gazebo_gymnasium_bringup cartpole_multi.launch.py \
        n_agents:=4 headless:=true

Then:
    python training_scripts/train.py --agent cartpole --n_agents 4 \
        --algo ppo --timesteps 200000
"""

import argparse
import os
from pathlib import Path

import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import CheckpointCallback

from gazebo_gymnasium_bridge.envs import get_spec
from gazebo_gymnasium_bridge.envs import make_harness
from gazebo_gymnasium_bridge.envs import make_inprocess
from gazebo_gymnasium_bridge.envs import make_multi
from gazebo_gymnasium_bridge.envs import registered_specs
from gazebo_gymnasium_bridge.envs import wrap_for_observations


MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"
_ALGOS = {
    "ppo": sb3.PPO, "a2c": sb3.A2C, "sac": sb3.SAC,
    "td3": sb3.TD3, "ddpg": sb3.DDPG,
}


def _device() -> str:
    return os.environ.get("GAZEBO_GYM_DEVICE", "cpu").lower()


def _scaled_timeouts(n_agents: int):
    """Scale state-wait timeouts with agent count (O(N) discovery; 16 ~5s/15s)."""
    return max(3.0, 0.9 * n_agents), max(1.0, 0.3 * n_agents)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--agent", default="cartpole",
                   help=f"registered spec: {registered_specs()}")
    p.add_argument("--n_agents", type=int, default=4,
                   help="agents-in-one-sim (1 = single-agent)")
    p.add_argument("--algo", default="ppo", choices=sorted(_ALGOS))
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--world", default=None,
                   help="gz world name (default: <agent>_multi)")
    p.add_argument("--frame-stack", type=int, default=4,
                   help="frames stacked for IMAGE observations (1 disables; "
                        "ignored for state observations)")
    p.add_argument("--backend", default="inprocess",
                   choices=("inprocess", "harness", "peragent"),
                   help="inprocess: sim hosted in this process, no launch "
                        "needed (fastest, default); harness: batched O(1) "
                        "transport to a launched gz sim; peragent: per-agent "
                        "topics + respawn reset")
    p.add_argument("--push-to-hub", metavar="REPO_ID", default=None,
                   help="after training, upload the model + an auto-generated "
                        "model card to this Hugging Face Hub repo (user/name). "
                        "Requires a token (HF_TOKEN or `hf auth login`).")
    p.add_argument("--hub-private", action="store_true",
                   help="create the Hub repo as private (with --push-to-hub)")
    p.add_argument("--push-video", action="store_true",
                   help="also record a replay video for the model card "
                        "(camera/image-observation envs only)")
    args = p.parse_args()

    reset_to, step_to = _scaled_timeouts(args.n_agents)
    _factories = {"inprocess": make_inprocess, "harness": make_harness,
                  "peragent": make_multi}
    vec_env = _factories[args.backend](
        args.agent, n_agents=args.n_agents, world_name=args.world,
        reset_timeout=reset_to, step_timeout=step_to)
    vec_env, policy = wrap_for_observations(vec_env, args.frame_stack)

    model_dir = MODELS_ROOT / f"{args.agent}_multi"
    model_dir.mkdir(parents=True, exist_ok=True)
    final_path = model_dir / f"final_{args.algo}_n{args.n_agents}.zip"

    AlgoCls = _ALGOS[args.algo]
    # n_steps is per-env; with many agents the rollout is already large, so
    # shrink per-env steps to keep update size sane (PPO/A2C only).
    algo_kwargs = {"verbose": 1, "device": _device()}
    if args.algo in ("ppo", "a2c"):
        algo_kwargs["n_steps"] = max(32, 256 // args.n_agents)

    resumed = False
    if final_path.exists():
        try:
            print(f"[train] resuming from {final_path}")
            model = AlgoCls.load(final_path, env=vec_env)
            resumed = True
        except (ValueError, RuntimeError) as exc:
            print(f"[train] WARNING cannot load ({type(exc).__name__}: {exc}); "
                  f"starting fresh")
            model = AlgoCls(policy, vec_env, **algo_kwargs)
    else:
        model = AlgoCls(policy, vec_env, **algo_kwargs)

    ckpt = CheckpointCallback(
        save_freq=max(1, 10_000 // args.n_agents), save_path=str(model_dir),
        name_prefix=f"{args.algo}_{args.agent}_n{args.n_agents}")

    print(f"[train] agent={args.agent} n_agents={args.n_agents} "
          f"algo={args.algo} timesteps={args.timesteps} -> {model_dir}")
    # Hub extras (eval + replay video) need the env still open, so gather them
    # on the success path before the finally closes it.
    eval_result, eval_metrics, video_path = None, {}, None
    try:
        model.learn(total_timesteps=args.timesteps, callback=ckpt,
                    reset_num_timesteps=not resumed)
        if args.push_to_hub:
            from hub import evaluate_model, record_replay
            try:
                eval_result, eval_metrics = evaluate_model(model, vec_env)
                print(f"[train] eval: {eval_result}")
            except Exception as exc:              # noqa: B902
                print(f"[train] eval skipped ({type(exc).__name__}: {exc})")
            if args.push_video:
                try:
                    vp = str(model_dir / f"{args.agent}_replay.mp4")
                    video_path = record_replay(
                        model, vec_env, get_spec(args.agent), vp)
                    print(f"[train] replay video: {video_path}" if video_path
                          else "[train] --push-video: no headless video for "
                               "this env (camera envs only); skipping.")
                except Exception as exc:          # noqa: B902
                    print(f"[train] video skipped ({type(exc).__name__}: "
                          f"{exc})")
    finally:
        model.save(final_path)
        print(f"[train] saved {final_path}")
        vec_env.close()

    if args.push_to_hub:
        from hub import push_to_hub
        hp = {"timesteps": args.timesteps, "backend": args.backend,
              "device": _device()}
        hp.update({k: v for k, v in algo_kwargs.items() if k != "verbose"})
        hp.update(eval_metrics)
        url = push_to_hub(
            final_path, args.push_to_hub, agent=args.agent, algo=args.algo,
            n_agents=args.n_agents, hyperparams=hp, eval_result=eval_result,
            video_path=video_path, private=args.hub_private)
        print(f"[train] pushed to Hugging Face Hub: {url}")


if __name__ == "__main__":
    main()
