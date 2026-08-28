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

Logging: ``--tensorboard`` writes SB3's standard scalars to
``models/<agent>_multi/tb/`` (``tensorboard --logdir models``); ``--wandb``
additionally mirrors that same run to Weights & Biases (needs ``wandb``
installed + authenticated -- soft dependency, same as ``sweep.py``'s
``--wandb``) via ``sync_tensorboard``, so it implies ``--tensorboard`` even
if that flag isn't also given.

Hugging Face Hub: ``--push-to-hub <repo_id>`` uploads the trained model
plus an auto-generated model card after training finishes (needs
``huggingface_hub`` installed + authenticated -- soft dependency, same
treatment as ``--wandb``). The card's evaluation numbers come from a real
deterministic eval run right here, not the training curve's own rolling
metric -- see ``docs/examples/line_follower.md``'s Engineering Notes for
why that distinction has mattered before in this project.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from gazebo_gymnasium_bridge.envs import get_spec
from gazebo_gymnasium_bridge.envs import make_harness
from gazebo_gymnasium_bridge.envs import make_inprocess
from gazebo_gymnasium_bridge.envs import make_multi
from gazebo_gymnasium_bridge.envs import registered_specs
from gazebo_gymnasium_bridge.envs import wrap_for_observations
from gazebo_gymnasium_bridge.envs.agent_spec import register_track_randomized
from gazebo_gymnasium_bridge.envs.agent_spec import TRACK_SHAPE_MODES

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"
_ALGOS = {
    "ppo": sb3.PPO,
    "a2c": sb3.A2C,
    "sac": sb3.SAC,
    "td3": sb3.TD3,
    "ddpg": sb3.DDPG,
}


def _device() -> str:
    return os.environ.get("GAZEBO_GYM_DEVICE", "cpu").lower()


def _scaled_timeouts(n_agents: int):
    """Scale state-wait timeouts with agent count (O(N) discovery; 16 ~5s/15s)."""
    return max(3.0, 0.9 * n_agents), max(1.0, 0.3 * n_agents)


def _deterministic_eval(vec_env, model, n_agents, n_episodes):
    """Real per-agent eval, not the training curve's own rolling metric.

    dones conflates termination and truncation (SB3's own VecEnv contract),
    so a plain dones.all() loop-break gets confused once agents desync --
    tracks each agent's own term_step precisely instead, same approach
    this project's DR-tuning work already validated. Returns
    (mean_reward, mean_steps, n_reached_cap, n_agents * n_episodes).
    """
    cap = vec_env.max_episode_steps
    rewards, full_cap = [], 0
    for _ in range(n_episodes):
        obs = vec_env.reset()
        term_step = np.full(n_agents, -1)
        live = np.ones(n_agents, dtype=bool)
        ep_reward = np.zeros(n_agents)
        for t in range(cap):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, dones, _infos = vec_env.step(action)
            ep_reward += r * live
            newly_done = live & dones
            term_step[newly_done] = t + 1
            live &= ~dones
        full_cap += int(((term_step == -1) | (term_step == cap)).sum())
        rewards.extend(ep_reward.tolist())
    total = n_agents * n_episodes
    return float(np.mean(rewards)), cap, full_cap, total


def _model_card(args, algo_kwargs, mean_reward, cap, full_cap, total):
    return f"""---
tags:
- reinforcement-learning
- stable-baselines3
- gazebo-gymnasium
---

# {args.algo.upper()} agent for `{args.agent}` (gazebo_gymnasium)

Trained with [gazebo_gymnasium](https://github.com/CursedRock17/gazebo_gymnasium), a
spec-driven, N-agents-in-one-Gazebo-world reinforcement learning framework.

## Training configuration

| Parameter | Value |
|---|---|
| Algorithm | {args.algo.upper()} |
| Environment | `{args.agent}` |
| Backend | `{args.backend}` |
| n_agents | {args.n_agents} |
| timesteps | {args.timesteps} |
| device | {algo_kwargs.get("device", "cpu")} |
{
        "".join(
            f"| {k} | {v} |\n"
            for k, v in algo_kwargs.items()
            if k not in ("verbose", "tensorboard_log", "device")
        )
    }

## Evaluation

Real deterministic evaluation ({args.hub_eval_episodes} episodes x {args.n_agents} agents =
{total} agent-episodes), not the training curve's own rolling metric:

| Metric | Value |
|---|---|
| Mean episode reward | {mean_reward:.2f} |
| Episode step cap | {cap} |
| Reached the cap | {full_cap}/{total} ({100 * full_cap / total:.1f}%) |

## Usage

```python
from huggingface_hub import hf_hub_download
from stable_baselines3 import {args.algo.upper()}

model_path = hf_hub_download(repo_id="{args.push_to_hub}", filename="model.zip")
model = {args.algo.upper()}.load(model_path)
```
"""


def _push_to_hub(repo_id, final_path, args, algo_kwargs, mean_reward, cap, full_cap, total):
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        print(
            f"[train] WARNING: --push-to-hub requested but huggingface_hub not "
            f"installed ({exc}); skipping upload"
        )
        return
    card = _model_card(args, algo_kwargs, mean_reward, cap, full_cap, total)
    card_path = final_path.with_name("README.md")
    card_path.write_text(card)
    api = HfApi()
    try:
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        api.upload_file(path_or_fileobj=str(final_path), path_in_repo="model.zip", repo_id=repo_id)
        api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md", repo_id=repo_id)
        # Verify, don't assume: confirm the upload actually landed rather
        # than trusting a non-exception return.
        if api.file_exists(repo_id, "model.zip") and api.file_exists(repo_id, "README.md"):
            print(f"[train] pushed to https://huggingface.co/{repo_id} (verified)")
        else:
            print(
                f"[train] WARNING: upload call succeeded but file_exists check failed "
                f"for {repo_id} -- verify manually"
            )
    except Exception as exc:  # noqa: BLE001 -- any Hub failure is non-fatal, training already saved
        print(f"[train] WARNING: push to hub failed ({type(exc).__name__}: {exc})")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--agent", default="cartpole", help=f"registered spec: {registered_specs()}")
    p.add_argument("--n_agents", type=int, default=4, help="agents-in-one-sim (1 = single-agent)")
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
    # Derived spec name when randomizing (else args.agent unchanged), so a
    # multi-track run never resumes from -- or overwrites -- a single-track
    # checkpoint of the same agent.
    agent_name = register_track_randomized(args.agent, args.track_shapes)
    _factories = {"inprocess": make_inprocess, "harness": make_harness, "peragent": make_multi}
    vec_env = _factories[args.backend](
        agent_name,
        n_agents=args.n_agents,
        world_name=args.world,
        reset_timeout=reset_to,
        step_timeout=step_to,
    )
    # VecMonitor is what populates model.ep_info_buffer -- without it, SB3 has
    # no per-episode reward/length to report, so rollout/ep_rew_mean never
    # appears in the console table OR in --tensorboard/--wandb (same reason
    # sweep.py already wraps with this to get its own reward curve).
    vec_env = VecMonitor(vec_env)
    vec_env, policy = wrap_for_observations(vec_env, args.frame_stack)

    model_dir = MODELS_ROOT / f"{agent_name}_multi"
    model_dir.mkdir(parents=True, exist_ok=True)
    final_path = model_dir / f"final_{args.algo}_n{args.n_agents}.zip"

    # --wandb implies --tensorboard: W&B's sync_tensorboard=True (below) reads
    # from a real tensorboard log dir, it doesn't replace the need for one.
    tb_log = str(model_dir / "tb") if (args.tensorboard or args.wandb) else None

    wb_run = None
    if args.wandb:
        try:
            import wandb

            wb_run = wandb.init(
                project=args.project,
                name=f"{args.agent}_{args.algo}_n{args.n_agents}",
                config={
                    "agent": args.agent,
                    "algo": args.algo,
                    "n_agents": args.n_agents,
                    "timesteps": args.timesteps,
                    "backend": args.backend,
                },
                sync_tensorboard=True,
            )
        except Exception as exc:  # not installed / not authed
            print(f"[train] WARNING: --wandb requested but disabled ({type(exc).__name__}: {exc})")
            wb_run = None

    AlgoCls = _ALGOS[args.algo]
    # n_steps is per-env; with many agents the rollout is already large, so
    # shrink per-env steps to keep update size sane (PPO/A2C only).
    algo_kwargs = {"verbose": 1, "device": _device(), "tensorboard_log": tb_log}
    if args.algo in ("ppo", "a2c"):
        algo_kwargs["n_steps"] = max(32, 256 // args.n_agents)

    resumed = False
    if final_path.exists():
        try:
            print(f"[train] resuming from {final_path}")
            # tensorboard_log is a kwarg override, not a saved hyperparameter --
            # pass it explicitly so a resumed run still logs (a plain reload
            # would otherwise silently drop it even if the original run had it).
            model = AlgoCls.load(final_path, env=vec_env, tensorboard_log=tb_log)
            resumed = True
        except (ValueError, RuntimeError) as exc:
            print(f"[train] WARNING cannot load ({type(exc).__name__}: {exc}); starting fresh")
            # AlgoCls is a Union of SB3 algorithm classes and algo_kwargs is a
            # runtime-built dict, so ty can't verify individual kwargs against
            # each class's constructor and flags every parameter across every
            # class in the union -- a known static-analysis limitation of
            # **kwargs spreading, not a real bug (this call runs correctly
            # throughout this project's training history).
            model = AlgoCls(policy, vec_env, **algo_kwargs)  # ty: ignore[invalid-argument-type]
    else:
        model = AlgoCls(policy, vec_env, **algo_kwargs)  # ty: ignore[invalid-argument-type]

    ckpt = CheckpointCallback(
        save_freq=max(1, 10_000 // args.n_agents),
        save_path=str(model_dir),
        name_prefix=f"{args.algo}_{args.agent}_n{args.n_agents}",
    )

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
                    vp = str(model_dir / f"{args.agent}_replay.gif")
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
        # Only push a model that actually finished training -- an exception
        # mid-run still gets its partial progress saved above, but nothing
        # gets published for it.
        if training_succeeded and args.push_to_hub:
            print(
                f"[train] running {args.hub_eval_episodes}-episode deterministic "
                f"eval for the model card..."
            )
            mean_reward, cap, full_cap, total = _deterministic_eval(
                vec_env, model, args.n_agents, args.hub_eval_episodes
            )
            print(
                f"[train] eval: mean_reward={mean_reward:.2f} "
                f"reached_cap={full_cap}/{total} ({100 * full_cap / total:.1f}%)"
            )
            _push_to_hub(
                args.push_to_hub, final_path, args, algo_kwargs, mean_reward, cap, full_cap, total
            )
        vec_env.close()
        if wb_run is not None:
            wb_run.finish()

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
