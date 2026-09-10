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
"""Hyperparameter sweep over the in-process backend (headless, no launch).

Trains one algorithm (``--algo``: ppo/a2c/ddpg/sac) on the given spec across a
grid of hyperparameters, records each trial's learning curve, and reports the
best config + whether it reached ``--solved``. Every trial logs to a CSV
(always) and to Weights & Biases (optionally, if ``--wandb`` is passed and
``wandb`` is installed + authenticated). Because the sim is in-process and
unthrottled, a full sweep runs in minutes on a laptop CPU.

    python training_scripts/sweep.py --timesteps 250000 --n-agents 16
    python training_scripts/sweep.py --wandb --project gazebo-cartpole
    python training_scripts/sweep.py --agent line_follower --algo ddpg
    python training_scripts/sweep.py --agent hopper --algo sac
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from gazebo_gymnasium_bridge.envs import make_inprocess
from gazebo_gymnasium_bridge.envs import wrap_for_observations
from gazebo_gymnasium_bridge.envs.agent_spec import register_track_randomized
from gazebo_gymnasium_bridge.envs.agent_spec import TRACK_SHAPE_MODES

_ALGOS = {"ppo": sb3.PPO, "a2c": sb3.A2C, "ddpg": sb3.DDPG, "sac": sb3.SAC}

# The checks validate_rover.py reports, in the order it prints them.
_CHECKS = ("camera", "forward", "speed")


def linear_schedule(initial):
    """SB3 schedule decaying `initial` linearly to 0 across training.

    SB3 calls these with progress_remaining, which runs 1.0 -> 0.0.

    The PPO paper anneals BOTH the Adam stepsize and the clipping parameter
    this way for its vision experiments (Table 5: 2.5e-4*alpha and 0.1*alpha,
    alpha linear 1 -> 0). This project has never used it, and its training
    history is full of the failure that annealing exists to prevent: runs that
    peak and then destroy themselves, including four independent seeds
    regressing from 83.3% to 60/27/0/0, and a sweep winner that regressed when
    extended. A constant stepsize late in training keeps taking full-size
    steps on an already-good policy.
    """
    initial = float(initial)

    def _schedule(progress_remaining: float) -> float:
        return progress_remaining * initial

    return _schedule


def _sweep_agent_name(args):
    """Spec name this sweep actually trains (derived when randomizing tracks)."""
    return register_track_randomized(args.agent, args.track_shapes)


# Search spaces — small and hand-picked, one grid per algo since their
# hyperparameters don't overlap (PPO/A2C are on-policy with different knobs;
# DDPG is off-policy with a replay buffer). Not agent-tuned -- a reasonable
# generic starting grid, not guaranteed optimal for every --agent.
CONFIGS = {
    "ppo": [
        {
            "learning_rate": 3e-4,
            "ent_coef": 0.01,
            "n_steps": 64,
            "n_epochs": 6,
            "batch_size": 256,
            "gamma": 0.99,
        },
        {
            "learning_rate": 1e-3,
            "ent_coef": 0.0,
            "n_steps": 128,
            "n_epochs": 10,
            "batch_size": 256,
            "gamma": 0.99,
        },
        {
            "learning_rate": 5e-4,
            "ent_coef": 0.005,
            "n_steps": 128,
            "n_epochs": 10,
            "batch_size": 512,
            "gamma": 0.98,
        },
        {
            "learning_rate": 3e-4,
            "ent_coef": 0.0,
            "n_steps": 256,
            "n_epochs": 10,
            "batch_size": 512,
            "gamma": 0.99,
        },
    ],
    "a2c": [
        {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "ent_coef": 0.01,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "vf_coef": 0.5,
        },
        {
            "learning_rate": 3e-4,
            "n_steps": 16,
            "ent_coef": 0.0,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "vf_coef": 0.5,
        },
        {
            "learning_rate": 1e-3,
            "n_steps": 8,
            "ent_coef": 0.005,
            "gamma": 0.98,
            "gae_lambda": 0.9,
            "vf_coef": 0.25,
        },
        {
            "learning_rate": 5e-4,
            "n_steps": 32,
            "ent_coef": 0.0,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "vf_coef": 0.5,
        },
    ],
    # buffer_size kept modest (not SB3's 1M default): image observations
    # (n_stack, 64, 64) uint8 make a large replay buffer expensive in RAM,
    # and a short sweep budget doesn't benefit from a buffer far bigger than
    # the total step budget anyway.
    "ddpg": [
        {
            "learning_rate": 1e-3,
            "buffer_size": 50_000,
            "learning_starts": 1000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
        },
        {
            "learning_rate": 3e-4,
            "buffer_size": 50_000,
            "learning_starts": 2000,
            "batch_size": 128,
            "tau": 0.01,
            "gamma": 0.98,
            "train_freq": 4,
        },
        {
            "learning_rate": 1e-4,
            "buffer_size": 50_000,
            "learning_starts": 1000,
            "batch_size": 256,
            "tau": 0.02,
            "gamma": 0.99,
            "train_freq": 1,
        },
        {
            "learning_rate": 5e-4,
            "buffer_size": 50_000,
            "learning_starts": 5000,
            "batch_size": 64,
            "tau": 0.005,
            "gamma": 0.995,
            "train_freq": 8,
        },
    ],
    # Anchored on real published numbers, not guessed: rl-baselines3-zoo's
    # own tuned configs barely touch SAC's defaults for real (non-PyBullet)
    # MuJoCo envs -- HalfCheetah-v4/Walker2d-v4/Hopper-v4/Ant-v4 all just use
    # SB3's SAC() class defaults (lr=3e-4, buffer=1e6, batch=256, tau=0.005,
    # gamma=0.99, train_freq/gradient_steps=1, ent_coef='auto') plus a delayed
    # learning_starts=10000. Config 0 below IS that finding, buffer_size cut
    # down from 1e6 since a short sweep budget doesn't benefit from a replay
    # buffer far bigger than the total step count anyway. Configs 1-3 vary
    # around it (the zoo's own PyBullet-flavor defaults for the faster-update
    # variant in particular) since a short sweep can't afford 1e6-step trials
    # the way the zoo's own runs do, and faster updates matter more at a
    # smaller budget.
    "sac": [
        {
            "learning_rate": 3e-4,
            "buffer_size": 200_000,
            "learning_starts": 10_000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
        },
        {
            "learning_rate": 7.3e-4,
            "buffer_size": 300_000,
            "learning_starts": 10_000,
            "batch_size": 256,
            "tau": 0.02,
            "gamma": 0.98,
            "train_freq": 8,
            "gradient_steps": 8,
        },
        {
            "learning_rate": 3e-4,
            "buffer_size": 100_000,
            "learning_starts": 2_000,
            "batch_size": 128,
            "tau": 0.01,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 4,
        },
        {
            "learning_rate": 1e-4,
            "buffer_size": 300_000,
            "learning_starts": 10_000,
            "batch_size": 512,
            "tau": 0.005,
            "gamma": 0.995,
            "train_freq": 1,
            "gradient_steps": 1,
        },
    ],
}

# Per-agent overrides for CONFIGS, for the agents this project's own MuJoCo
# ports are analogous to -- trial 0 in each list is a real published config
# (rl-baselines3-zoo's own tuned hyperparameters for the closest standard
# Gymnasium/MuJoCo task, https://github.com/DLR-RM/rl-baselines3-zoo), not a
# guess; trials 1-3 are hand-picked variations around it. The generic CONFIGS
# grids above were never agent-tuned (see this file's own long-standing
# docstring note) -- these are, for the six agents that had a real published
# reference to anchor on. Ant has no PPO entry in the zoo at all (a known,
# documented difficulty, not an oversight -- see docs/examples/README.md);
# its SAC grid leans on real, sourced advice instead: SAC's entropy bonus
# specifically counters the "stand still and collect the alive bonus" local
# optimum PPO falls into on this task (Regularization Matters in Policy
# Optimization, https://arxiv.org/pdf/1910.09191).
AGENT_CONFIGS = {
    "hopper": {
        "ppo": [
            {
                "learning_rate": 9.80828e-05,
                "n_steps": 512,
                "n_epochs": 5,
                "batch_size": 32,
                "gamma": 0.999,
                "gae_lambda": 0.99,
                "ent_coef": 0.00229519,
                "clip_range": 0.2,
                "max_grad_norm": 0.7,
                "vf_coef": 0.835671,
                "policy_kwargs": dict(
                    log_std_init=-2, ortho_init=False, net_arch=dict(pi=[256, 256], vf=[256, 256])
                ),
            },
            {
                "learning_rate": 3e-4,
                "n_steps": 512,
                "n_epochs": 10,
                "batch_size": 64,
                "gamma": 0.999,
                "gae_lambda": 0.99,
                "ent_coef": 0.001,
                "clip_range": 0.2,
            },
            {
                "learning_rate": 1e-4,
                "n_steps": 1024,
                "n_epochs": 5,
                "batch_size": 64,
                "gamma": 0.995,
                "gae_lambda": 0.95,
                "ent_coef": 0.005,
                "clip_range": 0.3,
            },
            {
                "learning_rate": 2e-4,
                "n_steps": 256,
                "n_epochs": 5,
                "batch_size": 32,
                "gamma": 0.999,
                "gae_lambda": 0.99,
                "ent_coef": 0.002,
                "clip_range": 0.2,
            },
        ],
        "sac": CONFIGS["sac"],
    },
    "walker2d": {
        "ppo": [
            {
                "learning_rate": 5.05041e-05,
                "n_steps": 512,
                "n_epochs": 20,
                "batch_size": 32,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": 0.000585045,
                "clip_range": 0.1,
                "max_grad_norm": 1.0,
                "vf_coef": 0.871923,
            },
            {
                "learning_rate": 1e-4,
                "n_steps": 512,
                "n_epochs": 10,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": 0.0005,
                "clip_range": 0.2,
            },
            {
                "learning_rate": 3e-4,
                "n_steps": 256,
                "n_epochs": 10,
                "batch_size": 64,
                "gamma": 0.995,
                "gae_lambda": 0.95,
                "ent_coef": 0.0,
                "clip_range": 0.2,
            },
            {
                "learning_rate": 5e-5,
                "n_steps": 1024,
                "n_epochs": 20,
                "batch_size": 32,
                "gamma": 0.99,
                "gae_lambda": 0.9,
                "ent_coef": 0.001,
                "clip_range": 0.1,
            },
        ],
        "sac": CONFIGS["sac"],
    },
    "half_cheetah": {
        "ppo": [
            {
                "learning_rate": 2.0633e-05,
                "n_steps": 512,
                "n_epochs": 20,
                "batch_size": 64,
                "gamma": 0.98,
                "gae_lambda": 0.92,
                "ent_coef": 0.000401762,
                "clip_range": 0.1,
                "max_grad_norm": 0.8,
                "vf_coef": 0.58096,
                "policy_kwargs": dict(
                    log_std_init=-2, ortho_init=False, net_arch=dict(pi=[256, 256], vf=[256, 256])
                ),
            },
            {
                "learning_rate": 5e-5,
                "n_steps": 512,
                "n_epochs": 10,
                "batch_size": 64,
                "gamma": 0.98,
                "gae_lambda": 0.92,
                "ent_coef": 0.0005,
                "clip_range": 0.2,
            },
            {
                "learning_rate": 3e-4,
                "n_steps": 256,
                "n_epochs": 10,
                "batch_size": 128,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": 0.0,
                "clip_range": 0.2,
            },
            {
                "learning_rate": 1e-4,
                "n_steps": 1024,
                "n_epochs": 20,
                "batch_size": 64,
                "gamma": 0.98,
                "gae_lambda": 0.9,
                "ent_coef": 0.001,
                "clip_range": 0.1,
            },
        ],
        "sac": CONFIGS["sac"],
    },
    "reacher": {
        "ppo": [
            {
                "learning_rate": 0.000104019,
                "n_steps": 512,
                "n_epochs": 5,
                "batch_size": 32,
                "gamma": 0.9,
                "gae_lambda": 1.0,
                "ent_coef": 7.52585e-08,
                "clip_range": 0.3,
                "max_grad_norm": 0.9,
                "vf_coef": 0.950368,
            },
            {
                "learning_rate": 3e-4,
                "n_steps": 256,
                "n_epochs": 10,
                "batch_size": 64,
                "gamma": 0.9,
                "gae_lambda": 0.95,
                "ent_coef": 0.0,
                "clip_range": 0.2,
            },
            {
                "learning_rate": 5e-5,
                "n_steps": 1024,
                "n_epochs": 5,
                "batch_size": 32,
                "gamma": 0.95,
                "gae_lambda": 1.0,
                "ent_coef": 0.0,
                "clip_range": 0.3,
            },
            {
                "learning_rate": 2e-4,
                "n_steps": 512,
                "n_epochs": 10,
                "batch_size": 64,
                "gamma": 0.9,
                "gae_lambda": 0.9,
                "ent_coef": 0.001,
                "clip_range": 0.2,
            },
        ],
        # Not in the zoo's tuned SAC list at all (Reacher-v4 absent there
        # too), but gazebo_gymnasium's own probe already found SAC's
        # defaults beating PPO here (-8.48 vs -13 @100k) -- reuse the
        # generic SAC grid as the real starting point.
        "sac": CONFIGS["sac"],
    },
    "inverted_double_pendulum": {
        "ppo": [
            {
                "learning_rate": 0.000155454,
                "n_steps": 128,
                "n_epochs": 10,
                "batch_size": 512,
                "gamma": 0.98,
                "gae_lambda": 0.8,
                "ent_coef": 1.05057e-06,
                "clip_range": 0.4,
                "max_grad_norm": 0.5,
                "vf_coef": 0.695929,
            },
            {
                "learning_rate": 3e-4,
                "n_steps": 256,
                "n_epochs": 10,
                "batch_size": 256,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": 0.0,
                "clip_range": 0.2,
            },
            {
                "learning_rate": 1e-4,
                "n_steps": 128,
                "n_epochs": 5,
                "batch_size": 512,
                "gamma": 0.98,
                "gae_lambda": 0.8,
                "ent_coef": 0.0,
                "clip_range": 0.3,
            },
            {
                "learning_rate": 5e-4,
                "n_steps": 64,
                "n_epochs": 10,
                "batch_size": 256,
                "gamma": 0.99,
                "gae_lambda": 0.9,
                "ent_coef": 0.001,
                "clip_range": 0.2,
            },
        ],
        "sac": CONFIGS["sac"],
    },
    "ant": {
        # No PPO entry -- see this dict's own docstring note above; the
        # generic CONFIGS["ppo"] grid is the fallback and is not expected to
        # do much better than the 0.0 m/s local optimum already found.
        "sac": CONFIGS["sac"],
    },
}


def _configs_for(agent, algo):
    return AGENT_CONFIGS.get(agent, {}).get(algo, CONFIGS[algo])


class CurveLogger(BaseCallback):
    """Snapshot the rolling mean episode reward every `every` steps."""

    def __init__(self, every, trial, writer, wb):
        super().__init__()
        self.every = every
        self.trial = trial
        self.writer = writer
        self.wb = wb
        self.curve = []
        self._next = 0

    def _on_step(self):
        if self.num_timesteps >= self._next and self.model.ep_info_buffer:
            r = float(np.mean([e["r"] for e in self.model.ep_info_buffer]))
            self.curve.append((self.num_timesteps, r))
            self.writer.writerow(
                {"trial": self.trial, "step": self.num_timesteps, "mean_ep_reward": round(r, 2)}
            )
            if self.wb is not None:
                self.wb.log({"mean_ep_reward": r}, step=self.num_timesteps)
            self._next += self.every
        return True


def _trial_model_path(out, idx):
    # Namespaced by --out's own filename stem, not just its parent dir --
    # every sweep writes into the same models/ directory, so a bare
    # ".sweep_trial_{idx}.zip" collides across concurrently-running sweeps
    # (found running six sweeps in parallel: one sweep's real trial-0 winner
    # got silently overwritten/deleted by another sweep's own trial-0, mid-run
    # -- "No trial produced a usable model" despite a real 533.0 reward
    # sitting in the log two lines above). --out already differs per sweep
    # (explicit --out, or the timestamp-based default), so its stem is a
    # cheap, already-unique namespace.
    return Path(out).parent / f".{Path(out).stem}_trial_{idx}.zip"


def _validate_trial(idx, args):
    """Run validate_rover.py against one finished trial's model.

    Its own subprocess, for the same reason every trial gets one: a camera env
    can only be built once per process (gz-sim's render scene is a
    process-wide singleton), and the orchestrator has already spent that slot
    if it ever built one. Returns a one-line verdict.
    """
    model = _trial_model_path(args.out, idx)
    if not model.exists():
        return "SKIPPED (no model)"
    out_json = args.out.parent / f".{args.out.stem}_trial_{idx}_validation.json"
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("validate_rover.py")),
        "--agent",
        args.agent,
        "--n-agents",
        str(min(args.n_agents, 4)),
        "--track-shapes",
        args.track_shapes,
        f"--forward-axis={args.forward_axis}",
        "--model",
        str(model),
        "--seed",
        str(args.seed),
        "--json-out",
        str(out_json),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return "ERROR (validation timed out)"
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    verdicts = [ln.strip() for ln in lines if ln.strip().split(":")[0].strip() in _CHECKS]
    summary = "; ".join(verdicts) if verdicts else (proc.stderr.strip()[-200:] or "no output")
    return ("PASS " if proc.returncode == 0 else "FAIL ") + summary


def _trial_csv_path(out, idx):
    """Per-trial curve CSV, merged into the sweep's own CSV by the parent."""
    return Path(out).parent / f".{Path(out).stem}_trial_{idx}.csv"


def _trial_json_path(out, idx):
    """Per-trial result handoff: final score and curve, parent-readable."""
    return Path(out).parent / f".{Path(out).stem}_trial_{idx}.json"


def _spawn_trial(idx, args):
    """Run one trial in a CHILD PROCESS and read back its result.

    Not an optimization. A camera-backed env can only be built once per
    process -- gz-sim's rendering scene is a process-wide singleton that
    close() does not tear down -- so a sweep that trained every trial in the
    orchestrator would raise on its second image-agent trial. Every trial
    therefore gets a fresh interpreter, and results cross the boundary on disk
    (the model via _trial_model_path, the score and curve via a small JSON).

    :param idx: index into the config grid.
    :param args: the parsed sweep arguments.
    :return: (final score, curve) -- the model stays on disk.
    """
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--agent",
        args.agent,
        "--algo",
        args.algo,
        "--n-agents",
        str(args.n_agents),
        "--seed",
        str(args.seed),
        "--timesteps",
        str(args.timesteps),
        "--track-shapes",
        args.track_shapes,
        "--out",
        str(args.out),
        "--only-trial",
        str(idx),
        "--run-one-trial",  # the child marker; without it this would recurse
    ]
    for flag, value in (
        ("--lr", args.lr),
        ("--gamma", args.gamma),
        ("--epochs", args.epochs),
        ("--clip-range", args.clip_range),
        ("--checkpoint-every", args.checkpoint_every),
        ("--init-from", args.init_from),
    ):
        if value is not None:
            cmd += [flag, str(value)]
    if args.anneal:
        cmd.append("--anneal")
    # "=" form, not two tokens: a value like "-y" is read as a FLAG when passed
    # separately, and argparse rejects the whole command.
    cmd.append(f"--forward-axis={args.forward_axis}")
    proc = subprocess.run(cmd, text=True)
    result = _trial_json_path(args.out, idx)
    if proc.returncode != 0 or not result.exists():
        print(f"[trial {idx}] FAILED (rc={proc.returncode}); skipped")
        return None, None
    payload = json.loads(result.read_text())
    return payload["final"], payload["curve"]


def run_trial(idx, cfg, args, writer):
    """Train one config; save its model to a temp per-trial path.

    Returns the final score and curve, not the model object -- this always
    runs as its own subprocess (see _spawn_trial), so the model has to cross
    that boundary via disk, not memory.

    :param idx: index into the config grid, used for every per-trial path.
    :param cfg: the hyperparameter dict for this trial.
    :param args: the parsed sweep arguments.
    :param writer: csv.DictWriter the CurveLogger snapshots into.
    :return: (final mean episode reward, learning curve).
    """
    wb = None
    if args.wandb:
        try:
            import wandb

            wb = wandb
            wb.init(project=args.project, name=f"trial{idx}", config=cfg, reinit=True)
        except Exception as exc:  # not installed / not authed
            print(f"  [wandb disabled: {exc}]")
            wb = None

    # Same seed across every trial in a sweep (fixed by --seed, not per-trial)
    # so differences between configs reflect the hyperparameters, not which
    # trial happened to get luckier env-reset/weight-init randomness -- this
    # was previously fully unseeded (a real confound: rerunning the same
    # config, or comparing two configs, gave different results purely from
    # randomness). Still one sample per config, not averaged over seeds, but
    # deterministic and fairly-compared beats random.
    agent_name = register_track_randomized(args.agent, args.track_shapes)
    env = VecMonitor(make_inprocess(agent_name, n_agents=args.n_agents, seed=args.seed))
    env, policy = wrap_for_observations(env)
    if args.lr is not None:
        cfg = dict(cfg, learning_rate=args.lr)
    if args.gamma is not None:
        cfg = dict(cfg, gamma=args.gamma)
    if args.epochs is not None:
        # Vision PPO in the paper uses 3 epochs (Table 5); this grid's 6-10
        # come from the continuous-control setting (Table 3), which reuses each
        # rollout far more aggressively than an image task should.
        cfg = dict(cfg, n_epochs=args.epochs)
    if args.anneal and args.algo in ("ppo",):
        cfg = dict(
            cfg,
            learning_rate=linear_schedule(cfg["learning_rate"]),
            clip_range=linear_schedule(args.clip_range),
        )
    model = _ALGOS[args.algo](policy, env, verbose=0, seed=args.seed, **cfg)
    if args.init_from:
        # Fine-tune from a real checkpoint rather than fresh weights: a short
        # from-scratch screen on a hard task measures nothing, because every
        # trial fails for the same reason (not enough budget to solve the BASE
        # task) and the grid then ranks noise.
        #
        # Copy the WEIGHTS into a freshly-constructed model rather than calling
        # .load(**cfg). load() applies unknown kwargs with __dict__.update, so
        # a swept learning_rate never reaches the lr_schedule that training
        # actually reads, and a swept n_steps never rebuilds the rollout
        # buffer -- the sweep would silently compare identical runs. This also
        # starts each trial with fresh optimizer state, which is deliberate:
        # continuing an already-many-times-continued checkpoint is this
        # project's leading suspect for accumulated training fragility.
        donor = _ALGOS[args.algo].load(args.init_from, device="auto")
        model.policy.load_state_dict(donor.policy.state_dict())
        del donor
    cb = CurveLogger(max(2000, args.timesteps // 25), idx, writer, wb)
    if args.checkpoint_every:
        # Fine-tuning an already-good policy can peak and then destroy it --
        # measured on this task, where four independent seeds all regressed
        # from 83.3% to 60/27/0/0 over 500k steps. Saving only the FINAL model
        # then throws the good policy away. Keep periodic snapshots so the
        # best point can be recovered by eval instead of hoped for.
        ckpt_dir = Path(args.out).parent / f".{Path(args.out).stem}_trial_{idx}_ckpts"
        # SB3 counts save_freq in CALLS, and each call advances num_timesteps
        # by n_envs -- so passing the raw value snapshots n_agents times less
        # often than the flag name promises (with 4 agents, --checkpoint-every
        # 20000 really saved every 80k steps). Divide so the flag means
        # timesteps, which is what the filenames report.
        every = max(1, args.checkpoint_every // max(1, args.n_agents))
        cb = [cb, CheckpointCallback(every, str(ckpt_dir), name_prefix="ck")]
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.timesteps, callback=cb)
    dt = time.perf_counter() - t0

    final = np.mean([e["r"] for e in model.ep_info_buffer]) if model.ep_info_buffer else 0.0
    model.save(_trial_model_path(args.out, idx))
    env.close()
    if wb is not None:
        wb.finish()
    print(
        f"[trial {idx}] {cfg}\n    -> final mean_ep_reward={final:.1f} "
        f"({args.timesteps} steps in {dt:.0f}s)"
    )
    curve = cb[0].curve if isinstance(cb, list) else cb.curve
    return float(final), curve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", default="cartpole")
    p.add_argument("--algo", default="ppo", choices=sorted(_ALGOS))
    p.add_argument("--n-agents", type=int, default=16)
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="same seed used for every trial -- fair comparison, not variance-over-seeds",
    )
    p.add_argument("--timesteps", type=int, default=250000)
    p.add_argument("--solved", type=float, default=475.0, help="mean-reward bar counted as solved")
    p.add_argument(
        "--track-shapes",
        default="off",
        choices=TRACK_SHAPE_MODES,
        help="track-shape/spawn randomization for image agents (see train.py)",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="trials to run concurrently. Each trial is already its own "
        "process, so this is bounded by cores (~3 per env) and, for image "
        "agents, by how many worlds can render at once before frames start "
        "arriving late",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="override every config's learning_rate (e.g. to fine-tune an "
        "already-good policy without destroying it)",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="save a snapshot every N steps so a peak-then-regress run can be "
        "recovered (0 = final model only)",
    )
    p.add_argument(
        "--anneal",
        action="store_true",
        help="decay learning rate AND clip range linearly to 0 over training, "
        "as the PPO paper does for its vision experiments (Table 5). PPO only",
    )
    p.add_argument(
        "--clip-range",
        type=float,
        default=0.1,
        help="initial PPO clip range when --anneal is set (paper's vision "
        "value is 0.1; SB3's unannealed default is 0.2)",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="override every config's discount. The default grid uses 0.98-0.99, "
        "whose effective horizon is 50-100 steps; a lap of this track is ~444 "
        "steps, so reward from completing it is discounted to ~0.007 and the "
        "value function effectively cannot see the finish. Use 0.999 (horizon "
        "1000) when the goal is completing the track rather than surviving the "
        "next few seconds",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override every config's n_epochs (the paper uses 3 for vision, "
        "10 for state-based continuous control)",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="after each trial, run validate_rover.py against its model: "
        "camera pitched 45 degrees, rover actually advancing, and doing so at "
        "a speed the hardware could execute. Reward alone sees none of these "
        "(a creeping policy scores well), so a trial can win the sweep and "
        "still fail here",
    )
    p.add_argument(
        "--forward-axis",
        default="y",
        help="body axis the chassis advances along, passed to validate_rover.py "
        "with --validate. rover_bare is +y; rover_line_bare carries its lens on "
        "the other end and drives along -y, so its check FAILS on the default",
    )
    p.add_argument(
        "--best-out",
        default=None,
        metavar="PATH",
        help="where to save the winning model (default: "
        "models/<agent>_<algo>_sweep_best.zip). Concurrent sweeps of the same "
        "agent MUST pass this, or they overwrite each other's winner",
    )
    p.add_argument(
        "--only-trial",
        type=int,
        default=None,
        metavar="IDX",
        help="run just this config index from the grid (e.g. to give a sweep "
        "winner a longer budget, or to run it again under other seeds) "
        "instead of the whole grid",
    )
    p.add_argument(
        "--init-from",
        default=None,
        metavar="CHECKPOINT",
        help="fine-tune every trial from this .zip instead of fresh weights",
    )
    p.add_argument(
        "--run-one-trial",
        action="store_true",
        help=argparse.SUPPRESS,  # internal: marks a child spawned by _spawn_trial
    )
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--project", default="gazebo-cartpole")
    p.add_argument("--out", default=None, help="CSV path (default: models/)")
    p.add_argument(
        "--push-to-hub",
        metavar="REPO_ID",
        default=None,
        help="upload the BEST model + its config and learning curve "
        "to this Hugging Face Hub repo (user/name)",
    )
    p.add_argument(
        "--hub-private",
        action="store_true",
        help="create the Hub repo as private (with --push-to-hub)",
    )
    args = p.parse_args()

    models = Path(__file__).resolve().parent.parent / "models"
    models.mkdir(exist_ok=True)
    # Resolved here and written back onto args, because run_trial,
    # _trial_model_path and _validate_trial all index args.out as a Path.
    args.out = (
        Path(args.out) if args.out else models / f"sweep_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )
    out = args.out

    # CONFIGS is keyed by ALGO; the per-agent grid is what a sweep runs over.
    grid = _configs_for(args.agent, args.algo)
    chosen = (
        [(args.only_trial, grid[args.only_trial])]
        if args.only_trial is not None
        else list(enumerate(grid))
    )

    # --- child: train exactly one trial, hand the result back on disk. ---
    if args.run_one_trial:
        idx, cfg = chosen[0]
        with open(_trial_csv_path(out, idx), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["trial", "step", "mean_ep_reward"])
            writer.writeheader()
            final, curve = run_trial(idx, cfg, args, writer)
        _trial_json_path(out, idx).write_text(json.dumps({"final": final, "curve": curve}))
        return

    # --- parent: one child process per trial, --jobs of them at a time. ---
    print(
        f"Sweep: {len(chosen)} configs x {args.timesteps} steps, "
        f"n_agents={args.n_agents}, jobs={args.jobs}. Logging to {out}"
    )
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        # Threads, not processes: each one only waits on a child subprocess.
        results = list(pool.map(lambda ic: _spawn_trial(ic[0], args), chosen))

    # Merge the children's curve CSVs into the sweep's own, in trial order.
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["trial", "step", "mean_ep_reward"])
        writer.writeheader()
        for idx, _cfg in chosen:
            child_csv = _trial_csv_path(out, idx)
            if not child_csv.exists():
                continue
            with open(child_csv, newline="") as cf:
                for row in csv.DictReader(cf):
                    writer.writerow(row)

    best, best_idx, best_cfg, best_curve = -1.0, -1, None, None
    for (idx, cfg), (final, curve) in zip(chosen, results, strict=True):
        if final is not None and final > best:
            best, best_idx, best_cfg, best_curve = final, idx, cfg, curve

    if best_idx >= 0:
        best_path = (
            Path(args.best_out) if args.best_out else models / f"{args.agent}_sweep_best.zip"
        )
        best_path.parent.mkdir(parents=True, exist_ok=True)
        # The winner is on disk, saved by its own child; copy it into place.
        best_path.write_bytes(_trial_model_path(out, best_idx).read_bytes())
        status = "SOLVED" if best >= args.solved else "did not reach bar"
        print(
            f"\nBest: trial {best_idx} mean_ep_reward={best:.1f} [{status} "
            f"@ {args.solved}] -> saved {best_path}"
        )
        print(f"Curves in {out}")
        if args.validate:
            # Reward cannot see whether the rover actually drove; this can.
            for idx, _cfg in chosen:
                print(f"[validate trial {idx}] {_validate_trial(idx, args)}")
        if args.push_to_hub:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from hub import push_to_hub

            hp = dict(best_cfg)
            hp["timesteps"] = args.timesteps
            url = push_to_hub(
                best_path,
                args.push_to_hub,
                agent=args.agent,
                algo=args.algo,
                n_agents=args.n_agents,
                hyperparams=hp,
                curve=best_curve,
                eval_result=f"**Best of {len(chosen)} swept configs** "
                f"(trial {best_idx}): mean episode reward "
                f"{best:.1f} [{status} @ {args.solved}].",
                private=args.hub_private,
            )
            print(f"Pushed best model to Hugging Face Hub: {url}")
    else:
        print("\nNo trial produced a usable model.")


if __name__ == "__main__":
    main()
