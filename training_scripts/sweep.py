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

Trains PPO on the cartpole spec across a grid of hyperparameters, records each
trial's learning curve, and reports the best config + whether it reached the
500-step survival cap. Every trial logs to a CSV (always) and to Weights &
Biases (optionally, if ``--wandb`` is passed and ``wandb`` is installed +
authenticated). Because the sim is in-process and unthrottled, a full sweep
runs in minutes on a laptop CPU.

    python training_scripts/sweep.py --timesteps 250000 --n-agents 16
    python training_scripts/sweep.py --wandb --project gazebo-cartpole
"""

import argparse
import csv
from pathlib import Path
import time

import numpy as np
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor

from gazebo_gymnasium_bridge.envs import make_inprocess
from gazebo_gymnasium_bridge.envs import wrap_for_observations


# Search space — small and hand-picked to reliably solve the cartpole spec.
CONFIGS = [
    {"learning_rate": 3e-4, "ent_coef": 0.01, "n_steps": 64,
     "n_epochs": 6, "batch_size": 256, "gamma": 0.99},
    {"learning_rate": 1e-3, "ent_coef": 0.0, "n_steps": 128,
     "n_epochs": 10, "batch_size": 256, "gamma": 0.99},
    {"learning_rate": 5e-4, "ent_coef": 0.005, "n_steps": 128,
     "n_epochs": 10, "batch_size": 512, "gamma": 0.98},
    {"learning_rate": 3e-4, "ent_coef": 0.0, "n_steps": 256,
     "n_epochs": 10, "batch_size": 512, "gamma": 0.99},
]


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
            self.writer.writerow({"trial": self.trial, "step": self.num_timesteps,
                                  "mean_ep_reward": round(r, 2)})
            if self.wb is not None:
                self.wb.log({"mean_ep_reward": r}, step=self.num_timesteps)
            self._next += self.every
        return True


def run_trial(idx, cfg, args, writer):
    wb = None
    if args.wandb:
        try:
            import wandb
            wb = wandb
            wb.init(project=args.project, name=f"trial{idx}", config=cfg,
                    reinit=True)
        except Exception as exc:            # not installed / not authed
            print(f"  [wandb disabled: {exc}]")
            wb = None

    env = VecMonitor(make_inprocess(args.agent, n_agents=args.n_agents))
    env, policy = wrap_for_observations(env)
    model = sb3.PPO(policy, env, verbose=0, **cfg)
    cb = CurveLogger(max(2000, args.timesteps // 25), idx, writer, wb)
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.timesteps, callback=cb)
    dt = time.perf_counter() - t0

    final = np.mean([e["r"] for e in model.ep_info_buffer]) if \
        model.ep_info_buffer else 0.0
    env.close()
    if wb is not None:
        wb.finish()
    print(f"[trial {idx}] {cfg}\n    -> final mean_ep_reward={final:.1f} "
          f"({args.timesteps} steps in {dt:.0f}s)")
    return final, model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", default="cartpole")
    p.add_argument("--n-agents", type=int, default=16)
    p.add_argument("--timesteps", type=int, default=250000)
    p.add_argument("--solved", type=float, default=475.0,
                   help="mean-reward bar counted as solved")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--project", default="gazebo-cartpole")
    p.add_argument("--out", default=None, help="CSV path (default: models/)")
    args = p.parse_args()

    models = Path(__file__).resolve().parent.parent / "models"
    models.mkdir(exist_ok=True)
    out = Path(args.out) if args.out else models / \
        f"sweep_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    print(f"Sweep: {len(CONFIGS)} configs x {args.timesteps} steps, "
          f"n_agents={args.n_agents}. Logging to {out}")
    best, best_model, best_idx = -1.0, None, -1
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["trial", "step",
                                                "mean_ep_reward"])
        writer.writeheader()
        for idx, cfg in enumerate(CONFIGS):
            final, model = run_trial(idx, cfg, args, writer)
            fh.flush()
            if final > best:
                best, best_model, best_idx = final, model, idx

    if best_model is not None:
        best_path = models / f"{args.agent}_sweep_best.zip"
        best_model.save(best_path)
        status = "SOLVED" if best >= args.solved else "did not reach bar"
        print(f"\nBest: trial {best_idx} mean_ep_reward={best:.1f} [{status} "
              f"@ {args.solved}] -> saved {best_path}")
        print(f"Curves in {out}")


if __name__ == "__main__":
    main()
