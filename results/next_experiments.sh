#!/usr/bin/env bash
# Auto-chained PPO-setting follow-ups to the speed-scaled-reward baseline.
#
# Run A isolates LR+clip annealing (PPO paper Table 5's alpha schedule).
# Run B adds the paper's vision epoch count (3) on top of A.
# Kept separate so each is attributable: running both at once would leave us
# unable to say which one moved the number.
#
# There is deliberately NO "wait for the previous sweep" pgrep loop here. The
# launcher process that starts this script carries the whole script text on
# its own command line, so `pgrep -f "<sweep command>"` matches ITSELF and the
# loop never exits. That bug cost 90 minutes of idle machine on 2026-08-28.
# Launch this only when the machine is already free.
set -u
cd /home/usmsm_1302_1/Documents/gazebo_gymnasium

run () {  # name, extra flags...
  local name="$1"; shift
  mkdir -p "models/$name"
  echo "[chain] === $name : $* === $(date -Is)"
  pixi run sweep --agent line_follower --algo ppo --n-agents 4 \
      --timesteps 400000 --seed 0 --jobs 4 --checkpoint-every 25000 --validate \
      "$@" \
      --best-out "models/$name/ppo_sweep_best.zip" --out "models/$name/sweep.csv" \
      > "results/${name}.log" 2>&1
  echo "[chain] $name finished rc=$? at $(date -Is)"
  grep -E "final mean|Best|trial [0-9]:" "results/${name}.log" | grep -v Warning || true
  echo "[chain] --- distance eval for $name ---"
  pixi run bash -c "source install/setup.sh && python training_scripts/dr_eval.py \
      models/$name/ppo_sweep_best.zip --agent line_follower --track-shapes off \
      --n-agents 8 --rounds 2 --solved-distance 8.0" 2>&1 | grep -E "^EVAL" || true
}

run lf_anneal --anneal --clip-range 0.1
run lf_anneal_ep3 --anneal --clip-range 0.1 --epochs 3

echo "[chain] ALL RUNS COMPLETE at $(date -Is)"
