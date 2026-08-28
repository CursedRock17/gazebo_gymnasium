#!/usr/bin/env bash
# Two isolated follow-ups to the anneal+ep3 baseline (best so far: 1.85 m).
#
# Diagnosis being tested: the policy never learned to steer. It commands
# [+1.00,+1.00] for the whole final stretch while cross-track error blows out
# from +0.20 to -0.98. Two candidate causes, one run each.
#
#   corners  ~81% of every episode is straight-line driving, because every
#            episode spawns at the same fixed mid-straight point. The policy
#            converges correctly on the data it is given; there is barely any
#            corner in it. --track-shapes reset draws a fresh shape AND a
#            tangent-aligned spawn every reset, so corners arrive immediately.
#   pivot    the shipped action map caps the turn radius at 0.12 m and forbids
#            reversing a wheel. RoboCup Junior practice for 90 degree corners
#            is explicitly to reverse the inner wheel. line_follower_pivot
#            constrains only the MEAN wheel speed, giving a 0.02 m radius.
#
# NOT tested separately: ent_coef. The grid already spans 0.0 to 0.01 and the
# four trials landed within 1.5 reward points of each other, so it is not the
# lever.
#
# No pgrep wait loop here on purpose -- see next_experiments.sh for why.
set -u
cd /home/usmsm_1302_1/Documents/gazebo_gymnasium

run () {  # name, agent, extra flags...
  local name="$1"; local agent="$2"; shift 2
  mkdir -p "models/$name"
  echo "[chain] === $name (agent=$agent) : $* === $(date -Is)"
  pixi run sweep --agent "$agent" --algo ppo --n-agents 4 \
      --timesteps 400000 --seed 0 --jobs 4 --checkpoint-every 25000 --validate \
      --anneal --clip-range 0.1 --epochs 3 "$@" \
      --best-out "models/$name/ppo_sweep_best.zip" --out "models/$name/sweep.csv" \
      > "results/${name}.log" 2>&1
  echo "[chain] $name finished rc=$? at $(date -Is)"
  grep -E "final mean|Best|trial [0-9]:" "results/${name}.log" | grep -v Warning || true
  echo "[chain] --- distance eval for $name ---"
  pixi run bash -c "source install/setup.sh && python training_scripts/dr_eval.py \
      models/$name/ppo_sweep_best.zip --agent $agent --track-shapes ${TS:-off} \
      --n-agents 8 --rounds 2 --solved-distance 8.0" 2>&1 | grep -E "^EVAL" || true
}

TS=reset run lf_corners       line_follower       --track-shapes reset
TS=off   run lf_pivot         line_follower_pivot --track-shapes off

echo "[chain] ALL RUNS COMPLETE at $(date -Is)"
