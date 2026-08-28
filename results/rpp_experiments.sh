#!/usr/bin/env bash
# Regulated-Pure-Pursuit speed cap, on both action maps.
#
# Evidence this is testing: the previous best policy DOES corner (steer |L-R|
# 0.171, 68 deg of yaw swept) but enters the corner at [+1.00,+1.00] and only
# starts steering 4 steps before it dies, committing to a 0.39 m turn radius
# against a corner needing 0.12 m. It overshoots to 0.27 m outside the track.
# It will not slow down, because slowing costs immediate reward.
#
# The speed cap removes the incentive to charge without paying for crawling.
# Calibrated from measurement: straights read |curvature| ~0.09 (cap 0.43 m/s),
# the corner peaks at 0.55 (cap 0.10 m/s, i.e. fully damped).
#
#   lf_rpp        forward-only map, 0.12 m minimum turn radius
#   lf_rpp_pivot  mean-forward map, 0.02 m radius, inner wheel may reverse.
#                 Now worth a fair test: the reward no longer punishes the
#                 slowdown a tight turn needs, and RoboCup practice for a 90
#                 degree corner is explicitly to reverse the inner wheel.
#
# build/ SYMLINKS to source -- do not edit anything under
# gazebo_gymnasium_bridge/ while this runs.
set -u
cd /home/usmsm_1302_1/Documents/gazebo_gymnasium

run () {
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
      models/$name/ppo_sweep_best.zip --agent $agent --track-shapes off \
      --n-agents 8 --rounds 2 --solved-distance 8.0" 2>&1 | grep -E "^EVAL" || true
}

run lf_rpp       line_follower
run lf_rpp_pivot line_follower_pivot

echo "[chain] ALL RUNS COMPLETE at $(date -Is)"
