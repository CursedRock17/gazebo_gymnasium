#!/usr/bin/env bash
# Testing the two specification changes, isolated, plus the action map.
#
# Baseline to beat: anneal+ep3, median episode 75 steps / 1.85 m, 0/16 laps.
#
#   lf_aux        scan-band features added to the OBSERVATION (Dict of image +
#                 10 features, MultiInputPolicy). Grid gamma (0.98/0.99), so
#                 this vs the baseline isolates the observation change.
#   lf_aux_g999   adds gamma 0.999. vs lf_aux this isolates the discount. A lap
#                 is ~444 steps; at gamma 0.99 reward from completing it is
#                 discounted to 0.0066, so the value function literally could
#                 not see the finish line. This is the change that makes
#                 "complete it slower rather than crash fast" expressible.
#   lf_pivot_g999 the mean-forward action map (0.02 m turn radius, inner wheel
#                 may reverse) under the best discount. Its earlier run was
#                 corrupted by live source edits and never actually trained.
#
# IMPORTANT: build/ SYMLINKS to source (colcon --symlink-install), so editing
# any file under gazebo_gymnasium_bridge/ takes effect IMMEDIATELY in running
# trials. Do not touch source while this is running.
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
      models/$name/ppo_sweep_best.zip --agent $agent --track-shapes off \
      --n-agents 8 --rounds 2 --solved-distance 8.0" 2>&1 | grep -E "^EVAL" || true
}

run lf_aux        line_follower
run lf_aux_g999   line_follower       --gamma 0.999
run lf_pivot_g999 line_follower_pivot --gamma 0.999

echo "[chain] ALL RUNS COMPLETE at $(date -Is)"
