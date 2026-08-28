#!/bin/bash
# Re-score every meaningful checkpoint WITH progress, sequentially on an idle
# machine. The question: does any of these actually drive the track, or has
# the whole 12.5% -> 97.5% arc been measuring line-keeping?
cd /home/usmsm_1302_1/Documents/gazebo_gymnasium
OUT=results/rescore.txt
while pgrep -f dr_eval.py > /dev/null; do sleep 30; done
echo "=== re-score with progress $(date -Is), 8 agents x 5 rounds ===" >> $OUT
run() {  # label, checkpoint, mode
  [ -f "$2" ] || { echo "MISSING $1" >> $OUT; return; }
  R=$(timeout 3600 pixi run -e default bash -c \
      "source install/setup.sh && python training_scripts/dr_eval.py $2 --n-agents 8 --rounds 5 --track-shapes $3" \
      2>&1 | grep -E "^EVAL")
  echo "[$1] $R" >> $OUT
}
run "multitrack-published"  models/line_follower_multi/multitrack_lr5e5_200k_975.zip           reset
run "phase5-lr3e-5-120k"    models/.phase5_lr3e-5_s0_trial_3_ckpts/ck_120000_steps.zip         reset
run "multitrack-yesterday"  models/line_follower_multi/multitrack_finetune.zip                 reset
run "singletrack-solved"    models/line_follower_multi/retrain_2026-08-21_45deg_ppo_n4_seed0_400k.zip off
run "singletrack-eighthDR"  models/line_follower_multi/eighth_finetune.zip                     off
echo "=== re-score done $(date -Is) ===" >> $OUT
