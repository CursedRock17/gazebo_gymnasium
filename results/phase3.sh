#!/bin/bash
# Phase 3: low-lr fine-tune from the 83.3% checkpoint, WITH periodic snapshots.
# Four 500k continuations at lr=3e-4 destroyed the policy (83.3% -> 60/27/0/0),
# so this drops the lr by 6-15x, shortens the budget, and keeps snapshots so the
# peak can be recovered by eval instead of trusting the final model.
cd /home/usmsm_1302_1/Documents/gazebo_gymnasium
BASE=models/line_follower_multi/multitrack_sweep_w3_300k.zip
OUT=results/phase3_eval.txt
echo "=== phase3 started $(date -Is) ===" >> $OUT

for spec in "5e-5:0" "5e-5:1" "2e-5:0" "2e-5:1"; do
  LR=${spec%%:*}; S=${spec##*:}
  TAG=lr${LR}_s${S}
  nohup timeout 21600 pixi run sweep --agent line_follower --algo ppo --n-agents 4 \
    --track-shapes reset --jobs 1 --only-trial 3 --seed $S --lr $LR \
    --timesteps 250000 --checkpoint-every 25000 --solved 300 --init-from $BASE \
    --out models/phase3_$TAG.csv \
    --best-out models/line_follower_multi/phase3_${TAG}_final.zip \
    > results/phase3_$TAG.log 2>&1 &
done
wait
echo "--- training done $(date -Is) ---" >> $OUT

# Evaluate EVERY snapshot (not just the final model) two at a time, so a
# peak-then-regress run still yields its best point.
i=0
for CK in models/.phase3_*_trial_3_ckpts/*.zip models/line_follower_multi/phase3_*_final.zip; do
  [ -f "$CK" ] || continue
  ( timeout 3600 pixi run -e default bash -c \
      "source install/setup.sh && python training_scripts/dr_eval.py $CK --n-agents 8 --rounds 3 --track-shapes reset" \
      2>&1 | grep -E "^EVAL" >> $OUT ) &
  i=$((i+1)); [ $((i % 2)) -eq 0 ] && wait
done
wait
echo "=== phase3 done $(date -Is) ===" >> $OUT
sort -t'(' -k2 -rn $OUT | grep "^EVAL" | head -3 >> $OUT
