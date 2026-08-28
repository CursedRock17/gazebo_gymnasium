#!/bin/bash
cd /home/usmsm_1302_1/Documents/gazebo_gymnasium
OUT=results/verify100.txt
# Wait for the phase3 eval wave to clear so numbers are uncontended.
while pgrep -f dr_eval.py > /dev/null; do sleep 30; done
echo "=== labeled verification $(date -Is) ===" >> $OUT
for CK in models/.phase3_*_trial_3_ckpts/ck_200000_steps.zip \
          models/.phase3_*_trial_3_ckpts/ck_100000_steps.zip; do
  [ -f "$CK" ] || continue
  TAG=$(echo "$CK" | sed 's#models/\.phase3_##; s#_trial_3_ckpts/#:#')
  R=$(timeout 3600 pixi run -e default bash -c \
      "source install/setup.sh && python training_scripts/dr_eval.py $CK --n-agents 8 --rounds 5 --track-shapes reset" \
      2>&1 | grep -E "^EVAL")
  echo "$TAG -> $R" >> $OUT
done
echo "=== verification done $(date -Is) ===" >> $OUT
