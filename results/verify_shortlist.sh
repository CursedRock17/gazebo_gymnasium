#!/bin/bash
# Deliberate 80-episode verification of the most CONSISTENT candidates.
# phase5.sh's own picker grabbed whatever its fragile sort emitted; a snapshot
# that screens 100% while its neighbours swing 62-95% is likely a lucky
# window, whereas lr3e-5_s0 screened 100% at seven separate points.
cd /home/usmsm_1302_1/Documents/gazebo_gymnasium
OUT=results/verify_shortlist.txt
while pgrep -f dr_eval.py > /dev/null; do sleep 30; done
echo "=== shortlist verification $(date -Is), 10 rounds x 8 agents = 80 episodes, idle machine ===" >> $OUT
for CK in models/.phase5_lr3e-5_s0_trial_3_ckpts/ck_180000_steps.zip \
          models/.phase5_lr3e-5_s0_trial_3_ckpts/ck_120000_steps.zip \
          models/.phase5_lr3e-5_s0_trial_3_ckpts/ck_100000_steps.zip \
          models/.phase5_lr5e-5_s0_trial_3_ckpts/ck_200000_steps.zip; do
  [ -f "$CK" ] || { echo "MISSING $CK" >> $OUT; continue; }
  R=$(timeout 7200 pixi run -e default bash -c \
      "source install/setup.sh && python training_scripts/dr_eval.py $CK --n-agents 8 --rounds 10 --track-shapes reset" \
      2>&1 | grep -oP 'solved=\K[0-9]+/[0-9]+ \([0-9.]+%\)')
  echo "VERIFY80 $CK -> $R" >> $OUT
done
echo "=== shortlist done $(date -Is) ===" >> $OUT
