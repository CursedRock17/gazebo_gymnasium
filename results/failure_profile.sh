#!/bin/bash
# With spawns now always valid, WHERE do the last ~2.5% of episodes fail?
# Early termination would mean a reset/first-frame issue still; mid-episode
# would mean genuine driving difficulty (a corner the policy cannot hold).
cd /home/usmsm_1302_1/Documents/gazebo_gymnasium
OUT=results/failure_profile.txt
while pgrep -f dr_eval.py > /dev/null; do sleep 30; done
for CK in models/.phase5_lr3e-5_s0_trial_3_ckpts/ck_120000_steps.zip \
          models/line_follower_multi/multitrack_lr5e5_200k_975.zip; do
  echo "=== $CK ===" >> $OUT
  timeout 7200 pixi run -e default bash -c \
    "source install/setup.sh && python training_scripts/dr_eval.py $CK --n-agents 8 --rounds 10 --track-shapes reset" \
    2>&1 | grep -E "^  round|^EVAL" >> $OUT
done
echo "=== failure profile done $(date -Is) ===" >> $OUT
