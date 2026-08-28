#!/bin/bash
# Definitive A/B for the spawn-redraw fix, on an IDLE machine (contention
# costs several points, and we are measuring a ~2.5% effect).
cd /home/usmsm_1302_1/Documents/gazebo_gymnasium
OUT=results/spawnfix_ab.txt
M=models/line_follower_multi/multitrack_lr5e5_200k_975.zip
while pgrep -f "only-trial|dr_eval.py" > /dev/null; do sleep 60; done
echo "=== spawn-redraw A/B $(date -Is), 10 rounds x 8 agents = 80 episodes each ===" >> $OUT
for RD in 0 3; do
  R=$(timeout 7200 pixi run -e default bash -c \
      "source install/setup.sh && python training_scripts/dr_eval.py $M --n-agents 8 --rounds 10 --track-shapes reset --spawn-redraw $RD" \
      2>&1 | grep -oP 'solved=\K[0-9]+/[0-9]+ \([0-9.]+%\)')
  echo "spawn_redraw=$RD -> $R" >> $OUT
done
echo "=== done $(date -Is) ===" >> $OUT
