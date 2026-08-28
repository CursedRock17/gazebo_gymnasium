#!/bin/bash
# Waits for the seeded continuations, then evaluates every resulting
# checkpoint with the per-agent eval. Detached: survives the agent session.
cd /home/usmsm_1302_1/Documents/gazebo_gymnasium
OUT=results/multitrack_eval.txt
echo "=== pipeline started $(date -Is) ===" >> $OUT

# 1. wait for the continuation sweeps to exit
while pgrep -f "only-trial" > /dev/null; do sleep 60; done
echo "--- continuations finished $(date -Is) ---" >> $OUT
grep -h "final mean_ep_reward" /tmp/claude-1000/-home-usmsm-1302-1-Documents-gazebo-gymnasium/e777b9e9-48db-446f-b12d-cc4b6d5c8c6b/scratchpad/cont_seed*.log >> $OUT 2>/dev/null

# 2. evaluate each seed's checkpoint, 2 at a time (concurrency is safe now
#    that camera topics are namespaced per process)
for S in 0 1 2 3; do
  M=models/line_follower_multi/multitrack_cont_seed$S.zip
  [ -f "$M" ] || { echo "seed$S: NO CHECKPOINT (trial failed?)" >> $OUT; continue; }
  ( timeout 3600 pixi run -e default bash -c \
      "source install/setup.sh && python training_scripts/dr_eval.py $M --n-agents 8 --rounds 5 --track-shapes reset" \
      2>&1 | grep -E "^EVAL" | sed "s/^/seed$S /" >> $OUT ) &
  if [ $((S % 2)) -eq 1 ]; then wait; fi
done
wait
echo "=== pipeline done $(date -Is) ===" >> $OUT
