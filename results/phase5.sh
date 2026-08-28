#!/bin/bash
# Phase 5: first training run with spawn_redraw_attempts active, so no episode
# starts on an unwinnable blank view. Continues from the 97.5% checkpoint.
# lr=5e-5 produced that checkpoint, so bracket it rather than re-run it.
cd /home/usmsm_1302_1/Documents/gazebo_gymnasium
BASE=models/line_follower_multi/multitrack_lr5e5_200k_975.zip
OUT=results/phase5_eval.txt
echo "=== phase5 started $(date -Is) base=97.5% spawn_redraw=ON ===" >> $OUT

for spec in "5e-5:0" "5e-5:1" "3e-5:0" "8e-5:0"; do
  LR=${spec%%:*}; S=${spec##*:}; TAG=lr${LR}_s${S}
  nohup timeout 21600 pixi run sweep --agent line_follower --algo ppo --n-agents 4 \
    --track-shapes reset --jobs 1 --only-trial 3 --seed $S --lr $LR \
    --timesteps 200000 --checkpoint-every 20000 --solved 300 --init-from $BASE \
    --out models/phase5_$TAG.csv \
    --best-out models/line_follower_multi/phase5_${TAG}_final.zip \
    > results/phase5_$TAG.log 2>&1 &
done
wait
echo "--- training done $(date -Is) ---" >> $OUT

# Screen every snapshot at 3 rounds (2 at a time), labeled by RUN not basename.
i=0
for CK in models/.phase5_*_trial_3_ckpts/*.zip models/line_follower_multi/phase5_*_final.zip; do
  [ -f "$CK" ] || continue
  TAG=$(echo "$CK" | sed 's#models/\.phase5_##; s#models/line_follower_multi/##; s#_trial_3_ckpts/#:#')
  ( R=$(timeout 3600 pixi run -e default bash -c \
        "source install/setup.sh && python training_scripts/dr_eval.py $CK --n-agents 8 --rounds 3 --track-shapes reset" \
        2>&1 | grep -oP 'solved=\K[0-9]+/[0-9]+ \([0-9.]+%\)')
    echo "SCREEN $TAG -> $R  [$CK]" >> $OUT ) &
  i=$((i+1)); [ $((i % 2)) -eq 0 ] && wait
done
wait
echo "--- screen done $(date -Is) ---" >> $OUT

# Verify the top screens at 80 episodes, SEQUENTIALLY on an idle machine.
# Small samples over-report here: a 24-episode 100% became 97.5% at 40.
for CK in $(grep "^SCREEN" $OUT | grep -oP '\(\K[0-9.]+(?=%\))\s*|\[.*\]' | paste - - 2>/dev/null | sort -rn | head -3 | grep -oP '(?<=\[).*(?=\])'); do
  [ -f "$CK" ] || continue
  R=$(timeout 7200 pixi run -e default bash -c \
      "source install/setup.sh && python training_scripts/dr_eval.py $CK --n-agents 8 --rounds 10 --track-shapes reset" \
      2>&1 | grep -oP 'solved=\K[0-9]+/[0-9]+ \([0-9.]+%\)')
  echo "VERIFY80 $CK -> $R" >> $OUT
done
echo "=== phase5 done $(date -Is) ===" >> $OUT
