#!/bin/bash
# Phase 4: push 97.5% -> 100%, continuing from the best checkpoint.
# lr=5e-5 is where phase 3's gains came from, so explore around it rather than
# re-run the same point. Snapshots every 20k because this task's peaks are
# transient (phase 3 saw 95.8% -> 100% -> 62.5% inside one run).
cd /home/usmsm_1302_1/Documents/gazebo_gymnasium
BASE=models/line_follower_multi/multitrack_lr5e5_200k_975.zip
OUT=results/phase4_eval.txt
echo "=== phase4 started $(date -Is) base=$BASE (97.5%) ===" >> $OUT

for spec in "5e-5:0" "5e-5:1" "3e-5:0" "8e-5:0"; do
  LR=${spec%%:*}; S=${spec##*:}; TAG=lr${LR}_s${S}
  nohup timeout 21600 pixi run sweep --agent line_follower --algo ppo --n-agents 4 \
    --track-shapes reset --jobs 1 --only-trial 3 --seed $S --lr $LR \
    --timesteps 200000 --checkpoint-every 20000 --solved 300 --init-from $BASE \
    --out models/phase4_$TAG.csv \
    --best-out models/line_follower_multi/phase4_${TAG}_final.zip \
    > results/phase4_$TAG.log 2>&1 &
done
wait
echo "--- training done $(date -Is) ---" >> $OUT

# Screen every snapshot at 3 rounds, 2 at a time. Label with the RUN, not just
# the basename -- phase 3 logged bare filenames and the winner was ambiguous.
i=0
for CK in models/.phase4_*_trial_3_ckpts/*.zip models/line_follower_multi/phase4_*_final.zip; do
  [ -f "$CK" ] || continue
  TAG=$(echo "$CK" | sed 's#models/\.phase4_##; s#models/line_follower_multi/##; s#_trial_3_ckpts/#:#')
  ( R=$(timeout 3600 pixi run -e default bash -c \
        "source install/setup.sh && python training_scripts/dr_eval.py $CK --n-agents 8 --rounds 3 --track-shapes reset" \
        2>&1 | grep -oP 'solved=\K[0-9]+/[0-9]+ \([0-9.]+%\)')
    echo "SCREEN $TAG -> $R  [$CK]" >> $OUT ) &
  i=$((i+1)); [ $((i % 2)) -eq 0 ] && wait
done
wait
echo "--- screen done $(date -Is) ---" >> $OUT

# Re-verify the top 3 at 40 episodes, SEQUENTIALLY on an idle machine: small
# samples over-report here (a 24-ep 100% became 97.5% at 40 ep), and
# concurrent evals cost a few more points.
for CK in $(grep "^SCREEN" $OUT | sed 's/.*\[\(.*\)\]/\1/;' | head -100 | \
            paste -d' ' - <(grep "^SCREEN" $OUT | grep -oP '\(\K[0-9.]+') 2>/dev/null | \
            sort -k2 -rn | head -3 | cut -d' ' -f1); do
  [ -f "$CK" ] || continue
  R=$(timeout 3600 pixi run -e default bash -c \
      "source install/setup.sh && python training_scripts/dr_eval.py $CK --n-agents 8 --rounds 5 --track-shapes reset" \
      2>&1 | grep -oP 'solved=\K[0-9]+/[0-9]+ \([0-9.]+%\)')
  echo "VERIFY40 $CK -> $R" >> $OUT
done
echo "=== phase4 done $(date -Is) ===" >> $OUT
