#!/bin/bash
# Record a Gazebo GUI video of a deterministic multi-track evaluation.
# The in-process world publishes scene topics, so `gz sim -g` attaches to it
# live -- that matters because the HARNESS backend has no track randomization,
# so its GUI would show the old single fixed track instead of the real task.
set -u
CKPT="$1"; SECONDS_TO_RECORD="${2:-60}"; NAGENTS="${3:-4}"
SP=/tmp/claude-1000/-home-usmsm-1302-1-Documents-gazebo-gymnasium/e777b9e9-48db-446f-b12d-cc4b6d5c8c6b/scratchpad
cd /home/usmsm_1302_1/Documents/gazebo_gymnasium
OUT=docs/images/line_follower_multitrack_eval.mp4
mkdir -p docs/images

nohup pixi run -e default bash -c \
  "source install/setup.sh && python $SP/record_eval.py $CKPT $NAGENTS 12 12" \
  > results/record_eval.log 2>&1 &
EVAL_PID=$!
for _ in $(seq 1 60); do grep -q WORLD_UP results/record_eval.log 2>/dev/null && break; sleep 1; done

DISPLAY=:1 nohup pixi run -e default bash -c \
  "source install/setup.sh && gz sim -g" > results/record_gui.log 2>&1 &
sleep 8   # let the GUI attach to the running world and render a first frame

# This pixi ffmpeg has no x11grab, so capture with PIL + cv2 instead.
pixi run -e default python "$SP/record_screen.py" "$OUT" "$SECONDS_TO_RECORD" 12 :1 0.5 \
  > results/record_capture.log 2>&1

pkill -f 'gz[ ]sim -g' 2>/dev/null
wait $EVAL_PID 2>/dev/null
echo "--- eval result ---"; grep -E "RECORDED_EVAL|episode " results/record_eval.log | tail -14
ls -la "$OUT"
