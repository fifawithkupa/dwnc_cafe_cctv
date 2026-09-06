#!/usr/bin/env bash
# Start a timed live-mode run on the box with the CPU/memory sampler beside it.
# Usage: ~/seatnow/edge/run_live.sh <name> <run_seconds> [extra seatnow args...]
# Needs the fake camera (edge/fake_camera.sh) or a real camera URL in place of 127.0.0.1.
set -u
cd ~/seatnow
NAME="$1"; RUN_SECONDS="$2"; shift 2
pkill -f "live_sampler.py" 2>/dev/null
pkill -f "engine.seatnow rtsp" 2>/dev/null
sleep 1
mkdir -p results/live
nohup python3 ~/seatnow/edge/live_sampler.py "results/live/sample_${NAME}.csv" 10 > /dev/null 2>&1 < /dev/null &
disown
nohup ./venv/bin/python -m engine.seatnow rtsp://127.0.0.1:8554/seatnow \
  --layout layouts/cafe_angle1.json \
  --det-model yolov8n_openvino_model --pose-model yolov8n-pose_openvino_model \
  --log "results/live/run_${NAME}.jsonl" --no-video --run-seconds "$RUN_SECONDS" "$@" \
  > "results/live/run_${NAME}.txt" 2>&1 < /dev/null &
disown
sleep 3
date
pgrep -af "sampler.py|engine.seatnow rtsp" | grep -v pgrep | cut -c1-120
