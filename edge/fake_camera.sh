#!/usr/bin/env bash
# Start (or restart) the fake camera on the box: mediamtx + ffmpeg republisher.
# Usage: ~/seatnow/edge/fake_camera.sh <clip path relative to ~/seatnow>
# Needs mediamtx unpacked in ~/tools (docs/edge-setup.md 11단계).
set -u
cd ~/seatnow
CLIP="${1:-results/edge/clips/4mp_h265.mp4}"
pkill -x mediamtx 2>/dev/null
pkill -f "edge.rtsp_republish" 2>/dev/null
pkill -f "f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/seatnow" 2>/dev/null
sleep 1
nohup ~/tools/mediamtx ~/tools/mediamtx.yml > ~/tools/mediamtx.log 2>&1 < /dev/null &
disown
sleep 2
nohup python3 -m edge.rtsp_republish "$CLIP" --no-server --path seatnow > ~/tools/republish.log 2>&1 < /dev/null &
disown
sleep 4
echo "== procs"
pgrep -af "mediamtx|rtsp_republish|ffmpeg" | grep -v pgrep | cut -c1-160
echo "== port"
ss -ltn | grep 8554 || echo "8554 not listening"
echo "== probe"
timeout 30 ffprobe -v error -rtsp_transport tcp -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate -of csv=p=0 \
  -i rtsp://127.0.0.1:8554/seatnow
echo "probe_exit=$?"
