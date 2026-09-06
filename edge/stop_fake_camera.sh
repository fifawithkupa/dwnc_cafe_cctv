#!/usr/bin/env bash
# Stop ONLY the fake camera (mediamtx + its publisher), leaving any live run alone.
# Usage: ~/seatnow/edge/stop_fake_camera.sh
#
# Why a script: `pkill -f <pattern>` from an ssh one-liner also matches the ssh
# command line itself, and a pattern containing the stream URL matches the
# judging process too — both mistakes killed the wrong thing (2026-09-06).
set -u
pkill -x mediamtx 2>/dev/null
pkill -f "edge.rtsp_republish" 2>/dev/null
pkill -f "stream_loop -1 -i" 2>/dev/null
sleep 1
left=$(pgrep -c -x mediamtx 2>/dev/null || echo 0)
echo "mediamtx left: $left"
pgrep -af "engine.seatnow rtsp" | grep -v pgrep | cut -c1-70 || echo "no live run"
