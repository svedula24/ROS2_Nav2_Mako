#!/usr/bin/env bash
# Stop Real-time B and free the machine: the VLM + episode containers, then everything godot_sim_down.sh stops
# (ROS stack, Ollama if up, a clean Godot quit). --all is passed through (also `docker compose down`).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
pkill -f "bash .*tools/realtime_b/[r]ealtime_b_up.sh" 2>/dev/null || true
sleep 1                                        # let its EXIT trap finish before we remove the same containers
echo "[realtime_b_down] stopping the VLM and the episode node"
docker rm -f m20_vlm m20_vlmnav m20_itm m20_vlfm m20_detector >/dev/null 2>&1 || true
bash "$REPO/tools/godot/godot_sim_down.sh" "$@"
