#!/usr/bin/env bash
# (Re)start the image-text matching server (m20_itm: BLIP-2 ITC, tools/realtime_b/itm_server.py) for the VLFM-style
# search.   bash tools/realtime_b/itm_up.sh [--wait]      --wait blocks until the warm-up score is done (up to 3 min)
# Weights from ~/m20_sim/models (HF cache; download once). ~2.5 GB of GPU memory.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WAIT=0
case "${1:-}" in --wait) WAIT=1 ;; "") ;; *) echo "usage: itm_up.sh [--wait]"; exit 2 ;; esac
MODELS="$HOME/m20_sim/models"
[ -d "$MODELS/hub/models--Salesforce--blip2-itm-vit-g" ] || { echo "ERROR: blip2-itm-vit-g not in $MODELS (download it first)"; exit 1; }
docker rm -f m20_itm >/dev/null 2>&1 || true
docker run -d --name m20_itm --gpus all --network host --ipc host \
  -e HF_HOME=/models -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  -e ITM_MODEL="${ITM_MODEL:-Salesforce/blip2-itm-vit-g}" -e ITM_PORT="${ITM_PORT:-8091}" \
  -v "$MODELS":/models -v "$REPO/tools/realtime_b":/rtb:ro \
  m20_splat:latest python3 /rtb/itm_server.py >/dev/null
if [ "$WAIT" = "1" ]; then
  for i in $(seq 1 90); do
    docker logs m20_itm 2>&1 | grep -q "warm-up done" && { docker logs m20_itm 2>&1 | grep -E "loaded in|warm-up done" | cut -c1-160; exit 0; }
    docker ps --format '{{.Names}}' | grep -qx m20_itm || { echo "ERROR: m20_itm exited:"; docker logs m20_itm 2>&1 | tail -20; exit 1; }
    sleep 2
  done
  echo "ERROR: ITM warm-up not done after 3 min"; docker logs m20_itm 2>&1 | tail -10; exit 1
fi
