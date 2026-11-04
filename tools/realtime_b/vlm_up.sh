#!/usr/bin/env bash
# (Re)start the Real-time B VLM server (m20_vlm: Cosmos-Reason1-7B, tools/realtime_b/cosmos_server.py) on the GPU.
#   bash tools/realtime_b/vlm_up.sh [--wait]        --wait blocks until the warm-up decision is done (up to 5 min)
# The model is read from ~/m20_sim/models (HF cache layout; download once, see tools/realtime_b/README notes in
# cosmos_server.py). Offline at runtime (HF_HUB_OFFLINE=1). Needs ~16 GB of GPU memory: don't run Ollama/OWLv2 with it.
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WAIT=0; [ "${1:-}" = "--wait" ] && WAIT=1
MODELS="$HOME/m20_sim/models"
[ -d "$MODELS/hub/models--nvidia--Cosmos-Reason1-7B" ] || { echo "ERROR: Cosmos-Reason1-7B not in $MODELS (download it first)"; exit 1; }
docker rm -f m20_vlm >/dev/null 2>&1 || true
docker run -d --name m20_vlm --gpus all --network host --ipc host \
  -e HF_HOME=/models -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  -e COSMOS_MODEL="${COSMOS_MODEL:-nvidia/Cosmos-Reason1-7B}" \
  -e COSMOS_PORT="${COSMOS_PORT:-8090}" -e COSMOS_MAX_SIDE="${COSMOS_MAX_SIDE:-448}" \
  -v "$MODELS":/models -v "$REPO/tools/realtime_b":/rtb:ro \
  m20_splat:latest python3 /rtb/cosmos_server.py >/dev/null
if [ "$WAIT" = "1" ]; then
  for i in $(seq 1 150); do
    docker logs m20_vlm 2>&1 | grep -q "warm-up done" && { docker logs m20_vlm 2>&1 | grep -E "loaded in|warm-up done" | cut -c1-220; exit 0; }
    docker ps --format '{{.Names}}' | grep -qx m20_vlm || { echo "ERROR: m20_vlm exited:"; docker logs m20_vlm 2>&1 | tail -20; exit 1; }
    sleep 2
  done
  echo "ERROR: VLM warm-up not done after 5 min"; docker logs m20_vlm 2>&1 | tail -10; exit 1
fi
