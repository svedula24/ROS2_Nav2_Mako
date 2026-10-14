#!/usr/bin/env bash
# Offline phase, step 2: object map from a recorded run (GPU, m20_splat image, offline).
#   bash tools/semmap/build_object_map.sh ~/m20_sim/semmap/circuit4/run_<ts> [extra args]
#   -> ~/m20_sim/semmap/circuit4/object_map.json + crops/
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="$(cd "$1" && pwd)"; shift
SITE_DIR="$(dirname "$RUN")"
docker run --rm --gpus all -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  -v "$REPO":/repo:ro -v "$SITE_DIR":/site \
  m20_splat:latest python3 /repo/tools/semmap/build_object_map.py "/site/$(basename "$RUN")" \
  --out /site/object_map.json "$@"
