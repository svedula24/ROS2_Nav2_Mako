#!/usr/bin/env bash
# Offline phase, step 2: object map from recorded runs (GPU, m20_splat image, offline).
#   bash tools/semmap/build_object_map.sh ~/m20_sim/semmap/circuit4/run_<ts> [run_<ts2> ...] [-- extra args]
#   bash tools/semmap/build_object_map.sh --all circuit4      every run of the site with keyframes
#   -> ~/m20_sim/semmap/<site>/object_map.json + crops/ (the previous map is kept as object_map.prev.json)
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RUNS=()
if [ "$1" = "--all" ]; then
  SITE_DIR="$HOME/m20_sim/semmap/$2"; shift 2
  for d in "$SITE_DIR"/run_*/; do
    [ "$(wc -l < "$d/index.jsonl" 2>/dev/null || echo 0)" -gt 20 ] && RUNS+=("$(cd "$d" && pwd)")
  done
else
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do RUNS+=("$(cd "$1" && pwd)"); shift; done
  SITE_DIR="$(dirname "${RUNS[0]}")"
fi
[ "${1:-}" = "--" ] && shift
[ ${#RUNS[@]} -gt 0 ] || { echo "ERROR: no runs"; exit 1; }
for r in "${RUNS[@]}"; do                                 # one site per build: the container sees only $SITE_DIR
  [ "$(dirname "$r")" = "$SITE_DIR" ] || { echo "ERROR: $r is not in $SITE_DIR (build one site at a time)"; exit 1; }
done
echo "[build] runs: ${RUNS[*]##*/}"
[ -f "$SITE_DIR/object_map.json" ] && cp "$SITE_DIR/object_map.json" "$SITE_DIR/object_map.prev.json"
docker run --rm --gpus all -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  -v "$REPO":/repo:ro -v "$SITE_DIR":/site \
  m20_splat:latest python3 /repo/tools/semmap/build_object_map.py "${RUNS[@]/#$SITE_DIR//site}" \
  --out /site/object_map.json "$@"
