#!/usr/bin/env bash
# Offline phase, step 1: drive a coverage tour of the site and record keyframes for the object map.
# Run with the stack up (tools/godot/godot_sim_up.sh in another terminal; robot armed ASSISTED).
#
#   bash tools/semmap/record_mapping_run.sh [--site circuit4] [--spacing 3.0] [--resume]
#   -> ~/m20_sim/semmap/<site>/run_<timestamp>/ (keyframes), then build the map with
#      bash tools/semmap/build_object_map.sh ~/m20_sim/semmap/<site>/run_<timestamp> [more runs...]
# --resume: drive only the tour stops that no earlier run of this site passed within 2 m of (nearest first
#   from the robot), e.g. after a run that ended early; then build from all the runs together.
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SITE=circuit4; SPACING=3.0; RESUME=0
while [ $# -gt 0 ]; do
  case "$1" in
    --site) SITE="$2"; shift ;;
    --spacing) SPACING="$2"; shift ;;
    --resume) RESUME=1 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
MAP_YAML="$REPO/tools/sim/godot_${SITE}_map.yaml"
[ -f "$MAP_YAML" ] || { echo "ERROR: $MAP_YAML missing (tools/godot/site_map.py)"; exit 1; }
SITE_DIR="$HOME/m20_sim/semmap/$SITE"
SKIP=()
if [ "$RESUME" = "1" ]; then
  for d in "$SITE_DIR"/run_*/; do                        # earlier runs with real data (not aborted stubs)
    [ "$(wc -l < "$d/index.jsonl" 2>/dev/null || echo 0)" -gt 20 ] && SKIP+=("/site/$(basename "$d")")
  done
  [ ${#SKIP[@]} -gt 0 ] || { echo "ERROR: --resume but no earlier run with keyframes in $SITE_DIR"; exit 1; }
fi
RUN="$SITE_DIR/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"

pose=$(python3 - <<'PY'
import json, socket
s = socket.create_connection(("127.0.0.1", 7799), timeout=5); s.sendall(b'{"cmd":"telemetry"}\n'); b = b""
while not b.endswith(b"\n"): b += s.recv(1 << 20)
t = json.loads(b)["base_enu"]["t"]; print(f"{t[0]:.2f} {t[1]:.2f}")
PY
)
echo "[mapping] robot at $pose; planning a coverage tour (spacing $SPACING m)"
mapfile -t STOPS < <(docker run --rm -v "$REPO":/repo:ro -v "$SITE_DIR":/site:ro m20_splat:latest \
  python3 /repo/tools/semmap/coverage_tour.py /repo/tools/sim/$(basename "$MAP_YAML") --start $pose --spacing "$SPACING" \
  ${SKIP[@]:+--skip-seen "${SKIP[@]}"})
echo "[mapping] ${#STOPS[@]} stops"
[ ${#STOPS[@]} -gt 0 ] || { echo "[mapping] nothing left to cover"; rmdir "$RUN"; exit 0; }

docker rm -f m20_semmap_rec >/dev/null 2>&1 || true
docker run -d --name m20_semmap_rec --network host --ipc host \
  -e ROS_DOMAIN_ID=42 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e PYTHONUNBUFFERED=1 -e SEMMAP_OUT=/out \
  -v "$REPO/tools/semmap":/semmap:ro -v "$RUN":/out \
  m20_splat:latest bash -lc "source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; python3 /semmap/keyframe_recorder.py" >/dev/null
trap 'docker rm -f m20_semmap_rec >/dev/null 2>&1 || true' EXIT
sleep 4

echo "[mapping] driving the tour (one mission; Ctrl-C to abort)"
"$REPO/tools/dev/m20" mission "${STOPS[@]}" || echo "[mapping] mission ended early: keeping what was recorded"
sleep 2
N=$(wc -l < "$RUN/index.jsonl" 2>/dev/null || echo 0)
echo "[mapping] $N keyframes in $RUN"
echo "next: bash tools/semmap/build_object_map.sh $RUN"
