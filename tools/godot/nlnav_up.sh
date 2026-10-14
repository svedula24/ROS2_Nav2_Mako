#!/usr/bin/env bash
# (Re)start the NL navigation container (m20_nlnav) on the Godot stack. Used by godot_sim_up.sh at bringup and by
# tools/nlnav_eval/eval.sh to swap the LLM between runs.
#   bash tools/godot/nlnav_up.sh --object-map ~/m20_sim/semmap/circuit4/object_map.json --scene godot_circuit4
#        [--llm ollama|claude|json] [--model qwen2.5:7b-instruct] [--min-label-score 0.4] [--wait]
# --wait blocks until the node logs "warm-up done" (text encoder + one real LLM parse), up to 3 min.
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LLM="${NLNAV_LLM:-ollama}"; MODEL="${NLNAV_LLM_MODEL:-qwen2.5:7b-instruct}"; MIN_LABEL="${MIN_LABEL_SCORE:-0.4}"
OBJECT_MAP=""; SCENE=""; WAIT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --object-map) OBJECT_MAP="$2"; shift ;;
    --scene) SCENE="$2"; shift ;;
    --llm) LLM="$2"; shift ;;
    --model) MODEL="$2"; shift ;;
    --min-label-score) MIN_LABEL="$2"; shift ;;
    --wait) WAIT=1 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
[ -n "$OBJECT_MAP" ] && [ -n "$SCENE" ] || { echo "usage: nlnav_up.sh --object-map FILE --scene NAME [...]"; exit 2; }
mkdir -p "$(dirname "$OBJECT_MAP")"
NL_ENV=(-e ROS_DOMAIN_ID=42 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e PYTHONUNBUFFERED=1
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NLNAV_LLM="$LLM" -e NLNAV_LLM_MODEL="$MODEL")
[ -n "${ANTHROPIC_API_KEY:-}" ] && NL_ENV+=(-e ANTHROPIC_API_KEY)
NL_CMD="source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export PYTHONPATH=/nlnav:\$PYTHONPATH;"
NL_CMD+=" python3 -c 'from m20_nlnav.nlnav_node import main; main()' --ros-args"
NL_CMD+=" -p scene_yaml:=/scene/${SCENE%.xml}.scene.yaml -p object_map:=/semmap/$(basename "$OBJECT_MAP")"
NL_CMD+=" -p llm:=$LLM -p text_encoder:=owl -p min_label_score:=$MIN_LABEL"
docker rm -f m20_nlnav >/dev/null 2>&1 || true
docker run -d --name m20_nlnav --network host --ipc host "${NL_ENV[@]}" \
  -v "$REPO/src/m20_nlnav":/nlnav:ro -v "$REPO/tools/sim":/scene:ro -v "$(dirname "$OBJECT_MAP")":/semmap:ro \
  m20_splat:latest bash -lc "$NL_CMD" >/dev/null
echo "[nlnav_up] m20_nlnav started: llm=$LLM model=$MODEL min_label_score=$MIN_LABEL"
if [ "$WAIT" = "1" ]; then
  for i in $(seq 1 90); do docker logs m20_nlnav 2>&1 | grep -q "warm-up done\|warm-up failed\|Traceback" && break; sleep 2; done
  docker logs m20_nlnav 2>&1 | grep -E "objects kept|warm-up|Traceback" | sed -E 's/^\[INFO\] \[[0-9.]+\] \[m20_nlnav\]: /[nlnav_up] /' | tail -3 || true
  # an unready node must stop the caller (eval.sh), not be measured: only "warm-up done" counts as ready
  docker logs m20_nlnav 2>&1 | grep -q "warm-up done" ||
    { echo "[nlnav_up] ERROR: m20_nlnav not ready (warm-up failed, crashed, or > 180 s): docker logs m20_nlnav"; exit 1; }
fi
