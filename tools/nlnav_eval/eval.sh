#!/usr/bin/env bash
# NL-navigation eval on the live Godot stack (tools/godot/godot_sim_up.sh must be up).
#   bash tools/nlnav_eval/eval.sh [--tag NAME] [--repeat N] [--model qwen2.5:3b-instruct] [--drive] [--only a,b]
#                                 [--action /m20/vlmnav/navigate --cases tools/realtime_b/cases.yaml]
#                                 [--action /m20/vlfm/navigate --cases tools/realtime_b/cases_search.yaml]
# --model restarts the NL container on that Ollama model (tools/godot/nlnav_up.sh) and waits for its warm-up.
# --drive runs the drive-level cases (moves the robot; ~1 min per case) instead of the plan-level ones.
# --action / --cases: Real-time B (tools/realtime_b/realtime_b_up.sh up): the same harness drives the VLM episodes.
# Output: ~/m20_sim/evals/<tag>_<timestamp>.json (+ the table on stdout). Compare runs: compare.py a.json b.json
set -eo pipefail               # a failure inside a pipe (docker exec | ...) must stop the run, not pass silently
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
C=docker-commander-1
SRC='source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp'
TAG=run; REPEAT=1; MODEL=""; DRIVE=""; ONLY=""; SITE=circuit4; ACTION=/m20/nlnav/navigate; CASES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift ;;
    --repeat) REPEAT="$2"; shift ;;
    --model) MODEL="$2"; shift ;;
    --drive) DRIVE="--drive" ;;
    --only) ONLY="$2"; shift ;;
    --site) SITE="$2"; shift ;;
    --action) ACTION="$2"; shift ;;
    --cases) CASES="$2"; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
OBJECT_MAP="$HOME/m20_sim/semmap/$SITE/object_map.json"
CASES="${CASES:-$REPO/tools/nlnav_eval/cases.yaml}"
[ -f "$CASES" ] || { echo "ERROR: cases file $CASES not found"; exit 2; }
OUT_DIR="$HOME/m20_sim/evals"; mkdir -p "$OUT_DIR"
TS=$(date +%Y%m%d_%H%M%S); OUT="$OUT_DIR/${TAG}_${TS}.json"

# localization: odometry from the bridge's GODOT_ODOM; corrected by AMCL if it runs (gt = truth odom, no AMCL;
# odom = drifting odom, no correction; amcl = drifting odom + AMCL)
ODOM=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' m20_godot_bridge 2>/dev/null | sed -n 's/^GODOT_ODOM=//p')
ODOM=${ODOM:-truth}
if docker exec $C pgrep -f nav2_amcl >/dev/null 2>&1; then LOCMODE=amcl
elif docker exec $C pgrep -f map_tf_adapter >/dev/null 2>&1; then LOCMODE=vendor     # the robot's own pose report -> map->odom
elif [ "$ODOM" = drift ]; then LOCMODE=odom; else LOCMODE=gt; fi

if [ -n "$MODEL" ]; then
  echo "[eval] switching the NL node to $MODEL"
  docker exec m20_ollama ollama list | grep -q "^${MODEL}" || docker exec m20_ollama ollama pull "$MODEL"
  # only the model changes: carry the running node's LLM backend and label threshold over from its command line
  # (they may have been set at launch, not in this shell), and keep "here" on the stack's pose source
  OLD_CMD=$(docker inspect -f '{{join .Config.Cmd " "}}' m20_nlnav 2>/dev/null || true)
  OLD_LLM=$(sed -n 's/.*-p llm:=\([^ ]*\).*/\1/p' <<<"$OLD_CMD")
  OLD_MIN=$(sed -n 's/.*-p min_label_score:=\([^ ]*\).*/\1/p' <<<"$OLD_CMD")
  bash "$REPO/tools/godot/nlnav_up.sh" --object-map "$OBJECT_MAP" --scene "godot_$SITE" --model "$MODEL" \
    ${OLD_LLM:+--llm "$OLD_LLM"} ${OLD_MIN:+--min-label-score "$OLD_MIN"} \
    --pose-source "$([ "$LOCMODE" = gt ] && echo odom_true || echo tf)" --wait
fi
# configuration block: what this run was measured with
# the model under test, chosen by the action: the offline NL node (m20_nlnav) or Real-time B's VLM server (m20_vlm).
# It must be RUNNING (docker inspect also answers for a stopped leftover container, which would mislabel the run).
# The search needs its matcher, its detector and its node: a missing one would read as a real not_found.
case "$ACTION" in */vlmnav/*) MODEL_CTR=m20_vlm; NEED="m20_vlm" ;; */vlfm/*) MODEL_CTR=m20_itm; NEED="m20_itm m20_detector m20_vlfm" ;;
                  *) MODEL_CTR=m20_nlnav; NEED="m20_nlnav" ;; esac
# waypoint walks (cases_waypoints.yaml, `kind: waypoints`): mission server only, no model container, always drive-level
WAYPOINTS=0
if grep -q '^kind: waypoints' "$CASES"; then WAYPOINTS=1; NEED=""; MODEL_CTR=""; ACTION=/m20/mission/run; fi
for ctr in $NEED; do
  [ "$(docker inspect -f '{{.State.Running}}' "$ctr" 2>/dev/null)" = "true" ] \
    || { echo "ERROR: $ctr is not running (needed for --action $ACTION)"; exit 1; }
done
CFG=$(for ctr in $NEED; do docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$ctr"; done | python3 -c "
import sys, json, os, subprocess
env = dict(l.split('=', 1) for l in sys.stdin.read().split() if '=' in l)
vlm = env.get('COSMOS_MODEL'); itm = env.get('ITM_MODEL')     # Real-time B: the VLM server, or the search's matcher
cfg = {'llm': env.get('NLNAV_LLM') or ('vlm' if vlm else 'vlfm_search' if itm else None),
       'model': env.get('NLNAV_LLM_MODEL') or vlm or itm,
       'encoder': 'owl' if not vlm else None,
       'det_pose_topic': env.get('DET_POSE_TOPIC'), 'explore_map': env.get('M20_EXPLORE_MAP'),
       'obstacle_scale': env.get('M20_OBSTACLE_SCALE'), 'min_vel_x': env.get('M20_MIN_VEL_X'), 'cam_min_z': env.get('M20_CAM_MIN_Z'),
       'localization': '$LOCMODE', 'odom': '$ODOM', 'action': '$ACTION',
       'object_map_bytes': os.path.getsize('$OBJECT_MAP') if os.path.exists('$OBJECT_MAP') else 0,
       'git': subprocess.run(['git', '-C', '$REPO', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()}
print(json.dumps(cfg))")
echo "[eval] config $CFG"
[ "$WAYPOINTS" = "1" ] || docker logs m20_nlnav 2>&1 | grep -E "objects kept|min_label" | tail -1 | sed -E 's/^\[INFO\] \[[0-9.]+\] \[m20_nlnav\]: /[eval] /' || true   # informational

# the runner + judge + cases + object map go into the commander (no repo mount there)
docker exec $C bash -lc "rm -rf /tmp/nlnav_eval && mkdir -p /tmp/nlnav_eval"
docker cp "$REPO/tools/nlnav_eval/run_eval.py" $C:/tmp/nlnav_eval/
docker cp "$REPO/tools/nlnav_eval/judge.py" $C:/tmp/nlnav_eval/
docker cp "$REPO/tools/missions/route.py" $C:/tmp/nlnav_eval/
[ "$WAYPOINTS" = "1" ] && docker cp "$REPO/tools/missions" $C:/tmp/nlnav_eval/missions
docker cp "$CASES" $C:/tmp/nlnav_eval/cases.yaml
[ -f "$OBJECT_MAP" ] && docker cp "$OBJECT_MAP" $C:/tmp/nlnav_eval/object_map.json
T0=$(date +%s)
docker exec -i $C bash -lc "$SRC; python3 /tmp/nlnav_eval/run_eval.py --cases /tmp/nlnav_eval/cases.yaml --out /tmp/nlnav_eval/run.json \
  --repeat $REPEAT --tag '$TAG' --config '$CFG' --object-map /tmp/nlnav_eval/object_map.json --action '$ACTION' $DRIVE ${ONLY:+--only $ONLY}"
docker cp $C:/tmp/nlnav_eval/run.json "$OUT"
if [ "$WAYPOINTS" = "1" ]; then : > "$OUT_DIR/${TAG}_${TS}.timing.log"; else
docker logs --since "$T0" m20_nlnav 2>&1 | grep "timing:" > "$OUT_DIR/${TAG}_${TS}.timing.log" || true; fi
python3 "$REPO/tools/nlnav_eval/finish_run.py" "$OUT" "$OUT_DIR/${TAG}_${TS}.timing.log"
echo "[eval] saved $OUT"
