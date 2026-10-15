#!/usr/bin/env bash
# NL-navigation eval on the live Godot stack (tools/godot/godot_sim_up.sh must be up).
#   bash tools/nlnav_eval/eval.sh [--tag NAME] [--repeat N] [--model qwen2.5:3b-instruct] [--drive] [--only a,b]
# --model restarts the NL container on that Ollama model (tools/godot/nlnav_up.sh) and waits for its warm-up.
# --drive runs the drive-level cases (moves the robot; ~1 min per case) instead of the plan-level ones.
# Output: ~/m20_sim/evals/<tag>_<timestamp>.json (+ the table on stdout). Compare runs: compare.py a.json b.json
set -eo pipefail               # a failure inside a pipe (docker exec | ...) must stop the run, not pass silently
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
C=docker-commander-1
SRC='source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp'
TAG=run; REPEAT=1; MODEL=""; DRIVE=""; ONLY=""; SITE=circuit4
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift ;;
    --repeat) REPEAT="$2"; shift ;;
    --model) MODEL="$2"; shift ;;
    --drive) DRIVE="--drive" ;;
    --only) ONLY="$2"; shift ;;
    --site) SITE="$2"; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
OBJECT_MAP="$HOME/m20_sim/semmap/$SITE/object_map.json"
OUT_DIR="$HOME/m20_sim/evals"; mkdir -p "$OUT_DIR"
TS=$(date +%Y%m%d_%H%M%S); OUT="$OUT_DIR/${TAG}_${TS}.json"

if [ -n "$MODEL" ]; then
  echo "[eval] switching the NL node to $MODEL"
  docker exec m20_ollama ollama list | grep -q "^${MODEL}" || docker exec m20_ollama ollama pull "$MODEL"
  bash "$REPO/tools/godot/nlnav_up.sh" --object-map "$OBJECT_MAP" --scene "godot_$SITE" --model "$MODEL" --wait
fi
# configuration block: what this run was measured with
# localization: odometry from the bridge's GODOT_ODOM; corrected by AMCL if it runs (gt = truth odom, no AMCL;
# odom = drifting odom, no correction; amcl = drifting odom + AMCL)
ODOM=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' m20_godot_bridge 2>/dev/null | sed -n 's/^GODOT_ODOM=//p')
ODOM=${ODOM:-truth}
if docker exec $C pgrep -f nav2_amcl >/dev/null 2>&1; then LOCMODE=amcl; elif [ "$ODOM" = drift ]; then LOCMODE=odom; else LOCMODE=gt; fi
CFG=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' m20_nlnav | python3 -c "
import sys, json, os, subprocess
env = dict(l.split('=', 1) for l in sys.stdin.read().split() if '=' in l)
cfg = {'llm': env.get('NLNAV_LLM'), 'model': env.get('NLNAV_LLM_MODEL'), 'encoder': 'owl',
       'localization': '$LOCMODE', 'odom': '$ODOM',
       'object_map_bytes': os.path.getsize('$OBJECT_MAP') if os.path.exists('$OBJECT_MAP') else 0,
       'git': subprocess.run(['git', '-C', '$REPO', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()}
print(json.dumps(cfg))")
echo "[eval] config $CFG"
docker logs m20_nlnav 2>&1 | grep -E "objects kept|min_label" | tail -1 | sed -E 's/^\[INFO\] \[[0-9.]+\] \[m20_nlnav\]: /[eval] /' || true   # informational

# the runner + judge + cases + object map go into the commander (no repo mount there)
docker exec $C bash -lc "rm -rf /tmp/nlnav_eval && mkdir -p /tmp/nlnav_eval"
docker cp "$REPO/tools/nlnav_eval/run_eval.py" $C:/tmp/nlnav_eval/
docker cp "$REPO/tools/nlnav_eval/judge.py" $C:/tmp/nlnav_eval/
docker cp "$REPO/tools/nlnav_eval/cases.yaml" $C:/tmp/nlnav_eval/
[ -f "$OBJECT_MAP" ] && docker cp "$OBJECT_MAP" $C:/tmp/nlnav_eval/object_map.json
T0=$(date +%s)
docker exec -i $C bash -lc "$SRC; python3 /tmp/nlnav_eval/run_eval.py --cases /tmp/nlnav_eval/cases.yaml --out /tmp/nlnav_eval/run.json \
  --repeat $REPEAT --tag '$TAG' --config '$CFG' --object-map /tmp/nlnav_eval/object_map.json $DRIVE ${ONLY:+--only $ONLY}"
docker cp $C:/tmp/nlnav_eval/run.json "$OUT"
docker logs --since "$T0" m20_nlnav 2>&1 | grep "timing:" > "$OUT_DIR/${TAG}_${TS}.timing.log" || true
python3 "$REPO/tools/nlnav_eval/finish_run.py" "$OUT" "$OUT_DIR/${TAG}_${TS}.timing.log"
echo "[eval] saved $OUT"
