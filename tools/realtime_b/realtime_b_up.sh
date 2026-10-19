#!/usr/bin/env bash
# Real-time B on the Godot sim (circuit4): a VLM (Cosmos-Reason1-7B, zero-shot) picks the next short move from the
# camera; Nav2 behaviors (DriveOnHeading / Spin; no backward move) execute it. NO maps: no site map, no object map, drifting
# odometry with map->odom fixed at identity. STAYS IN THE FOREGROUND: Ctrl-C tears the ROS stack down.
# The offline phase's launcher (tools/godot/godot_sim_up.sh) is separate and untouched; run one or the other.
#   bash tools/realtime_b/realtime_b_up.sh [--mode vlm|search|replay] [--stop-sim]
#     vlm     (default) Cosmos picks moves:   tools/dev/m20 vlm "go to the ladder" | --dry-run | stop
#     search  VLFM-style: value map + frontiers + live OWLv2 detector (no Cosmos):  tools/dev/m20 search "refrigerator"
#     replay  no VLM, no node: replay logged episodes through the behaviors (recordings)
# Logs per episode: ~/m20_sim/vlmnav/<stamp>/ (vlm) or ~/m20_sim/vlfm/<stamp>/ (search).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
GODOT_EXE="${GODOT_EXE:-/mnt/c/Users/$(cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r')/tools/godot-4.7.1/Godot_v4.7.1-stable_win64.exe}"
GODOT_SIM="${GODOT_SIM:-$(dirname "$REPO")/stateful-drone-sim}"
SITE="${GODOT_SITE:-circuit4_cam_mesh}"
SCENE="godot_circuit4.xml"
STOP_SIM=0; MODE="${RTB_MODE:-vlm}"
[ "${RTB_NO_VLM:-0}" = "1" ] && MODE=replay
while [ $# -gt 0 ]; do
  case "$1" in
    --stop-sim) STOP_SIM=1 ;;
    --mode) [ $# -ge 2 ] || { echo "--mode needs a value: vlm | search | replay"; exit 2; }; MODE="$2"; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
case "$MODE" in vlm|search|replay) ;; *) echo "unknown --mode $MODE (vlm | search | replay)"; exit 2 ;; esac
SRC="source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"

api_up() { python3 -c 'import socket,sys
try: socket.create_connection(("127.0.0.1", 7799), timeout=1).close()
except OSError: sys.exit(1)'; }
cleanup() {
  local rc=$?                                   # a failed bringup must exit non-zero, not be masked by the teardown
  trap - INT TERM EXIT
  echo; echo "[realtime_b] tearing down (exit $rc)..."
  [ -n "${WATCHDOG_PID:-}" ] && { kill "$WATCHDOG_PID" 2>/dev/null || true; }
  bash "$REPO/tools/nav/kill_stack.sh" >/dev/null 2>&1 || true
  docker rm -f m20_vlm m20_vlmnav m20_itm m20_vlfm m20_detector >/dev/null 2>&1 || true
  [ "$STOP_SIM" = "1" ] && { python3 -c 'import socket; socket.create_connection(("127.0.0.1",7799),timeout=2).sendall(b"{\"cmd\":\"quit\"}\n")' 2>/dev/null || true; }
  echo "[realtime_b] down."
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
bash "$REPO/tools/nav/kill_stack.sh" >/dev/null 2>&1 || true
docker rm -f m20_ollama >/dev/null 2>&1 || true      # free the GPU for the VLM (the offline launcher restarts it)

# 1. the Godot sim (reused if already up), window parked off-screen (rendering continues)
if api_up; then
  echo "[realtime_b] reusing the Godot sim already on 127.0.0.1:7799"
else
  [ -f "$GODOT_EXE" ] || { echo "ERROR: Godot not found at $GODOT_EXE (set GODOT_EXE)"; exit 1; }
  echo "[realtime_b] starting the Godot sim ($SITE, M20)..."
  (cd /mnt/c && cmd.exe /c start "Godot sim" "$(wslpath -w "$GODOT_EXE")" --path "$(wslpath -w "$GODOT_SIM/godot_sim")" \
    -- --api --scene=splat_scene --site="$SITE" --vehicle=m20 --videofps=10 </dev/null >/dev/null 2>&1 &)
  for i in $(seq 1 90); do api_up && break; sleep 2; done
  api_up || { echo "ERROR: the sim's API never came up"; exit 1; }
  sleep 5
fi
[ "${GODOT_SHOW:-0}" = "1" ] || powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w "$REPO/tools/godot/godot_window.ps1")" park 2>&1 | tr -d '\r' || true

# 2. the bridge: drifting odometry (no map to correct it, as on an unseen site); velocity from collision_monitor
echo "[realtime_b] starting the Godot ROS bridge (odometry: ${RTB_ODOM:-drift}, velocity in: /cmd_vel_safe)"
docker rm -f m20_godot_bridge >/dev/null 2>&1 || true
docker run -d --name m20_godot_bridge --network host --ipc host \
  -e ROS_DOMAIN_ID=42 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e PYTHONUNBUFFERED=1 \
  -e GODOT_VMAX="${GODOT_VMAX:-0.5}" -e GODOT_DEPTH_OBSTACLES=1 -e GODOT_ODOM="${RTB_ODOM:-drift}" -e GODOT_CMD_TOPIC=/cmd_vel_safe \
  -v "$REPO/tools/godot":/godot:ro \
  m20_sim:latest bash -lc "source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; python3 /godot/godot_bridge.py" >/dev/null
sleep 6
python3 -c 'import socket,time; s=socket.create_connection(("127.0.0.1",7799),timeout=10); s.sendall(b"{\"cmd\":\"reset\"}\n"); s.recv(4096); time.sleep(3)' \
  || echo "[realtime_b] WARNING: could not reset the sim to the spawn"

# 3. the stack without maps: identity map->odom (M20_GT_LOC=1 with drifting odom), rolling costmaps, + Real-time B hooks.
# The exploration map (fixed-extent, unknown-tracking global costmap) is the search's: the vlm and replay modes keep
# the stock rolling costmap unless RTB_EXPLORE_MAP=1 asks for it.
# Its extent is the SCANNED site (circuit4's site map: x -8.0..9.5, y -11.3..16.1, + margin): the Godot collision
# mesh lets the robot drive on past the scan into the splat's blurry outskirts, where there is nothing to find and
# the floor is not real (one search went 4 m past the west edge and tipped over, 2026-10-19). A real site ends in
# walls; this bound stands in for them. Only frontiers inside it exist.
EXPLORE=0; [ "$MODE" = search ] && EXPLORE=1
EXPLORE="${RTB_EXPLORE_MAP:-$EXPLORE}"
M20_EXTERNAL_SIM=1 M20_SCENE="$SCENE" M20_GT_LOC=1 M20_LOC=odom M20_NO_GOAL=1 M20_SPLAT_CAMERA=0 M20_NLNAV=0 \
M20_SITE_MAP="" M20_MAX_VEL="${GODOT_VMAX:-0.5}" M20_BEHAVIORS_EXTRA=drive_on_heading M20_COLLISION_MONITOR=1 \
M20_EXPLORE_MAP="$EXPLORE" M20_EXPLORE_ORIGIN="${RTB_EXPLORE_ORIGIN:--8.5 -11.8}" M20_EXPLORE_SIZE="${RTB_EXPLORE_SIZE:-19 29}" \
bash "$REPO/tools/nav/bringup_plan_a.sh"
arm() { docker exec docker-commander-1 bash -lc "$SRC; timeout 10 ros2 service call /m20/set_mode m20_msgs/srv/SetMode '{mode: $1, requester: realtime_b_up}'" | grep -c "accepted=True" || true; }
if [ "$(arm 2)" = "0" ]; then arm 0 >/dev/null; [ "$(arm 2)" = "1" ] || { echo "ERROR: the commander refused ASSISTED (tools/dev/m20 status)"; exit 1; }; fi
echo "[realtime_b] armed ASSISTED"

# 4. the VLM server, then the episode node (RTB_NO_VLM=1: skip both, e.g. to replay logged episodes through the
#    behaviors for a recording with the GPU free)
if [ "$MODE" = "replay" ]; then
  echo "[realtime_b] replay mode: no VLM, no episode node"
elif [ "$MODE" = "search" ]; then
  # VLFM-style search: the image-text matcher, the live OWLv2 detector on the BELIEVED pose, the search node
  echo "[realtime_b] starting the image-text matcher (BLIP-2 ITC)"
  bash "$REPO/tools/realtime_b/itm_up.sh" --wait
  echo "[realtime_b] starting the object detector (OWLv2, pose from /odom)"
  docker rm -f m20_detector >/dev/null 2>&1 || true
  docker run -d --name m20_detector --gpus all --network host --ipc host \
    -e ROS_DOMAIN_ID=42 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e PYTHONUNBUFFERED=1 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e DET_POSE_TOPIC=/odom \
    -v "$REPO/tools/splat":/splat_tools:ro \
    m20_splat:latest python3 -u /splat_tools/detector_node.py >/dev/null
  mkdir -p "$HOME/m20_sim/vlfm"
  docker rm -f m20_vlfm >/dev/null 2>&1 || true
  docker run -d --name m20_vlfm --network host --ipc host \
    -e ROS_DOMAIN_ID=42 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e PYTHONUNBUFFERED=1 -e M20_EXPLORE_MAP="$EXPLORE" \
    -v "$REPO/src/m20_vlmnav":/vlmnav:ro -v "$REPO/src/m20_nlnav":/nlnav:ro -v "$HOME/m20_sim/vlfm":/logs \
    m20_splat:latest bash -lc "source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export PYTHONPATH=/vlmnav:/nlnav:\$PYTHONPATH; \
python3 -c 'from m20_vlmnav.search_node import main; main()' --ros-args -p log_dir:=/logs" >/dev/null
  NODE_OK=0
  for i in $(seq 1 30); do docker logs m20_vlfm 2>&1 | grep -q "search up" && { NODE_OK=1; break; }; sleep 1; done
  [ "$NODE_OK" = "1" ] || { echo "ERROR: the search node did not come up:"; docker logs m20_vlfm 2>&1 | tail -15; exit 1; }
  DET_OK=0                                   # without the detector every search ends not_found: fail loudly instead
  for i in $(seq 1 60); do
    docker logs m20_detector 2>&1 | grep -q "object detector up" && { DET_OK=1; break; }
    docker ps --format '{{.Names}}' | grep -qx m20_detector || break
    sleep 2
  done
  [ "$DET_OK" = "1" ] || { echo "ERROR: the object detector did not come up:"; docker logs m20_detector 2>&1 | tail -15; exit 1; }
  docker logs m20_detector 2>&1 | grep "object detector up" | tail -1 | cut -c1-160
else
echo "[realtime_b] starting the VLM (Cosmos-Reason1-7B; ~1-2 min to load)"
bash "$REPO/tools/realtime_b/vlm_up.sh" --wait
mkdir -p "$HOME/m20_sim/vlmnav"
docker rm -f m20_vlmnav >/dev/null 2>&1 || true
docker run -d --name m20_vlmnav --network host --ipc host \
  -e ROS_DOMAIN_ID=42 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e PYTHONUNBUFFERED=1 \
  -v "$REPO/src/m20_vlmnav":/vlmnav:ro -v "$HOME/m20_sim/vlmnav":/logs \
  m20_splat:latest bash -lc "source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export PYTHONPATH=/vlmnav:\$PYTHONPATH; \
python3 -c 'from m20_vlmnav.vlmnav_node import main; main()' --ros-args -p log_dir:=/logs" >/dev/null
NODE_OK=0
for i in $(seq 1 30); do docker logs m20_vlmnav 2>&1 | grep -q "Real-time B up" && { NODE_OK=1; break; }; sleep 1; done
[ "$NODE_OK" = "1" ] || { echo "ERROR: the episode node did not come up:"; docker logs m20_vlmnav 2>&1 | tail -15; exit 1; }
docker logs m20_vlmnav 2>&1 | grep "Real-time B up" | head -1
fi

# 5. the Nav2 pose watchdog (as in the offline launcher)
mkdir -p "$HOME/m20_sim/watchdog"
bash "$REPO/tools/nav/nav2_pose_watchdog.sh" >> "$HOME/m20_sim/watchdog/watchdog.log" 2>&1 &
WATCHDOG_PID=$!

echo
echo "=================================================================="
if [ "$MODE" = "search" ]; then
  echo " REAL-TIME B:  VLFM-style search: value map + frontiers + OWLv2, no maps given (Godot, $SITE)"
  echo " COMMANDS:     tools/dev/m20 search \"refrigerator\"   |   tools/dev/m20 search stop"
  echo " LOGS:         ~/m20_sim/vlfm/<stamp>/   |   docker logs -f m20_vlfm   |   docker logs -f m20_detector"
else
  echo " REAL-TIME B:  VLM moves + Nav2 behaviors, no maps (Godot, $SITE) [mode $MODE]"
  echo " COMMANDS:     tools/dev/m20 vlm \"go to the ladder\"   |   tools/dev/m20 vlm stop"
  echo " LOGS:         ~/m20_sim/vlmnav/<stamp>/   |   docker logs -f m20_vlmnav"
fi
echo " >>> LEAVE THIS TERMINAL OPEN <<<   (Ctrl-C here = shut the ROS stack down)"
echo "=================================================================="
# `sleep &` + wait: bash runs a trap only after the foreground command ends, so a plain `sleep 3600` would defer a
# SIGTERM (realtime_b_down.sh) by up to an hour and then tear down whatever stack was started next
while true; do sleep 3600 & wait $!; done
