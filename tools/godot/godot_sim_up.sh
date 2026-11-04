#!/usr/bin/env bash
# The Mako stack on the Godot sim (Bright Robotics Simulator, M20 on circuit4). STAYS IN THE
# FOREGROUND: Ctrl-C tears the ROS stack down (the Godot window stays unless --stop-sim).
#
#   bash tools/godot/godot_sim_up.sh              Godot window + bridge + commander/Nav2 (site map)/mission
#                                                 + NL navigation (offline phase: LLM + object map)
#   bash tools/godot/godot_sim_up.sh --viser      ... plus the Viser viewer on http://localhost:8080
#   bash tools/godot/godot_sim_up.sh --detect     ... plus the live OWLv2 detector (mapping runs)
#   bash tools/godot/godot_sim_up.sh --no-site-map   live costmaps only (no Nav2 static map)
#   bash tools/godot/godot_sim_up.sh --object-map <json>   default ~/m20_sim/semmap/circuit4/object_map.json
#   bash tools/godot/godot_sim_up.sh --stop-sim   also quit the Godot sim on exit
#   bash tools/godot/godot_sim_up.sh --loc gt     localization: default amcl = AMCL on the site map from DRIFTING
#                                                 odometry, like the real robot; gt = the sim's ground-truth pose;
#                                                 odom = drifting odometry, NO correction (an eval baseline).
#                                                 With --no-site-map the default is gt (AMCL needs the map).
#   then:  tools/dev/m20 say "go to the ladder"
# The local LLM (Ollama, NLNAV_LLM_MODEL, default qwen2.5:7b-instruct) runs in m20_ollama and is kept
# running between launches (the model stays loaded); NLNAV_LLM=claude switches to the Anthropic API.
#
# Needs WSL mirrored networking (the sim serves its streams on Windows' 127.0.0.1 only), the Godot
# sim (GODOT_EXE, GODOT_SIM) and the circuit4 site baked on Windows. A sim that already answers on
# 127.0.0.1:7799 is reused as is.
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
GODOT_EXE="${GODOT_EXE:-/mnt/c/Users/$(cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r')/tools/godot-4.7.1/Godot_v4.7.1-stable_win64.exe}"
GODOT_SIM="${GODOT_SIM:-$(dirname "$REPO")/stateful-drone-sim}"
SITE="${GODOT_SITE:-circuit4_cam_mesh}"
SCENE="godot_circuit4.xml"                      # tools/sim/godot_circuit4.scene.yaml (waypoints, splat frame)
PORT="${VIEW_PORT:-8080}"
VISER=0; STOP_SIM=0; DETECT=0; SITE_MAP="tools/sim/godot_circuit4_map"; LOC="${M20_LOC:-}"
OBJECT_MAP="$HOME/m20_sim/semmap/circuit4/object_map.json"
LLM="${NLNAV_LLM:-ollama}"; LLM_MODEL="${NLNAV_LLM_MODEL:-qwen2.5:7b-instruct}"
while [ $# -gt 0 ]; do
  case "$1" in
    --viser) VISER=1 ;;
    --stop-sim) STOP_SIM=1 ;;
    --detect) DETECT=1 ;;
    --no-site-map) SITE_MAP="" ;;
    --object-map) OBJECT_MAP="$2"; shift ;;
    --loc) LOC="$2"; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)"; exit 2 ;;
  esac
  shift
done
# default: AMCL (since 2026-10-15), unless there is no site map to localize against
[ -n "$LOC" ] || { if [ -n "$SITE_MAP" ]; then LOC=amcl; else LOC=gt; fi; }
case "$LOC" in
  gt) ODOM=truth; GT_LOC=1; POSE_SOURCE=odom_true ;;
  odom) ODOM=drift; GT_LOC=1; POSE_SOURCE=tf ;;   # baseline: drifting odometry, map->odom fixed at identity
  amcl) ODOM=drift; GT_LOC=0; POSE_SOURCE=tf
        [ -n "$SITE_MAP" ] || { echo "ERROR: --loc amcl needs the site map (drop --no-site-map)"; exit 2; } ;;
  *) echo "unknown --loc $LOC (gt | amcl | odom)"; exit 2 ;;
esac
# the spawn = the scene's "start" waypoint: AMCL's initial pose
START_POSE=$(sed -n 's/^ *start: *\[\(.*\)\].*/\1/p' "$REPO/tools/sim/${SCENE%.xml}.scene.yaml" | tr -d ' ' | tr ',' ' ')

api_up() { python3 - <<'PY'
import socket, sys
try:
    socket.create_connection(("127.0.0.1", 7799), timeout=1).close()
except OSError:
    sys.exit(1)
PY
}
sim_quit() { python3 - <<'PY' || true
import json, socket
s = socket.create_connection(("127.0.0.1", 7799), timeout=2)
s.sendall(b'{"cmd":"quit"}\n')
PY
}

cleanup() {
  trap - INT TERM EXIT
  echo; echo "[godot_sim_up] tearing down..."
  # `|| true`: under set -e a failed kill (watchdog already gone) would abort the trap before kill_stack
  [ -n "${WATCHDOG_PID:-}" ] && { kill "$WATCHDOG_PID" 2>/dev/null || true; }
  bash "$REPO/tools/nav/kill_stack.sh" >/dev/null 2>&1 || true
  [ "$STOP_SIM" = "1" ] && { sim_quit; echo "[godot_sim_up] Godot sim asked to quit."; }
  echo "[godot_sim_up] down."
  exit 0
}
trap cleanup INT TERM EXIT
bash "$REPO/tools/nav/kill_stack.sh" >/dev/null 2>&1 || true

# 1. the Godot sim (Windows), windowed: headless renders no camera or lidar
if api_up; then
  echo "[godot_sim_up] reusing the Godot sim already on 127.0.0.1:7799"
else
  [ -f "$GODOT_EXE" ] || { echo "ERROR: Godot not found at $GODOT_EXE (set GODOT_EXE)"; exit 1; }
  [ -d "$GODOT_SIM/godot_sim" ] || { echo "ERROR: sim repo not found at $GODOT_SIM (set GODOT_SIM)"; exit 1; }
  echo "[godot_sim_up] starting the Godot sim ($SITE, M20) in a window..."
  # `start` detaches it as a plain Windows process (a direct child of this WSL session dies with
  # it). WSL interop still waits on the handles Godot inherits, so don't wait for cmd.exe.
  (cd /mnt/c && cmd.exe /c start "Godot sim" "$(wslpath -w "$GODOT_EXE")" --path "$(wslpath -w "$GODOT_SIM/godot_sim")" \
    -- --api --scene=splat_scene --site="$SITE" --vehicle=m20 --videofps=10 </dev/null >/dev/null 2>&1 &)
  for i in $(seq 1 90); do api_up && break; sleep 2; done
  api_up || { echo "ERROR: the sim's API never came up (check the Godot window for errors)"; exit 1; }
  sleep 5                                       # let the splat settle before streaming
fi
# Park the window almost entirely off-screen so the desk stays usable (rendering continues; only a MINIMIZED
# window stops the camera/lidar). GODOT_SHOW=1 keeps it visible; `godot_window.ps1 show` brings it back any time.
if [ "${GODOT_SHOW:-0}" != "1" ]; then
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w "$REPO/tools/godot/godot_window.ps1")" park 2>&1 | tr -d '\r' || true   # not fatal, but errors stay visible
fi

# 2. the bridge: sim <-> ROS topics (replaces the MuJoCo container + locomotion bridge)
echo "[godot_sim_up] starting the Godot ROS bridge"
docker rm -f m20_godot_bridge >/dev/null 2>&1 || true
docker run -d --name m20_godot_bridge --network host --ipc host \
  -e ROS_DOMAIN_ID=42 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e PYTHONUNBUFFERED=1 \
  -e GODOT_VMAX="${GODOT_VMAX:-0.5}" -e GODOT_DEPTH_OBSTACLES="${GODOT_DEPTH_OBSTACLES:-1}" -e GODOT_ODOM="$ODOM" \
  -v "$REPO/tools/godot":/godot:ro \
  m20_sim:latest bash -lc "source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; python3 /godot/godot_bridge.py" >/dev/null
sleep 6

# AMCL starts from the KNOWN spawn, so put the robot there first (a reused sim may have it anywhere)
if [ "$LOC" != gt ]; then      # also odom: drift is seeded at the true pose and map->odom is the identity
  python3 - <<'PY' || echo "[godot_sim_up] WARNING: could not reset the sim to the spawn; AMCL's initial pose may be wrong"
import json, socket, time
s = socket.create_connection(("127.0.0.1", 7799), timeout=10); s.sendall(b'{"cmd":"reset"}\n'); s.recv(4096); s.close()
time.sleep(3)
PY
fi

# 3. the usual stack, minus the MuJoCo pieces (bringup M20_EXTERNAL_SIM=1); localization per --loc
# NL navigation runs in its own container (step 4b: it needs OWLv2's text encoder), not the commander.
echo "[godot_sim_up] localization: $LOC (odometry: $ODOM$( [ "$LOC" = amcl ] && echo ", initial pose $START_POSE"))"
M20_EXTERNAL_SIM=1 M20_SCENE="$SCENE" M20_GT_LOC=$GT_LOC M20_LOC=$LOC M20_START_POSE="$START_POSE" \
M20_NO_GOAL=1 M20_SPLAT_CAMERA=0 M20_NLNAV=0 \
M20_SITE_MAP="$SITE_MAP" M20_MAX_VEL="${GODOT_VMAX:-0.5}" bash "$REPO/tools/nav/bringup_plan_a.sh"
# the mission server (m20 say / goto) needs ASSISTED; bringup leaves TELEOP
SRC="source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
arm() { docker exec docker-commander-1 bash -lc "$SRC; timeout 15 ros2 service call /m20/set_mode m20_msgs/srv/SetMode '{mode: $1, requester: godot_sim_up}' 2>/dev/null" | grep -c "accepted=True" || true; }
ARMED=0
for i in 1 2 3; do                                   # refused (a latched ESTOP, a commander still settling): via IDLE
  [ "$(arm 2)" = "1" ] && { ARMED=1; break; }
  arm 0 >/dev/null; sleep 2
done
[ "$ARMED" = "1" ] || { echo "[godot_sim_up] ERROR: the commander refused ASSISTED 3 times (check /m20/failsafe: docker logs docker-commander-1)"; exit 1; }
# the service said accepted; /m20/mode (latched, published on change) is what the mission server reads. Read with a
# transient_local subscriber in rclpy: `ros2 topic echo` goes through the CLI daemon, which is wedged on this
# container often enough that the read came back empty after 25 s (2026-11-04) while rclpy had the sample at once.
MODE=$(docker exec -i docker-commander-1 bash -lc "$SRC; python3 -" <<'PY'
import time, rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from m20_msgs.msg import RobotMode
rclpy.init(); n = rclpy.create_node("arm_probe"); got = []
n.create_subscription(RobotMode, "/m20/mode", got.append,
                      QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
t = time.monotonic()
while not got and time.monotonic() - t < 15:
    rclpy.spin_once(n, timeout_sec=0.2)
print(got[-1].mode if got else "")
rclpy.shutdown()
PY
)
[ "$MODE" = "2" ] || { echo "[godot_sim_up] ERROR: /m20/mode reads '$MODE' after an accepted arm (expected 2 = ASSISTED)"; exit 1; }
echo "[godot_sim_up] armed ASSISTED (verified on /m20/mode)"

# 4a. local LLM for the structured query (kept running across launches: the model stays loaded)
if [ "$LLM" = "ollama" ]; then
  if ! docker ps --format '{{.Names}}' | grep -q '^m20_ollama$'; then
    echo "[godot_sim_up] starting the local LLM (Ollama, $LLM_MODEL)"
    docker rm -f m20_ollama >/dev/null 2>&1 || true
    mkdir -p "$HOME/m20_sim/ollama"
    docker run -d --name m20_ollama --gpus all --network host -e OLLAMA_HOST=127.0.0.1:11434 \
      -e OLLAMA_KEEP_ALIVE=30m -v "$HOME/m20_sim/ollama":/root/.ollama ollama/ollama:latest >/dev/null
    for i in $(seq 1 30); do docker exec m20_ollama ollama list >/dev/null 2>&1 && break; sleep 1; done
  fi
  docker exec m20_ollama ollama list | grep -q "^${LLM_MODEL}" || {
    echo "[godot_sim_up] pulling $LLM_MODEL (once, a few GB)..."; docker exec m20_ollama ollama pull "$LLM_MODEL"; }
fi

# 4b. NL navigation: LLM query -> object map lookup -> approach pose -> RunMission
[ -f "$OBJECT_MAP" ] || echo "[godot_sim_up] NOTE: no object map at $OBJECT_MAP: only named places will resolve (tools/semmap/record_mapping_run.sh, then build_object_map.sh)"
echo "[godot_sim_up] starting NL navigation (llm=$LLM, model=$LLM_MODEL)"
# tools/godot/nlnav_up.sh owns the container start (tools/nlnav_eval/eval.sh restarts it with another model)
bash "$REPO/tools/godot/nlnav_up.sh" --object-map "$OBJECT_MAP" --scene "$SCENE" --llm "$LLM" --model "$LLM_MODEL" \
  --min-label-score "${MIN_LABEL_SCORE:-0.4}" --pose-source "$POSE_SOURCE" >/dev/null

# 4b'. Nav2 robot-pose watchdog: relaunches nav2 if a costmap's published footprint stops updating (the
# silent frozen-pose failure of 2026-10-13; see the script's header). Log + backtraces in ~/m20_sim/watchdog
# (persistent, ENGINEERING_RULES 4).
mkdir -p "$HOME/m20_sim/watchdog"
bash "$REPO/tools/nav/nav2_pose_watchdog.sh" >> "$HOME/m20_sim/watchdog/watchdog.log" 2>&1 &
WATCHDOG_PID=$!

# 4c. live object detector on the sim's camera (opt-in: the offline map is the source of truth)
if [ "$DETECT" = "1" ]; then
  echo "[godot_sim_up] starting object detector"
  docker rm -f m20_detector >/dev/null 2>&1 || true
  docker run -d --name m20_detector --gpus all --network host --ipc host \
    -e ROS_DOMAIN_ID=42 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v "$REPO/tools/splat":/splat_tools:ro \
    m20_splat:latest python3 -u /splat_tools/detector_node.py >/dev/null
fi

# 5. optional Viser viewer (photoreal splat at the robot's pose; NL panel, mini-map off: no topdown yet)
if [ "$VISER" = "1" ]; then
  SPLAT_FILE=$(awk '/^  file:/{print $2; exit}' "$REPO/tools/sim/${SCENE%.xml}.scene.yaml")
  [ -f "$HOME/m20_sim/splats/$SPLAT_FILE" ] || { echo "ERROR: ~/m20_sim/splats/$SPLAT_FILE missing (needed for --viser)"; exit 1; }
  echo "[godot_sim_up] starting Viser on :$PORT"
  M20_DESC="$HOME/m20_sim/sdk_deploy/src/M20_sdk_deploy/M20_description"
  GL_ARGS=(-e MUJOCO_GL=egl -e NVIDIA_DRIVER_CAPABILITIES=all)
  if [ -d /usr/lib/wsl/lib ]; then
    GL_ARGS+=(-v /usr/lib/wsl:/usr/lib/wsl:ro -e LD_LIBRARY_PATH=/usr/lib/wsl/lib
              -e GALLIUM_DRIVER=d3d12 -e MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA)
  fi
  docker rm -f m20_splat_view >/dev/null 2>&1 || true
  docker run -d --name m20_splat_view --gpus all --network host --ipc host \
    -e ROS_DOMAIN_ID=42 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e VIEW_PORT="$PORT" "${GL_ARGS[@]}" \
    -e SPLAT_PLY=/splat/$SPLAT_FILE -e SPLAT_SCENE=/scene/${SCENE%.xml}.scene.yaml \
    -v "$HOME/m20_sim/splats":/splat:ro -v "$REPO/tools/sim":/scene:ro \
    -v "$REPO/tools/splat":/splat_tools:ro -v "$HOME/m20_sim/m20_urdf":/urdf:ro \
    -v "$M20_DESC":/model:ro -v "$HOME/m20_sim/laps":/laps \
    m20_splat:latest python3 /splat_tools/splat_viewer.py >/dev/null
fi

echo
echo "=================================================================="
echo " GODOT SIM:  the Godot window (M20 on $SITE)"
[ "$VISER" = "1" ] && echo " VISER:      http://localhost:$PORT"
echo " COMMANDS:   tools/dev/m20 say \"go to the ladder\"   |   tools/dev/m20 say stop"
echo " MAPPING:    bash tools/semmap/record_mapping_run.sh   then   bash tools/semmap/build_object_map.sh <run>"
echo " >>> LEAVE THIS TERMINAL OPEN <<<   (Ctrl-C here = shut the ROS stack down)"
echo " GODOT WINDOW: parked off-screen (never minimize it: minimized = no camera/lidar). Show it: powershell tools/godot/godot_window.ps1 show"
echo "=================================================================="
# The bridge is the stack's heartbeat: gone for good = tear everything down. A `docker restart
# m20_godot_bridge` (picking up a code change) stops it for a second or two; give it 15 s to come
# back before deciding it has exited, otherwise a restart takes the whole stack with it.
gone=0
while [ "$gone" -lt 5 ]; do
  if docker ps --format '{{.Names}}' | grep -q '^m20_godot_bridge$'; then
    gone=0
    docker logs -f --since 1s m20_godot_bridge 2>&1 || true
  else
    gone=$((gone + 1))
  fi
  sleep 3
done
echo "[godot_sim_up] bridge exited — cleaning up the rest..."
