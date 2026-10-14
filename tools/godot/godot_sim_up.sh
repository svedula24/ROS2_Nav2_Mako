#!/usr/bin/env bash
# The Mako stack on the Godot sim (Bright Robotics Simulator, M20 on circuit4). STAYS IN THE
# FOREGROUND: Ctrl-C tears the ROS stack down (the Godot window stays unless --stop-sim).
#
#   bash tools/godot/godot_sim_up.sh            Godot window + bridge + commander/Nav2/mission/NL/detector
#   bash tools/godot/godot_sim_up.sh --viser    ... plus the Viser viewer on http://localhost:8080
#   bash tools/godot/godot_sim_up.sh --stop-sim also quit the Godot sim on exit
#   then:  tools/dev/m20 say "go to the ladder"
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
VISER=0; STOP_SIM=0
for a in "$@"; do
  case "$a" in
    --viser) VISER=1 ;;
    --stop-sim) STOP_SIM=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option $a (see --help)"; exit 2 ;;
  esac
done

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

# 2. the bridge: sim <-> ROS topics (replaces the MuJoCo container + locomotion bridge)
echo "[godot_sim_up] starting the Godot ROS bridge"
docker rm -f m20_godot_bridge >/dev/null 2>&1 || true
docker run -d --name m20_godot_bridge --network host --ipc host \
  -e ROS_DOMAIN_ID=42 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e PYTHONUNBUFFERED=1 \
  -e GODOT_VMAX="${GODOT_VMAX:-0.5}" -e GODOT_DEPTH_OBSTACLES="${GODOT_DEPTH_OBSTACLES:-1}" \
  -v "$REPO/tools/godot":/godot:ro \
  m20_sim:latest bash -lc "source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; python3 /godot/godot_bridge.py" >/dev/null
sleep 6

# 3. the usual stack, minus the MuJoCo pieces (bringup M20_EXTERNAL_SIM=1); ground-truth localisation
M20_EXTERNAL_SIM=1 M20_SCENE="$SCENE" M20_GT_LOC=1 M20_NO_GOAL=1 M20_SPLAT_CAMERA=0 \
M20_MAX_VEL="${GODOT_VMAX:-0.5}" bash "$REPO/tools/nav/bringup_plan_a.sh"
# the mission server (m20 say / goto) needs ASSISTED; bringup leaves TELEOP
SRC="source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
arm() { docker exec docker-commander-1 bash -lc "$SRC; ros2 service call /m20/set_mode m20_msgs/srv/SetMode '{mode: $1, requester: godot_sim_up}'" | grep -c "accepted=True" || true; }
if [ "$(arm 2)" = "0" ]; then arm 0 >/dev/null; arm 2 >/dev/null; fi   # refused (e.g. latched ESTOP): via IDLE
echo "[godot_sim_up] armed ASSISTED"

# 4. object detector on the sim's camera (same container as the splat demo)
if [ "${M20_DETECT:-1}" = "1" ]; then
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
echo " >>> LEAVE THIS TERMINAL OPEN <<<   (Ctrl-C here = shut the ROS stack down)"
echo "=================================================================="
while docker ps --format '{{.Names}}' | grep -q '^m20_godot_bridge$'; do
  docker logs -f --since 1s m20_godot_bridge 2>&1 || sleep 3
done
echo "[godot_sim_up] bridge exited — cleaning up the rest..."
