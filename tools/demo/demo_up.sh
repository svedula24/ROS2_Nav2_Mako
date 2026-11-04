#!/usr/bin/env bash
# Bring the Godot stack up for the NL navigation demo and wait until it can take an instruction.
#   bash tools/demo/demo_up.sh [godot_sim_up.sh options]      (WSL)
# Runs this worktree's tools/godot/godot_sim_up.sh in the background (log: ~/m20_sim/demo/godot_sim_up.log),
# waits for the stack, then for the NL node's warm-up. Then, on Windows:
#   python tools\demo\nlnav_demo_viewer.py "go to the ladder" --record ladder
# Down: bash tools/godot/godot_sim_down.sh
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LOG="$HOME/m20_sim/demo/godot_sim_up.log"; mkdir -p "$(dirname "$LOG")"
if pgrep -f "godot_sim_up.sh|realtime_b_up.sh|gazebo_sim_up.sh" >/dev/null; then
  # a launcher whose sim died (the Godot window closed) is only in the way: it keeps relaunching nothing
  if python3 -c "import socket; socket.create_connection(('127.0.0.1', 7799), timeout=1).close()" 2>/dev/null; then
    echo "[demo_up] a sim launcher is already running:"; pgrep -af "godot_sim_up.sh|realtime_b_up.sh|gazebo_sim_up.sh" | cut -c1-120
    echo "[demo_up] use it, or bring it down first (bash tools/godot/godot_sim_down.sh)"; exit 1
  fi
  echo "[demo_up] a stale launcher is running with no sim behind it: stopping it"
  pkill -f "godot_sim_up.sh|realtime_b_up.sh|gazebo_sim_up.sh|nav2_pose_watchdog.sh" || true
  # its exit trap tears the stack down; let that finish, or it would kill the stack we are about to start
  for i in $(seq 1 30); do pgrep -f "godot_sim_up.sh|realtime_b_up.sh|gazebo_sim_up.sh" >/dev/null || break; sleep 1; done
  bash "$REPO/tools/nav/kill_stack.sh" >/dev/null 2>&1 || true
fi
echo "[demo_up] starting the Godot stack from $REPO (log: $LOG)"
nohup bash "$REPO/tools/godot/godot_sim_up.sh" "$@" > "$LOG" 2>&1 &
for i in $(seq 1 120); do
  grep -q "COMMANDS:" "$LOG" && break
  grep -q "ERROR" "$LOG" && { echo "[demo_up] bringup failed:"; grep ERROR "$LOG"; exit 1; }
  sleep 3
done
grep -q "COMMANDS:" "$LOG" || { echo "[demo_up] the stack did not come up in 6 min (see $LOG)"; exit 1; }
grep -E "localization:|armed|starting NL" "$LOG" | sed 's/^/[demo_up] /'
echo "[demo_up] waiting for the NL node's warm-up (text encoder + one LLM parse)..."
for i in $(seq 1 90); do
  docker logs m20_nlnav 2>&1 | grep -qE "warm-up done|warm-up failed|Traceback" && break
  sleep 2
done
docker logs m20_nlnav 2>&1 | grep -E "objects kept|warm-up|Traceback" | sed -E 's/^\[INFO\] \[[0-9.]+\] \[m20_nlnav\]: /[demo_up] /' | tail -3
docker logs m20_nlnav 2>&1 | grep -q "warm-up done" || { echo "[demo_up] ERROR: NL node not ready (docker logs m20_nlnav)"; exit 1; }
# the mission server only drives an ARMED robot (ASSISTED); the bringup's arm can be refused on a fresh commander
SRC="source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
arm() { docker exec docker-commander-1 bash -lc "$SRC; timeout 10 ros2 service call /m20/set_mode m20_msgs/srv/SetMode '{mode: $1, requester: demo_up}' 2>/dev/null" | grep -c "accepted=True" || true; }
for i in 1 2 3; do
  [ "$(arm 2)" = "1" ] && break
  arm 0 >/dev/null; sleep 2
done
MODE=$(docker exec docker-commander-1 bash -lc "$SRC; timeout 5 ros2 topic echo /m20/mode --once 2>/dev/null | grep -E '^mode:' | awk '{print \$2}'")
[ "$MODE" = "2" ] || { echo "[demo_up] ERROR: robot not armed (mode=$MODE): tools/dev/m20 arm, or check /m20/failsafe"; exit 1; }
echo "[demo_up] armed ASSISTED (verified)"
echo
echo "[demo_up] READY. On Windows:  python tools\\demo\\nlnav_demo_viewer.py \"go to the ladder\" --record ladder"
echo "[demo_up] down:              bash tools/godot/godot_sim_down.sh"
