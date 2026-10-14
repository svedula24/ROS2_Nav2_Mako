#!/usr/bin/env bash
# Nav2 robot-pose watchdog (host side). Nav2's costmaps silently stopped updating their robot pose on 3 of
# ~12 launches (2026-10-13, Godot sim): the footprint each costmap publishes kept the spawn pose and a frozen
# stamp while /tf was fine, the planner planned from the spawn and the controller steered a robot that, in
# its model, never moved (it drove into a desk). Not reproduced under a gdb trap since. This watchdog:
#   every INTERVAL s, reads both costmaps' published_footprint stamps; if one is older than MAX_AGE s for two
#   checks in a row -> (1) backtraces of planner_server + controller_server via a ptrace-capable gdb sidecar
#   (image m20_autonomy:gdb, if present) into $OUT, (2) relaunch nav2 (same command as bringup). The
#   commander's mode is untouched by a nav2 relaunch, so there is nothing to re-arm.
#   A probe that FAILS (container down, ros2 error) is logged to $OUT/probe_errors.log and is not a strike:
#   only a costmap that answers late counts, so a dead container can't cause a relaunch loop.
#
#   bash tools/nav/nav2_pose_watchdog.sh [--once]        (godot_sim_up.sh starts it in the background)
# Env: C (container, docker-commander-1), INTERVAL (10), MAX_AGE (3), OUT (~/m20_sim/watchdog)
set -u
C=${C:-docker-commander-1}; INTERVAL=${INTERVAL:-10}; MAX_AGE=${MAX_AGE:-3}; OUT=${OUT:-$HOME/m20_sim/watchdog}
SRC='source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp'
mkdir -p "$OUT"
log() { echo "[nav2_watchdog $(date +%H:%M:%S)] $*"; }

ages() {   # prints "<local age> <global age>" in seconds (999 = no message in 3 s)
  docker exec -i "$C" bash -lc "$SRC; python3 -" <<'PY'
import time, rclpy
from geometry_msgs.msg import PolygonStamped
rclpy.init(); n = rclpy.create_node("nav2_pose_watchdog"); st = {}
for k in ("local", "global"):
    n.create_subscription(PolygonStamped, f"/{k}_costmap/published_footprint",
                          (lambda k: lambda m: st.__setitem__(k, m.header.stamp.sec + m.header.stamp.nanosec * 1e-9))(k), 10)
t = time.time()
while time.time() - t < 3 and len(st) < 2: rclpy.spin_once(n, timeout_sec=0.1)
now = time.time()
print(" ".join(f"{now - st[k]:.1f}" if k in st else "999" for k in ("local", "global")))
PY
}

backtrace() {   # $1 = process name
  docker image inspect m20_autonomy:gdb >/dev/null 2>&1 || { log "no m20_autonomy:gdb image: skipping backtrace of $1"; return; }
  local pid; pid=$(docker exec "$C" bash -lc "pgrep -f '/opt/ros/humble/lib/nav2_[a-z_]*/$1 ' | head -1")
  [ -z "$pid" ] && return
  local f="$1_$(date +%Y%m%d_%H%M%S).txt"
  docker run --rm --pid=container:"$C" --cap-add SYS_PTRACE --security-opt seccomp=unconfined -v "$OUT":/bt \
    m20_autonomy:gdb bash -lc "timeout 150 gdb -iex 'set sysroot /' -p $pid -batch -ex 'set pagination off' -ex 'thread apply all bt 45' > /bt/$f 2>&1"
  log "backtrace of $1 (pid $pid) -> $OUT/$f"
}

relaunch_nav2() {
  docker exec "$C" bash -lc "cp /cfg/out/nav2.log /cfg/out/nav2_frozen_$(date +%H%M%S).log 2>/dev/null; ps -eo pid,args | grep -E 'nav2_|navigation_launch|lifecycle_manager' | grep -vE 'grep|map_server|lifecycle_manager_map' | awk '{print \$1}' | xargs -r kill -9; sleep 2"
  local t0; t0=$(date +%s)
  docker exec -d "$C" bash -lc "$SRC; ros2 launch nav2_bringup navigation_launch.py params_file:=/cfg/nav2_params.yaml use_sim_time:=false >/cfg/out/nav2.log 2>&1"
  for i in $(seq 1 90); do
    local ts; ts=$(docker exec "$C" bash -lc "grep 'Managed nodes are active' /cfg/out/nav2.log 2>/dev/null | tail -1 | sed -E 's/.*\[([0-9]+)\.[0-9]+\].*/\1/'")
    [ -n "$ts" ] && [ "$ts" -ge "$t0" ] && { log "nav2 relaunched and active"; return 0; }
    sleep 1
  done
  log "nav2 relaunch: not active after 90 s"
}

strikes=0
while true; do
  if ! probe=$(ages 2>>"$OUT/probe_errors.log"); then
    log "probe failed (see $OUT/probe_errors.log): not counted as a strike"
    [ "${1:-}" = "--once" ] && exit 1
    sleep "$INTERVAL"; continue
  fi
  read -r la ga <<< "$probe"
  frozen=$(python3 -c "print(int(float('$la') > $MAX_AGE or float('$ga') > $MAX_AGE))")
  if [ "$frozen" = "1" ]; then
    strikes=$((strikes + 1))
    log "footprint age local ${la}s global ${ga}s (strike $strikes)"
    if [ "$strikes" -ge 2 ]; then
      log "NAV2 ROBOT POSE FROZEN: capturing backtraces, then relaunching nav2"
      backtrace planner_server; backtrace controller_server
      relaunch_nav2
      strikes=0
    fi
  else
    strikes=0
  fi
  [ "${1:-}" = "--once" ] && { echo "local ${la}s global ${ga}s"; exit 0; }
  sleep "$INTERVAL"
done
