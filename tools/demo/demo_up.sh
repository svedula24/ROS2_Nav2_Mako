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
LAUNCHER=$!
fail() {
  # the launcher we started must not outlive a failed demo_up: its exit trap tears the stack down, and a rerun
  # would otherwise refuse ("a sim launcher is already running")
  echo "[demo_up] ERROR: $1"
  if kill -0 "$LAUNCHER" 2>/dev/null; then
    echo "[demo_up] stopping the launcher (pid $LAUNCHER) and its stack"
    kill -TERM "$LAUNCHER" 2>/dev/null || true       # TERM: a background job of a non-interactive shell ignores INT
    for i in $(seq 1 30); do kill -0 "$LAUNCHER" 2>/dev/null || break; sleep 1; done
    kill -0 "$LAUNCHER" 2>/dev/null && { kill -9 "$LAUNCHER" 2>/dev/null || true; bash "$REPO/tools/nav/kill_stack.sh" >/dev/null 2>&1 || true; }
  fi
  exit 1
}
for i in $(seq 1 120); do
  grep -aq "COMMANDS:" "$LOG" && break
  grep -aq "ERROR" "$LOG" && { grep -a ERROR "$LOG" | sed 's/^/[demo_up] /'; fail "bringup failed (see $LOG)"; }
  kill -0 "$LAUNCHER" 2>/dev/null || fail "the launcher exited before the stack was up (see $LOG)"
  sleep 3
done
grep -aq "COMMANDS:" "$LOG" || fail "the stack did not come up in 6 min (see $LOG)"
grep -aE "localization:|armed|starting NL" "$LOG" | sed 's/^/[demo_up] /'
echo "[demo_up] waiting for the NL node's warm-up (text encoder + one LLM parse)..."
for i in $(seq 1 90); do
  docker logs m20_nlnav 2>&1 | grep -qE "warm-up done|warm-up failed|Traceback" && break
  sleep 2
done
docker logs m20_nlnav 2>&1 | grep -E "objects kept|warm-up|Traceback" | sed -E 's/^\[INFO\] \[[0-9.]+\] \[m20_nlnav\]: /[demo_up] /' | tail -3
docker logs m20_nlnav 2>&1 | grep -q "warm-up done" || fail "NL node not ready (docker logs m20_nlnav)"
# the mission server only drives an ARMED robot: godot_sim_up.sh arms, retries via IDLE and verifies /m20/mode
grep -aq "armed ASSISTED (verified" "$LOG" || fail "the launcher did not verify the arm (see $LOG)"
echo
echo "[demo_up] READY. On Windows:  python tools\\demo\\nlnav_demo_viewer.py \"go to the ladder\" --record ladder"
echo "[demo_up] down:              bash tools/godot/godot_sim_down.sh"
