#!/usr/bin/env bash
# Put the robot back at the spawn between demo runs: sim reset (a teleport odometry cannot see), then tell AMCL
# (/initialpose at the scene's "start" waypoint, the harness's covariance). Prints believed vs true pose after.
#   bash tools/demo/reset_robot.sh        (WSL, stack up)
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
read -r X Y YAW <<<"$(sed -n 's/^ *start: *\[\(.*\)\].*/\1/p' "$REPO/tools/sim/godot_circuit4.scene.yaml" | tr -d ' ' | tr ',' ' ')"
QZ=$(python3 -c "import math; print(round(math.sin($YAW / 2), 4))"); QW=$(python3 -c "import math; print(round(math.cos($YAW / 2), 4))")
# the sim's reset is a teleport: wait until the true pose IS the spawn (up to 10 s) rather than a fixed sleep
X="$X" Y="$Y" python3 - <<'PY'
import json, math, os, socket, sys, time
def call(d):
    s = socket.create_connection(("127.0.0.1", 7799), timeout=10); s.sendall((json.dumps(d) + "\n").encode()); b = b""
    while not b.endswith(b"\n"): b += s.recv(1 << 20)
    s.close(); return json.loads(b)
call({"cmd": "reset"})
x, y = float(os.environ["X"]), float(os.environ["Y"])
for i in range(50):
    t = call({"cmd": "telemetry"}).get("base_enu", {}).get("t")
    if t and math.hypot(t[0] - x, t[1] - y) < 0.3:
        break
    time.sleep(0.2)
else:
    print("[reset_robot] WARNING: the true pose did not reach the spawn within 10 s", file=sys.stderr)
PY
# AMCL's initial pose. -w 0: publish without waiting for a subscriber (a --loc gt stack has no AMCL: the default
# wait never returns); timeout: the CLI's discovery on a cold container can stall
docker exec docker-commander-1 bash -lc "$SRC; timeout 20 ros2 topic pub --once -w 0 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
 '{header: {frame_id: map}, pose: {pose: {position: {x: $X, y: $Y, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: $QZ, w: $QW}}, \
   covariance: [0.0025,0,0,0,0,0, 0,0.0025,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0.00274]}}' >/dev/null" || \
  echo "[reset_robot] WARNING: /initialpose not published (no AMCL, or the ROS CLI timed out)"
# the believed pose, once TF has settled (up to ~12 s); a missing TF is reported, not fatal (set -e)
BELIEVED=$(docker exec docker-commander-1 bash -lc "$SRC; timeout 12 ros2 run tf2_ros tf2_echo map base_link 2>&1 | grep -m1 Translation" || true)
echo "[reset_robot] believed: ${BELIEVED:-no map->base_link TF within 12 s}"
python3 - <<'PY'
import json, socket
s = socket.create_connection(("127.0.0.1", 7799), timeout=10); s.sendall(b'{"cmd":"telemetry"}\n'); b = b""
while not b.endswith(b"\n"): b += s.recv(1 << 20)
t = json.loads(b)["base_enu"]["t"]; print("[reset_robot] true:     x=%.2f y=%.2f" % (t[0], t[1]))
PY
