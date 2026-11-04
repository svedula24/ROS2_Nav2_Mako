#!/usr/bin/env bash
# Put the robot back at the spawn between demo runs: sim reset (a teleport odometry cannot see), then tell AMCL
# (/initialpose at the scene's "start" waypoint, the harness's covariance). Prints believed vs true pose after.
#   bash tools/demo/reset_robot.sh        (WSL, stack up)
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
read -r X Y YAW <<<"$(sed -n 's/^ *start: *\[\(.*\)\].*/\1/p' "$REPO/tools/sim/godot_circuit4.scene.yaml" | tr -d ' ' | tr ',' ' ')"
QZ=$(python3 -c "import math; print(round(math.sin($YAW / 2), 4))"); QW=$(python3 -c "import math; print(round(math.cos($YAW / 2), 4))")
python3 - <<'PY'
import json, socket, time
s = socket.create_connection(("127.0.0.1", 7799), timeout=10); s.sendall(b'{"cmd":"reset"}\n'); s.recv(4096); s.close()
time.sleep(3)
PY
docker exec docker-commander-1 bash -lc "$SRC; ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
 '{header: {frame_id: map}, pose: {pose: {position: {x: $X, y: $Y, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: $QZ, w: $QW}}, \
   covariance: [0.0025,0,0,0,0,0, 0,0.0025,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0.00274]}}' >/dev/null"
sleep 4
echo -n "[reset_robot] believed: "; docker exec docker-commander-1 bash -lc "$SRC; timeout 8 ros2 run tf2_ros tf2_echo map base_link 2>&1 | grep -m1 Translation"
python3 - <<'PY'
import json, socket
s = socket.create_connection(("127.0.0.1", 7799), timeout=10); s.sendall(b'{"cmd":"telemetry"}\n'); b = b""
while not b.endswith(b"\n"): b += s.recv(1 << 20)
t = json.loads(b)["base_enu"]["t"]; print("[reset_robot] true:     x=%.2f y=%.2f" % (t[0], t[1]))
PY
