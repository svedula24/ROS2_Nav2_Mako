#!/usr/bin/env bash
# The lidar stage Nav2 sees on the real M20 and in the robot-shape sim: two RoboSense clouds, each in its own frame
# (/rslidar_points_front in rslidar_front, /rslidar_points_rear in rslidar_rear), turned into /scan_front and
# /scan_rear by one pointcloud_to_laserscan node each (height band -0.36..0.54 m around base_link, 1 deg bins,
# 0.5-12 m: src/m20_navigation/config/pointcloud_to_laserscan.yaml). Shared byte for byte by the Mako sim launcher
# (tools/nav/bringup_plan_a.sh, M20_LIDAR_SPLIT=1) and the robot launcher (m20-sim-to-real tools/robot/robot_up.sh).
#
#   bash lidar_stage.sh <container> [--tf]      start the converters in <container>; --tf also publishes the two
#                                               static mount transforms (the robot: no sim bridge publishes them)
#   bash lidar_stage.sh <container> --check     fail unless both clouds AND both scans arrive within 10 s
#
# Mount numbers (--tf) = the robot's URDF, measured 2026-10-16: front (+0.32028, 0, -0.013) rpy (0, -1.57079, -3.14159),
# rear (-0.32028, 0, -0.013) rpy (0, -1.57079, 0): both units hang pitched -90 deg under the body and look DOWN/outward.
set -euo pipefail
C="${1:?usage: lidar_stage.sh <container> [--tf | --check]}"; shift
MODE="${1:-start}"
SRC="source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42} RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
docker exec "$C" test -f /cfg/pointcloud_to_laserscan.yaml || { echo "lidar_stage: /cfg/pointcloud_to_laserscan.yaml not staged in $C"; exit 1; }

if [ "$MODE" = "--check" ]; then
  # read ROS state with rclpy, not the ros2 CLI daemon (it wedges in the commander container)
  docker exec -i "$C" bash -lc "$SRC; timeout 20 python3 -" <<'PY'
import sys, time, rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, PointCloud2
rclpy.init(); n = rclpy.create_node("lidar_stage_check"); got = {}
for side in ("front", "rear"):
    n.create_subscription(PointCloud2, f"/rslidar_points_{side}", (lambda s: lambda m: got.__setitem__(f"cloud_{s}", m.width))(side), qos_profile_sensor_data)
    n.create_subscription(LaserScan, f"/scan_{side}", (lambda s: lambda m: got.__setitem__(f"scan_{s}", sum(1 for r in m.ranges if r == r and r < m.range_max)))(side), qos_profile_sensor_data)
t = time.monotonic()
while len(got) < 4 and time.monotonic() - t < 10:
    rclpy.spin_once(n, timeout_sec=0.2)
print("  lidar stage:", ", ".join(f"{k} {v}" for k, v in sorted(got.items())) or "nothing")
missing = [k for k in ("cloud_front", "cloud_rear", "scan_front", "scan_rear") if k not in got]
if missing:
    print("  lidar stage: MISSING", " ".join(missing)); sys.exit(1)
PY
  exit 0
fi

if [ "$MODE" = "--tf" ]; then
  echo "  lidar stage: static TFs base_link -> rslidar_front / rslidar_rear (robot mounts)"
  docker exec -d "$C" bash -lc "$SRC; ros2 run tf2_ros static_transform_publisher --x 0.32028 --y 0 --z -0.013 --roll 0 --pitch -1.57079 --yaw -3.14159 --frame-id base_link --child-frame-id rslidar_front"
  docker exec -d "$C" bash -lc "$SRC; ros2 run tf2_ros static_transform_publisher --x -0.32028 --y 0 --z -0.013 --roll 0 --pitch -1.57079 --yaw 0 --frame-id base_link --child-frame-id rslidar_rear"
fi
echo "  lidar stage: pointcloud_to_laserscan x2 (/rslidar_points_front|rear -> /scan_front|rear)"
docker exec "$C" mkdir -p /cfg/out
for side in front rear; do
  docker exec -d "$C" bash -lc "$SRC; ros2 run pointcloud_to_laserscan pointcloud_to_laserscan_node --ros-args -r __node:=laserscan_$side -r cloud_in:=/rslidar_points_$side -r scan:=/scan_$side --params-file /cfg/pointcloud_to_laserscan.yaml >/cfg/out/laserscan_$side.log 2>&1"
done
