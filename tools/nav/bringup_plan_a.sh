#!/usr/bin/env bash
# ============================================================================
# Plan A bringup: autonomous 2D navigation in the oil & gas field sim, using the
# PRODUCTION stack -- Nav2 (Regulated Pure Pursuit + NavFn/A*) + slam_toolbox.
#
#   sim (MuJoCo, m20_sim)  ->  /scan /odom /IMU + TF odom->base_link
#   bridge (m20_autonomy)  ->  /cmd_vel -> wheel/leg joint commands
#   slam_toolbox           ->  /map + TF map->odom  (scan-matching)
#   Nav2                   ->  global plan -> costmap-aware control -> /cmd_vel
#
# Honest scope (Plan A):
#   * Perception (/scan), map-building, global planning, obstacle avoidance and
#     path-following are ALL real (live 10 m LiDAR, costmaps, RPP replanning).
#   * Localization is currently fed by the sim's ground-truth odom TF, so
#     slam_toolbox's map->odom correction is ~identity. Honest dead-reckoned odom
#     (real drift for SLAM to correct) is the immediate Plan-A follow-on.
#   * The 2D scan (horizontal ring at z=0.35 m) does NOT see the ground pipelines
#     (tops at 0.30 m). Low-obstacle traversability is Plan B (3D perception).
#
# Prereqs: docker-commander-1 running (m20_autonomy image, ROS_DOMAIN_ID=42),
#          m20_sim:latest image, oil_gas_field.xml installed next to M20.xml.
# Usage:   bash tools/nav/bringup_plan_a.sh              # full run + verify
#          GOAL_X=-6 GOAL_Y=-0.5 bash tools/nav/bringup_plan_a.sh
# Flags (phase-map-loc):
#   M20_STATIC_MAP=maps/oil_gas_field  navigate against the PRE-BUILT map:
#       slam_toolbox localization mode (posegraph) + static-layer global costmap
#       (planner sees the whole field at t=0). Empty = live-SLAM path (master).
#   M20_STANDUP=1  spawn the sim FOLDED and rise via the commander-gated behavior
#       engine (standup, TIER_VENDOR) BEFORE the bridge starts. Real standup on
#       camera; joints logged for the replay render.
#   M20_NO_GOAL=1  bring the stack up but send no goal (mapping / coverage runs).
# Flags (3dgs-sim-env):
#   M20_SCENE=indoor_splat.xml  run a different world (any tools/sim/*.xml installed by
#       setup_sim.sh). Default oil_gas_field.xml. Non-default scenes have no pre-built
#       map (live SLAM only) and need explicit GOAL_X/GOAL_Y unless M20_NO_GOAL=1.
#   M20_SPLAT_CAMERA=1  photoreal /camera/image_raw rendered from the scene's Gaussian
#       splat (tools/splat/splat_camera_node.py, m20_splat image, GPU). Needs a
#       <scene>.scene.yaml next to the scene and the PLY in ~/m20_sim/splats/.
# ============================================================================
set -e
# FRAME CONTRACT GUARD (ship rule): M20_EKF_TF is a MAP-PRODUCTION tool only —
# the smooth GPS-pulled prior may never own odom->base_link while navigating
# against the shipped map. Runtime tree stays REP-105: sim dead-reckon owns
# odom->base_link, slam localization owns map->odom. See SECOND_BRAIN §3.
if [ "${M20_EKF_TF:-0}" = "1" ] && [ -n "${M20_STATIC_MAP:-}" ]; then
  echo "ERROR: M20_EKF_TF=1 with M20_STATIC_MAP is forbidden (mapping-pipeline tool leaking into runtime frames)"; exit 1
fi
C=docker-commander-1
DOM="${M20_DOMAIN:-42}"          # M20_DOMAIN: run beside another stack (gazebo_sim_up.sh GZ_ROS_DOMAIN)
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="$HOME/m20_sim/sdk_deploy/src/M20_sdk_deploy/M20_description"
SCENE="${M20_SCENE:-oil_gas_field.xml}"
# M20_EXTERNAL_SIM=1: the world is an external simulator bridged to ROS (tools/godot/godot_sim_up.sh
# starts the Godot sim + tools/godot/godot_bridge.py first). No MuJoCo container, no splat camera and no
# locomotion bridge here: the external bridge publishes the sim topics and takes /cmd_vel itself.
EXTERNAL_SIM="${M20_EXTERNAL_SIM:-0}"
if [ "$EXTERNAL_SIM" = "1" ]; then
  [ "${M20_STANDUP:-0}" != "1" ] || { echo "ERROR: M20_STANDUP needs the MuJoCo sim (joint-level PD); not with M20_EXTERNAL_SIM"; exit 1; }
else
  [ -f "$MODEL/m20_mjcf/mjcf/$SCENE" ] || { echo "ERROR: scene $SCENE not installed next to M20.xml (run tools/setup_sim.sh)"; exit 1; }
fi
if [ "$SCENE" != "oil_gas_field.xml" ]; then
  # the only pre-built map and the default goal both belong to the oil & gas field
  [ -z "${M20_STATIC_MAP:-}" ] || { echo "ERROR: M20_STATIC_MAP is an oil_gas_field map; $SCENE runs live SLAM only"; exit 1; }
  [ "${M20_NO_GOAL:-0}" = "1" ] || [ -n "${GOAL_X:-}" ] || { echo "ERROR: $SCENE needs GOAL_X/GOAL_Y (the default goal is an oil_gas_field point) or M20_NO_GOAL=1"; exit 1; }
fi
GOAL_X="${GOAL_X:--6.0}"; GOAL_Y="${GOAL_Y:--0.5}"
SRC="source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=$DOM RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"

# STALENESS ASSERTION (2026-07-28 lesson, L1.6): the image-baked commander
# predated the tip-over failsafe — an inverted robot stayed armed. The
# commander is PID 1 (not re-copyable like bridge/engine), so we ASSERT the
# installed copy matches the repo and refuse to run stale safety code.
# Override for deliberate old-image tests: M20_ALLOW_STALE=1.
if [ "${M20_ALLOW_STALE:-0}" != "1" ]; then
  REPO_MD5=$(md5sum "$REPO/src/m20_commander/m20_commander/commander_node.py" | cut -d" " -f1)
  INST_PATH=$(docker exec $C bash -c "find /ws -name commander_node.py 2>/dev/null | head -1")
  if [ -z "$INST_PATH" ]; then
    echo "ERROR: no commander_node.py found anywhere in /ws — image layout changed or container broken."
    exit 1
  fi
  INST_MD5=$(docker exec $C md5sum "$INST_PATH" | cut -d" " -f1)
  if [ "$REPO_MD5" != "$INST_MD5" ]; then
    echo "ERROR: installed commander != repo ($INST_PATH=$INST_MD5 repo=$REPO_MD5)."
    echo "       Rebuild the image (docker build + cyclone layer + recreate) or set M20_ALLOW_STALE=1."
    exit 1
  fi
fi

echo "[1/7] staging config + tools into $C:/cfg"
bash "$REPO/tools/dev/container_deps.sh" || exit 1
docker exec $C bash -lc "mkdir -p /cfg /cfg/out"
docker cp "$REPO/src/m20_locomotion_bridge/m20_locomotion_bridge/bridge_node.py" $C:/cfg/bridge_node.py
docker cp "$REPO/src/m20_navigation/config/nav2_params.yaml"                      $C:/cfg/nav2_params.yaml
docker cp "$REPO/tools/slam/mapper_params.yaml"                                   $C:/cfg/mapper_params.yaml
docker cp "$REPO/src/m20_navigation/config/pointcloud_to_laserscan.yaml"          $C:/cfg/pointcloud_to_laserscan.yaml
docker cp "$REPO/tools/nav/nav_logger.py"                                         $C:/cfg/nav_logger.py
docker cp "$REPO/tools/estimator.py"                                              $C:/cfg/estimator.py
docker cp "$REPO/src/m20_missions/m20_missions/mission_server.py"                 $C:/cfg/mission_server.py

if [ -n "${M20_STATIC_MAP:-}" ]; then
  echo "[1.5/7] static-map mode: staging pre-built map + localization config"
  [ -f "$REPO/$M20_STATIC_MAP.posegraph" ] || { echo "ERROR: $M20_STATIC_MAP.posegraph not found (run tools/map/build_map.sh first)"; exit 1; }
  docker exec $C bash -lc "mkdir -p /cfg/maps"
  docker cp "$REPO/$M20_STATIC_MAP.posegraph" $C:/cfg/maps/oil_gas_field.posegraph
  docker cp "$REPO/$M20_STATIC_MAP.data"      $C:/cfg/maps/oil_gas_field.data
  docker cp "$REPO/tools/slam/localization_params.yaml" $C:/cfg/localization_params.yaml
  # Global costmap: rolling window -> full-field static layer from the known map.
  # Patch the STAGED copy only — nav2_params.yaml stays single-source for master.
  docker exec $C python3 - <<'PYEOF'
import yaml
p = "/cfg/nav2_params.yaml"
cfg = yaml.safe_load(open(p))
g = cfg["global_costmap"]["global_costmap"]["ros__parameters"]
g["rolling_window"] = False
g.pop("width", None); g.pop("height", None)
g["track_unknown_space"] = True
g["plugins"] = ["static_layer", "obstacle_layer", "inflation_layer"]
g["static_layer"] = {"plugin": "nav2_costmap_2d::StaticLayer",
                     "map_subscribe_transient_local": True,
                     "map_topic": "/map"}
yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
print("patched global costmap -> static layer (full known field)")
PYEOF
fi

if [ "$SCENE" != "oil_gas_field.xml" ]; then
  # Splat/indoor scenes: faster cruise for exploration tours. STAGED copy only — the
  # oil & gas gates keep nav2_params.yaml as-is. Measured 2026-10-05 (splat demo): 1.0-1.6
  # m/s with 2.0 m/s^2 accel TIPPED the M20 — flooring it out of a tight spot while still
  # turning makes the skid-steer fishtail (uncommanded -1.2 rad/s yaw, body lifts, rolls).
  # Keep the ORIGINAL accel (1.0) and a modest cruise; DWB still turns in place.
  # Goal tolerance back to the live-SLAM value 0.45 (nav2_params.yaml history note:
  # 0.20 assumes known-map localization; live SLAM jitters -> arrive-never-declare).
  # Footprint: the real M20 rectangle (0.86 x 0.56 m incl. wheels) instead of the 0.45 m
  # circle — its corners reach 0.51 m, so the circle let them scrape boxes at speed.
  VMAX="${M20_MAX_VEL:-0.7}"; AMAX="${M20_MAX_ACC:-1.0}"
  echo "[1.6/7] ${SCENE%.xml}: cruise ${VMAX} m/s, accel ${AMAX} / decel 1.5, M20 footprint, keep-away inflation 1.0 m, goal tol 0.45 m (staged params only)"
  # -i is REQUIRED: without it docker exec drops stdin and `python3 -` runs an empty script
  docker exec -i -e VMAX="$VMAX" -e AMAX="$AMAX" $C python3 - <<'PYEOF'
import os, yaml
p = "/cfg/nav2_params.yaml"
v = float(os.environ["VMAX"])
cfg = yaml.safe_load(open(p))
c = cfg["controller_server"]["ros__parameters"]
f = c["FollowPath"]
f["max_vel_x"] = f["max_speed_xy"] = v
acc = float(os.environ["AMAX"])
f["acc_lim_x"], f["decel_lim_x"] = acc, -1.5
f["xy_goal_tolerance"] = c["goal_checker"]["xy_goal_tolerance"] = 0.45
s = cfg["velocity_smoother"]["ros__parameters"]
s["max_velocity"][0] = v
s["max_accel"][0], s["max_decel"][0] = acc, -1.5
fp = "[[0.43, 0.28], [0.43, -0.28], [-0.43, -0.28], [-0.43, 0.28]]"
# Keep-away (2026-10-05): the stock 0.55 m inflation decayed fast (scaling 3.0) and DWB
# barely weighed obstacle cost (BaseObstacle 0.02) -> laps spent 10-18 % within 0.3 m of
# boxes (closest 8 cm). Wider, slower-decaying inflation + a real obstacle weight make
# the planner use aisle centres and the controller steer away from cost.
for k in ("global_costmap", "local_costmap"):
    cm = cfg[k][k]["ros__parameters"]
    cm.pop("robot_radius", None)
    cm["footprint"], cm["footprint_padding"] = fp, 0.10
    cm["inflation_layer"]["inflation_radius"] = 1.0
    cm["inflation_layer"]["cost_scaling_factor"] = 1.5
f["BaseObstacle.scale"] = 0.15
# Anti-dither (2026-10-05): with keep-away costs DWB's best option in tight spots was
# "vx 0, wz +-0.04" — too small for the skid-steer to turn (bridge pivot assist starts at
# 0.15 rad/s), so the robot sat 30-160 s until the progress checker fired. DWB rejects any
# command below BOTH min_speed_xy and min_speed_theta: it must drive or really turn.
f["min_speed_xy"], f["min_speed_theta"] = 0.10, 0.25
# ...and if it still stalls, recover sooner: 40 s dated from slow 15-20 s pivots; a 180 deg
# turn now takes ~5 s (0.7 rad/s), so 15 s still never aborts a legitimate in-place turn.
c["progress_checker"]["movement_time_allowance"] = 15.0
yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
print(f"  patched: max_vel_x={v}, accel {acc}/-1.5, footprint 0.86x0.56 +0.10, inflation 1.0 (scale 1.5), BaseObstacle 0.15, goal tol 0.45")
PYEOF
fi

if [ "$EXTERNAL_SIM" = "1" ]; then
  # External (Godot) sim: its lidars only look UP from ~0.45 m, so low obstacles come from the
  # depth camera (/camera/obstacle_points, tools/godot/godot_bridge.py). They get their OWN
  # obstacle layer: in the lidar's layer, every upward lidar beam projected into /scan would
  # raytrace-clear the low obstacle it passes over (measured: the robot drove into furniture).
  # Here only the camera's own view clears its cells. Heights are base_link z (odom z = 0).
  # M20_SITE_MAP=<repo path without extension> (tools/godot/site_map.py output): the site's known map.
  # map_server serves it (step 4) and the global costmap plans on the WHOLE site from t=0 (static layer)
  # instead of only what the sensors have seen, the main cause of getting boxed in among furniture.
  if [ -n "${M20_SITE_MAP:-}" ]; then
    [ -f "$REPO/$M20_SITE_MAP.yaml" ] || { echo "ERROR: site map $M20_SITE_MAP.yaml not found (run tools/godot/site_map.py)"; exit 1; }
    echo "[1.65/7] external sim: site map $(basename "$M20_SITE_MAP") -> global costmap static layer"
    docker exec $C bash -lc "mkdir -p /cfg/maps && rm -f /cfg/maps/site.*"
    docker cp "$REPO/$M20_SITE_MAP.pgm" $C:/cfg/maps/$(basename "$M20_SITE_MAP").pgm
    docker cp "$REPO/$M20_SITE_MAP.yaml" $C:/cfg/maps/site.yaml
    docker exec -i $C python3 - <<'PYEOF'
import yaml
p = "/cfg/nav2_params.yaml"
cfg = yaml.safe_load(open(p))
g = cfg["global_costmap"]["global_costmap"]["ros__parameters"]
g["rolling_window"] = False
g.pop("width", None); g.pop("height", None)
g["track_unknown_space"] = True
g["plugins"] = ["static_layer", "obstacle_layer", "inflation_layer"]
g["static_layer"] = {"plugin": "nav2_costmap_2d::StaticLayer", "map_subscribe_transient_local": True,
                     "map_topic": "/map"}
yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
print("  patched: global costmap = static site map + lidar + inflation (full site, non-rolling)")
PYEOF
  fi
  echo "[1.7/7] external sim: depth-camera obstacle layer in both costmaps; footprint collision critic"
  docker exec -i $C python3 - <<'PYEOF'
import yaml
p = "/cfg/nav2_params.yaml"
cfg = yaml.safe_load(open(p))
# DWB's BaseObstacle critic scores only the robot's CENTRE cell, so a trajectory whose centre stays below
# inscribed cost can still put a wheel into a wall: the skid-steer M20 cut a corridor corner by 6 cm,
# rode up the wall and tipped over (mapping run, 2026-10-13). ObstacleFootprint checks the whole
# footprint along each trajectory; same weight as BaseObstacle had.
f = cfg["controller_server"]["ros__parameters"]["FollowPath"]
f["critics"] = ["ObstacleFootprint" if c == "BaseObstacle" else c for c in f["critics"]]
f["ObstacleFootprint.scale"] = f.pop("BaseObstacle.scale", 0.15)
for k in ("global_costmap", "local_costmap"):
    cm = cfg[k][k]["ros__parameters"]
    if "camera_layer" not in cm["plugins"]:
        cm["plugins"].insert(cm["plugins"].index("inflation_layer"), "camera_layer")
    cm["camera_layer"] = {
        "plugin": "nav2_costmap_2d::ObstacleLayer", "enabled": True, "observation_sources": "camera",
        "camera": {"topic": "/camera/obstacle_points", "data_type": "PointCloud2",
                   "sensor_frame": "camera_optical_link", "marking": True, "clearing": True,
                   "min_obstacle_height": -0.36, "max_obstacle_height": 0.54,
                   "obstacle_max_range": 4.0, "raytrace_max_range": 4.5, "raytrace_min_range": 0.0,
                   "expected_update_rate": 0.0}}
yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
print("  patched: camera_layer (PointCloud2 /camera/obstacle_points) in global + local costmaps")
PYEOF
fi

# the splat camera replaces the MuJoCo camera (one /camera/image_raw publisher)
SIM_CAMERA="${M20_CAMERA:-0}"; [ "${M20_SPLAT_CAMERA:-0}" = "1" ] && SIM_CAMERA=0
if [ "$EXTERNAL_SIM" = "1" ]; then
  echo "[2/7] external sim: world, sensors and drive come from its ROS bridge (no MuJoCo container)"
else
echo "[2/7] starting MuJoCo sim (${SCENE%.xml}, headless, start=$([ "${M20_STANDUP:-0}" = 1 ] && echo folded || echo stand))"
docker rm -f m20_sim_run 2>/dev/null || true
docker run -d --name m20_sim_run --network host --ipc host \
  -e ROS_DOMAIN_ID=$DOM -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e M20_SIM_GUI=0 -e M20_LIDAR_REALISM=${M20_LIDAR_REALISM:-1} \
  -e M20_START_POSE=${M20_STANDUP:+folded} \
  -e M20_CAMERA=$SIM_CAMERA -e MUJOCO_GL=egl \
  -e M20_NO_ODOM_TF=${M20_EKF_TF:-0} -e M20_UPRIGHT_ASSIST=${M20_UPRIGHT_ASSIST:-0} -e M20_GT_ODOM=${M20_GT_LOC:-0} \
  -e M20_MJCF=/model/m20_mjcf/mjcf/$SCENE \
  -v "$MODEL":/model:ro -v "$REPO/tools/mujoco_sim.py":/mujoco_sim.py:ro \
  m20_sim:latest python3 /mujoco_sim.py
sleep 8
fi

if [ "${M20_SPLAT_CAMERA:-0}" = "1" ] && [ "$EXTERNAL_SIM" != "1" ]; then
  SCENE_YAML="$REPO/tools/sim/${SCENE%.xml}.scene.yaml"
  [ -f "$SCENE_YAML" ] || { echo "ERROR: M20_SPLAT_CAMERA=1 but $SCENE has no scene yaml (not a splat world)"; exit 1; }
  SPLAT_FILE=$(awk '/^  file:/{print $2; exit}' "$SCENE_YAML")
  [ -f "$HOME/m20_sim/splats/$SPLAT_FILE" ] || { echo "ERROR: splat ~/m20_sim/splats/$SPLAT_FILE missing (see RUN_GUIDE 'Splat worlds')"; exit 1; }
  echo "[2.2/7] starting splat camera (gsplat on GPU, $SPLAT_FILE -> /camera/image_raw)"
  docker rm -f m20_splat_cam 2>/dev/null || true
  docker run -d --name m20_splat_cam --gpus all --network host --ipc host \
    -e ROS_DOMAIN_ID=$DOM -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e SPLAT_PLY=/splat/$SPLAT_FILE -e SPLAT_SCENE=/scene/$(basename "$SCENE_YAML") \
    -v "$HOME/m20_sim/splats":/splat:ro -v "$REPO/tools/sim":/scene:ro \
    -v "$REPO/tools/splat/splat_camera_node.py":/splat_camera_node.py:ro \
    m20_splat:latest python3 /splat_camera_node.py
fi

echo "[2.4/7] static TF base_link->lidar_link (commander-local; cross-container"
echo "        transient_local latching proved unreliable after WSL reboots)"
# 0.10 m z-offset = LIDAR_OFFSET in tools/mujoco_sim.py (keep in sync)
docker exec -d $C bash -lc "$SRC; ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 0.10 --frame-id base_link --child-frame-id lidar_link"
sleep 1

echo "[2.5/7] starting pointcloud_to_laserscan (/LIDAR/POINTS -> /scan)"
docker exec -d $C bash -lc "$SRC; ros2 run pointcloud_to_laserscan pointcloud_to_laserscan_node --ros-args -r cloud_in:=/LIDAR/POINTS -r scan:=/scan --params-file /cfg/pointcloud_to_laserscan.yaml"
sleep 2

if [ "${M20_EKF:-0}" = "1" ]; then
  echo "[2.6/7] starting GPS-fused EKF estimator (tf=${M20_EKF_TF:-0} — 1 = GPS-anchored mapping)"
  docker exec -d $C bash -lc "$SRC; export M20_EKF_TF=${M20_EKF_TF:-0}; python3 /cfg/estimator.py >/cfg/out/estimator.log 2>&1"
fi

if [ "${M20_STATION:-1}" = "1" ]; then
  echo "[2.7/7] starting operator station (Foxglove ws://localhost:8765, rosbridge :9090)"
  docker exec -d $C bash -lc "$SRC; ros2 launch m20_station station_bridge.launch.py >/cfg/out/station.log 2>&1"
fi

if [ "${M20_STANDUP:-0}" = "1" ]; then
  echo "[3/7] commander-gated STANDUP (folded -> standing via behavior engine), then bridge"
  # arm in ASSISTED (2); if the commander is stuck in ESTOP, go through IDLE first
  OK=$(docker exec $C bash -lc "$SRC; ros2 service call /m20/set_mode m20_msgs/srv/SetMode '{mode: 2, requester: bringup_standup}'" | grep -c "accepted=True" || true)
  if [ "$OK" = "0" ]; then
    docker exec $C bash -lc "$SRC; ros2 service call /m20/set_mode m20_msgs/srv/SetMode '{mode: 0, requester: bringup_standup}'" | tail -1
    docker exec $C bash -lc "$SRC; ros2 service call /m20/set_mode m20_msgs/srv/SetMode '{mode: 2, requester: bringup_standup}'" | tail -1
  fi
  # behavior engine from the REPO source (same freshness rule as bridge_node.py —
  # the image-baked copy burned us once already); m20_msgs/drdds come from /ws
  docker exec $C bash -lc "rm -rf /cfg/m20_behaviors"
  docker cp "$REPO/src/m20_behaviors/m20_behaviors" $C:/cfg/m20_behaviors
  docker exec -d $C bash -lc "$SRC; PYTHONPATH=/cfg:\$PYTHONPATH python3 -c 'from m20_behaviors.engine import main; main()' >/cfg/out/engine.log 2>&1"
  sleep 3
  # logger starts BEFORE the standup so joints.csv captures the rise
  docker exec -d $C bash -lc "$SRC; RUN_SECS=${RUN_SECS:-300} python3 /cfg/nav_logger.py >/cfg/out/logger.log 2>&1"
  sleep 2
  # request + wait in ONE node (no pub/echo race): subscribe status, wait for the
  # engine to match, publish the request, then wait for a terminal state.
  # STATE_SUCCEEDED=3, ABORTED=4, FAILED=5; 25 s hard stop.
  ST=$(docker exec $C bash -lc "$SRC; timeout 25 python3 - <<'PYEOF'
import rclpy
from m20_msgs.msg import BehaviorRequest, BehaviorStatus
rclpy.init(); n = rclpy.create_node('standup_request')
done = []
n.create_subscription(BehaviorStatus, '/m20/behavior/status',
    lambda m: done.append(m.state) if m.state in (3, 4, 5) else None, 10)
pub = n.create_publisher(BehaviorRequest, '/m20/behavior/request', 10)
while rclpy.ok() and pub.get_subscription_count() == 0:
    rclpy.spin_once(n, timeout_sec=0.2)
req = BehaviorRequest(); req.behavior_name = 'standup'; req.requester = 'bringup'
pub.publish(req)
while rclpy.ok() and not done:
    rclpy.spin_once(n, timeout_sec=0.2)
print(done[0] if done else 'timeout')
PYEOF" | tail -1)
  [ "$ST" = "3" ] || { echo "ERROR: standup did not succeed (state=$ST)"; exit 1; }
  echo "  standup SUCCEEDED — starting bridge (holds stance) + MODE_AUTONOMOUS"
  docker exec -d $C bash -lc "$SRC; python3 /cfg/bridge_node.py --ros-args -p sdk_backend:=sim"
  sleep 4
  docker exec $C bash -lc "$SRC; ros2 service call /m20/set_mode m20_msgs/srv/SetMode '{mode: 3, requester: bringup_standup}'" | tail -2
else
  if [ "$EXTERNAL_SIM" = "1" ]; then
    echo "[3/7] arming (the external sim's bridge takes /cmd_vel; no locomotion bridge)"
  else
    echo "[3/7] starting bridge (sim backend) + arming"
    docker exec -d $C bash -lc "$SRC; python3 /cfg/bridge_node.py --ros-args -p sdk_backend:=sim"
    sleep 4
  fi
  docker exec $C bash -lc "$SRC; ros2 service call /m20/set_mode m20_msgs/srv/SetMode '{mode: 1, requester: nav_test}'" | tail -2
fi

if [ -n "${M20_STATIC_MAP:-}" ]; then
  echo "[4/7] starting slam_toolbox in LOCALIZATION mode (pre-built posegraph)"
  docker exec -d $C bash -lc "$SRC; ros2 run slam_toolbox localization_slam_toolbox_node --ros-args --params-file /cfg/localization_params.yaml >/cfg/out/slam.log 2>&1"
  # SEED the localizer (2026-08-04 leg-1 root cause): map_start_at_dock is
  # SILENTLY UNSUPPORTED in localization mode ("correctly not supported" in
  # slam.log) — slam started UNSEEDED every mission; sometimes it converged
  # from identity, sometimes leg 1 never localized and the robot never moved.
  # Robot spawns at the map origin in sim; hardware will seed from GPS.
  docker exec $C bash -lc "$SRC; timeout 20 python3 - <<'PYEOF'
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
rclpy.init(); n = rclpy.create_node('loc_seed')
pub = n.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
while rclpy.ok() and pub.get_subscription_count() == 0:
    rclpy.spin_once(n, timeout_sec=0.2)
p = PoseWithCovarianceStamped()
p.header.frame_id = 'map'
p.header.stamp = n.get_clock().now().to_msg()
p.pose.pose.orientation.w = 1.0
p.pose.covariance[0] = p.pose.covariance[7] = 0.25
p.pose.covariance[35] = 0.05
pub.publish(p)
import time; time.sleep(0.5)
print('localization seeded at spawn')
PYEOF" | tail -1
  if [ "${M20_GUARDIAN:-1}" = "1" ]; then
    echo "[4.5/7] starting anchor_guardian (mission drift bounding; M20_GUARDIAN=0 disables)"
    docker cp "$REPO/tools/nav/anchor_guardian.py" $C:/cfg/anchor_guardian.py
    docker exec -d $C bash -lc "$SRC; python3 /cfg/anchor_guardian.py >/cfg/out/guardian.log 2>&1"
  fi
elif [ "${M20_LOC:-}" = "amcl" ]; then
  # AMCL on the known site map (external sim only, 2026-10-14). The bridge publishes DRIFTING odometry
  # (GODOT_ODOM=drift); AMCL matches /scan against the site map and publishes map->odom, so there is NO static
  # map->odom here. Initial pose = M20_START_POSE ("x y yaw", the spawn), set through AMCL's own params.
  [ "$EXTERNAL_SIM" = "1" ] && [ -n "${M20_SITE_MAP:-}" ] || { echo "ERROR: M20_LOC=amcl needs M20_EXTERNAL_SIM=1 and M20_SITE_MAP"; exit 1; }
  read -r SX SY SYAW <<< "${M20_START_POSE:-0 0 0}"
  echo "[4/7] AMCL localization on the site map (initial pose $SX $SY $SYAW)"
  docker exec -i -e SX="$SX" -e SY="$SY" -e SYAW="$SYAW" $C python3 - <<'PYEOF'
import os, yaml
p = "/cfg/nav2_params.yaml"
cfg = yaml.safe_load(open(p))
cfg["amcl"] = {"ros__parameters": {
    "use_sim_time": False,
    "global_frame_id": "map", "odom_frame_id": "odom", "base_frame_id": "base_link",
    "scan_topic": "scan", "map_topic": "map", "first_map_only": False,
    # motion model: how much the particles spread per metre / radian of odometry (rot-from-rot, rot-from-trans,
    # trans-from-trans, trans-from-rot). 0.1: the drift model is ~2 % + 0.002 rad/s; Nav2's default 0.2 left the
    # cloud ~0.3 m / 13 deg wide on circuit4 (2026-10-14) and the estimate wandered inside it in the long corridor.
    "robot_model_type": "nav2_amcl::DifferentialMotionModel",
    "alpha1": 0.1, "alpha2": 0.1, "alpha3": 0.1, "alpha4": 0.1, "alpha5": 0.1,
    # sensor model: likelihood field = per beam, how far its endpoint lands from the nearest map obstacle.
    # sigma_hit 0.1: 96 % of scan endpoints land within 0.10 m of the site map at the true pose (measured at the spawn)
    "laser_model_type": "likelihood_field", "max_beams": 120, "laser_min_range": 0.5, "laser_max_range": 12.0,
    "laser_likelihood_max_dist": 2.0, "sigma_hit": 0.1, "z_hit": 0.5, "z_rand": 0.5, "z_short": 0.05, "z_max": 0.05,
    "lambda_short": 0.1, "do_beamskip": False,
    "min_particles": 500, "max_particles": 2000, "pf_err": 0.05, "pf_z": 0.99, "resample_interval": 1,
    "recovery_alpha_fast": 0.0, "recovery_alpha_slow": 0.0,
    "update_min_d": 0.1, "update_min_a": 0.1,                 # filter update every 0.1 m or 0.1 rad of motion
    "tf_broadcast": True, "transform_tolerance": 1.0, "save_pose_rate": 0.5,
    "set_initial_pose": True, "always_reset_initial_pose": False,
    "initial_pose": {"x": float(os.environ["SX"]), "y": float(os.environ["SY"]), "z": 0.0,
                     "yaw": float(os.environ["SYAW"])},
}}
yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
print("  patched: amcl block (likelihood field, 500-2000 particles, initial pose from M20_START_POSE)")
PYEOF
  docker exec -d $C bash -lc "$SRC; ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=/cfg/maps/site.yaml -p use_sim_time:=false >/cfg/out/map_server.log 2>&1"
  docker exec -d $C bash -lc "$SRC; ros2 run nav2_amcl amcl --ros-args --params-file /cfg/nav2_params.yaml >/cfg/out/amcl.log 2>&1"
  docker exec -d $C bash -lc "$SRC; ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -r __node:=lifecycle_manager_localization -p autostart:=true -p node_names:=\"['map_server', 'amcl']\" >/cfg/out/loc_lifecycle.log 2>&1"
  sleep 4
elif [ "${M20_GT_LOC:-0}" = "1" ]; then
  # DEMO-ONLY ground-truth localization: the sim publishes odom = truth (M20_GT_ODOM),
  # map->odom is a fixed identity, and no SLAM runs. Nav2 can't mislocalize; obstacles
  # still come from the live LiDAR (rolling costmaps). NOT for gates / evidence runs.
  echo "[4/7] GROUND-TRUTH localization (demo only): static map->odom, no slam_toolbox"
  docker exec -d $C bash -lc "$SRC; ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 0 --frame-id map --child-frame-id odom"
  if [ -n "${M20_SITE_MAP:-}" ] && [ "$EXTERNAL_SIM" = "1" ]; then
    echo "[4.1/7] map_server: site map on /map (transient local) + its lifecycle manager"
    docker exec -d $C bash -lc "$SRC; ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=/cfg/maps/site.yaml -p use_sim_time:=false >/cfg/out/map_server.log 2>&1"
    docker exec -d $C bash -lc "$SRC; ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -r __node:=lifecycle_manager_map -p autostart:=true -p node_names:=\"['map_server']\" >/cfg/out/map_lifecycle.log 2>&1"
    sleep 3
  fi
else
  echo "[4/7] starting slam_toolbox (mapping)"
  docker exec -d $C bash -lc "$SRC; ros2 run slam_toolbox async_slam_toolbox_node --ros-args --params-file /cfg/mapper_params.yaml >/cfg/out/slam.log 2>&1"
fi
sleep 6

# Keep-away / escape knobs (unset = unchanged), for the boxed-in investigation (2026-10-19: the controller grazed a
# 0.3 m floor pipe to within 7 cm, every trajectory and the spin/back-up recoveries then collided; nav2.log "No valid
# trajectories out of 231"):  M20_OBSTACLE_SCALE  ObstacleFootprint critic weight (0.15 staged; higher = keep away)
#   M20_MIN_VEL_X  DWB min_vel_x (0 staged; negative lets the controller reverse out of a corner)
#   M20_CAM_MIN_Z  camera_layer min_obstacle_height (-0.36 staged; the floor sits at -0.43: -0.30 marks less floor noise)
# M20_OBSTACLE_SCALE only matters on the external-sim path (ObstacleFootprint); the MuJoCo path still scores
# BaseObstacle, where the key is written but unused. M20_CAM_MIN_Z needs the camera_layer (external sim) and fails
# loudly without it.
if [ -n "${M20_OBSTACLE_SCALE:-}${M20_MIN_VEL_X:-}${M20_CAM_MIN_Z:-}" ]; then
  echo "[4.7/7] keep-away knobs: obstacle scale ${M20_OBSTACLE_SCALE:-unchanged}, min_vel_x ${M20_MIN_VEL_X:-unchanged}, camera min z ${M20_CAM_MIN_Z:-unchanged}"
  docker exec -i -e OS="${M20_OBSTACLE_SCALE:-}" -e MV="${M20_MIN_VEL_X:-}" -e CZ="${M20_CAM_MIN_Z:-}" $C python3 - <<'PYEOF'
import os, yaml
p = "/cfg/nav2_params.yaml"
cfg = yaml.safe_load(open(p))
f = cfg["controller_server"]["ros__parameters"]["FollowPath"]
if os.environ["OS"]:
    f["ObstacleFootprint.scale"] = float(os.environ["OS"])
if os.environ["MV"]:
    f["min_vel_x"] = float(os.environ["MV"])
if os.environ["CZ"]:
    for k in ("global_costmap", "local_costmap"):
        cfg[k][k]["ros__parameters"]["camera_layer"]["camera"]["min_obstacle_height"] = float(os.environ["CZ"])
yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
print("  patched:", {k: v for k, v in (("ObstacleFootprint.scale", os.environ["OS"]), ("min_vel_x", os.environ["MV"]), ("camera min z", os.environ["CZ"])) if v})
PYEOF
fi

# Exploration-map hook (M20_EXPLORE_MAP=1; unset = unchanged): the global costmap becomes the robot's own memory of an
# UNSEEN site, VLFM-style: non-rolling, fixed to the site's extent, starting all-unknown and filling in from the global
# costmap's own layers (obstacle_layer on /scan, camera_layer on the depth cloud) as the robot moves (track_unknown_space).
# Frontiers = known-free cells next to unknown ones. The extent (M20_EXPLORE_ORIGIN "x y", M20_EXPLORE_SIZE "w h",
# metres) must cover wherever the robot may go, INCLUDING odometry drift: the defaults are circuit4's (28 x 36 m with
# ~4 m of margin); another site needs its own.
if [ "${M20_EXPLORE_MAP:-0}" = "1" ]; then
  echo "[4.8/7] exploration map: global costmap non-rolling, all-unknown at start, origin ${M20_EXPLORE_ORIGIN:--12 -16} size ${M20_EXPLORE_SIZE:-28 36} m"
  # M20_EXPLORE_CLEAR=camera (unset = unchanged: both layers clear): only the depth camera raytraces unknown space
  # away, the lidar only MARKS. VLFM's "explored" area is what the depth camera saw; a 360 deg lidar that clears
  # through every doorway declares rooms explored that the camera never looked into (Gazebo indoor_rooms, 2026-10-26:
  # 5.6 % unknown left and no frontier while the target room was unvisited -> "not_found").
  docker exec -i -e ORIGIN="${M20_EXPLORE_ORIGIN:--12 -16}" -e SIZE="${M20_EXPLORE_SIZE:-28 36}" -e CLEAR="${M20_EXPLORE_CLEAR:-}" $C python3 - <<'PYEOF'
import os, yaml
p = "/cfg/nav2_params.yaml"
cfg = yaml.safe_load(open(p))
g = cfg["global_costmap"]["global_costmap"]["ros__parameters"]
ox, oy = (float(v) for v in os.environ["ORIGIN"].split()); w, h = (float(v) for v in os.environ["SIZE"].split())
g["rolling_window"] = False
g["width"], g["height"], g["origin_x"], g["origin_y"] = int(w), int(h), ox, oy
g["resolution"] = 0.1
g["track_unknown_space"] = True
g["unknown_cost_value"] = 255
g["plugins"] = [pl for pl in g["plugins"] if pl != "static_layer"]
note = ""
if os.environ.get("CLEAR") == "camera":
    g["obstacle_layer"]["scan"]["clearing"] = False
    note = "; lidar marks only, the camera clears"
yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
print(f"  patched: global costmap {w:.0f}x{h:.0f} m at ({ox}, {oy}), 0.1 m cells, unknown tracked, no static layer{note}")
PYEOF
fi

# Real-time B hook (tools/realtime_b/realtime_b_up.sh sets it; unset = unchanged): extra Nav2 behavior plugins, e.g.
# M20_BEHAVIORS_EXTRA=drive_on_heading (the VLM's forward moves; the stock list is spin/backup/wait).
if [ -n "${M20_BEHAVIORS_EXTRA:-}" ]; then
  echo "[4.9/7] extra Nav2 behaviors: $M20_BEHAVIORS_EXTRA"
  docker exec -i -e EXTRA="$M20_BEHAVIORS_EXTRA" $C python3 - <<'PYEOF'
import os, yaml
p = "/cfg/nav2_params.yaml"
cfg = yaml.safe_load(open(p))
b = cfg["behavior_server"]["ros__parameters"]
for name in os.environ["EXTRA"].split(","):
    name = name.strip()
    if name and name not in b["behavior_plugins"]:
        b["behavior_plugins"].append(name)
        b[name] = {"plugin": "nav2_behaviors/" + "".join(w.capitalize() for w in name.split("_"))}
yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
print("  patched: behavior_plugins =", b["behavior_plugins"])
PYEOF
fi

echo "[5/7] starting Nav2 (logs -> /cfg/out/nav2.log; the 2026-07-17 lesson)"
docker exec -d $C bash -lc "$SRC; ros2 launch nav2_bringup navigation_launch.py params_file:=/cfg/nav2_params.yaml use_sim_time:=false >/cfg/out/nav2.log 2>&1"
sleep 14

# Real-time B hook: collision_monitor between /cmd_vel and the robot (M20_COLLISION_MONITOR=1; the external-sim bridge
# then reads /cmd_vel_safe via GODOT_CMD_TOPIC). One "approach" zone = the robot's own footprint simulated 1.2 s ahead
# at the commanded velocity: any motion that would reach an obstacle sooner is slowed, in every direction, while
# turning away stays possible (a fixed stop box around the body would also block turning away from a wall).
if [ "${M20_COLLISION_MONITOR:-0}" = "1" ]; then
  echo "[5.1/7] starting collision_monitor (/cmd_vel -> /cmd_vel_safe, footprint approach zone 1.2 s)"
  docker exec -i $C bash -lc "cat > /cfg/collision_monitor.yaml" <<'YEOF'
collision_monitor:
  ros__parameters:
    base_frame_id: base_link
    odom_frame_id: odom
    cmd_vel_in_topic: cmd_vel
    cmd_vel_out_topic: cmd_vel_safe
    transform_tolerance: 0.5
    source_timeout: 5.0
    stop_pub_timeout: 2.0
    polygons: ["FootprintApproach"]
    FootprintApproach:
      type: polygon
      action_type: approach
      footprint_topic: /local_costmap/published_footprint
      time_before_collision: 1.2
      simulation_time_step: 0.1
      max_points: 3
      visualize: false
    observation_sources: ["scan", "camera"]
    scan:
      type: scan
      topic: /scan
    camera:
      type: pointcloud
      topic: /camera/obstacle_points
      min_height: -0.36
      max_height: 0.54
YEOF
  docker exec -d $C bash -lc "$SRC; ros2 run nav2_collision_monitor collision_monitor --ros-args --params-file /cfg/collision_monitor.yaml >/cfg/out/collision_monitor.log 2>&1"
  docker exec -d $C bash -lc "$SRC; ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -r __node:=lifecycle_manager_collision -p autostart:=true -p node_names:=\"['collision_monitor']\" >/cfg/out/collision_lifecycle.log 2>&1"
  # it must really be active: with GODOT_CMD_TOPIC=/cmd_vel_safe and no monitor publishing it, the robot never moves
  # (its lifecycle manager's log, not `ros2 lifecycle get`: the ros2 CLI's discovery can take > 5 s on this box)
  CM_OK=0
  for i in $(seq 1 20); do
    docker exec $C grep -q "Managed nodes are active" /cfg/out/collision_lifecycle.log 2>/dev/null && { CM_OK=1; break; }
    sleep 1
  done
  [ "$CM_OK" = "1" ] || { echo "ERROR: collision_monitor did not become active (see /cfg/out/collision_monitor.log)"; docker exec $C tail -20 /cfg/out/collision_monitor.log; exit 1; }
  echo "  collision_monitor active"
fi

echo "[5.5/7] starting mission server (/m20/mission/run — tools/dev/m20 goto|mission)"
docker exec -d $C bash -lc "$SRC; python3 /cfg/mission_server.py >/cfg/out/mission.log 2>&1"

# NL navigation (src/m20_nlnav): text -> goals -> RunMission. Hot-loaded from the REPO like the
# behavior engine (python changes need no image rebuild; the NavigateNL action type does).
# Named waypoints come from the scene yaml when the scene has one. Default on; M20_NLNAV=0 = off.
if [ "${M20_NLNAV:-1}" = "1" ]; then
  echo "[5.6/7] starting NL navigation (/m20/nlnav/navigate — tools/dev/m20 say \"...\")"
  docker exec $C bash -lc "rm -rf /cfg/m20_nlnav /cfg/scene.yaml"
  docker cp "$REPO/src/m20_nlnav/m20_nlnav" $C:/cfg/m20_nlnav
  NL_SCENE=""
  if [ -f "$REPO/tools/sim/${SCENE%.xml}.scene.yaml" ]; then
    docker cp "$REPO/tools/sim/${SCENE%.xml}.scene.yaml" $C:/cfg/scene.yaml
    NL_SCENE="-p scene_yaml:=/cfg/scene.yaml"
  fi
  docker exec -d $C bash -lc "$SRC; PYTHONPATH=/cfg:\$PYTHONPATH python3 -c 'from m20_nlnav.nlnav_node import main; main()' --ros-args $NL_SCENE >/cfg/out/nlnav.log 2>&1"
fi

if [ "${M20_STANDUP:-0}" != "1" ]; then
  echo "[6a/7] starting logger (standup path started it earlier)"
  docker exec -d $C bash -lc "$SRC; RUN_SECS=${RUN_SECS:-300} python3 /cfg/nav_logger.py >/cfg/out/logger.log 2>&1"
  sleep 2
fi
if [ "${M20_NO_GOAL:-0}" = "1" ]; then
  echo "[6/7] M20_NO_GOAL=1 — stack up, no goal sent (mapping/coverage run)"
else
  echo "[6/7] sending goal ($GOAL_X, $GOAL_Y)"
  docker exec $C bash -lc "$SRC; ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
    '{pose: {header: {frame_id: map}, pose: {position: {x: $GOAL_X, y: $GOAL_Y}, orientation: {z: 1.0, w: 0.0}}}}'" | tail -3
fi

echo "[7/7] run complete. Copy results + render with:"
echo "  docker cp $C:/cfg/out tools/nav/out && OUTDIR=tools/nav/out python3 tools/nav/render_nav.py"
