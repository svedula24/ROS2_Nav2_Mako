#!/usr/bin/env bash
# Gazebo Harmonic + the M20 stack (gazebo-sim branch): the standard-simulator counterpart of tools/godot/godot_sim_up.sh.
#   bash tools/gazebo/gazebo_sim_up.sh [--world indoor_rooms] [--gui] [--sim-only] [--loc gt|amcl|odom] [--odom truth|drift]
#     --world <name>   tools/gazebo/worlds/<name>.sdf (default indoor_rooms)
#     --gui            also open the Gazebo GUI (WSLg window); default headless server + sensors only
#     --sim-only       Gazebo + ros_gz bridge + the glue node, no Nav2/commander (topic checks)
#     --loc            localization for bringup: gt (default here: no site map yet) | amcl | odom
#     --odom           what /odom carries: truth (default) | drift (GZ_ODOM)
# Containers: m20_gazebo (gz sim + parameter_bridge, host network), m20_gazebo_bridge (glue, m20_sim image),
# then tools/nav/bringup_plan_a.sh with M20_EXTERNAL_SIM=1 (no MuJoCo, no splat camera).
# Rendering: GPU through WSLg (Mesa d3d12: /usr/lib/wsl, /mnt/wslg mounted); GZ_SOFTWARE_GL=1 forces llvmpipe.
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WORLD="indoor_rooms"; GUI=0; SIM_ONLY=0; LOC="${M20_LOC:-gt}"; ODOM="${GZ_ODOM:-truth}"
while [ $# -gt 0 ]; do
  case "$1" in
    --world) WORLD="$2"; shift ;;
    --gui) GUI=1 ;;
    --sim-only) SIM_ONLY=1 ;;
    --loc) LOC="$2"; shift ;;
    --odom) ODOM="$2"; shift ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac; shift
done
[ -f "$REPO/tools/gazebo/worlds/$WORLD.sdf" ] || { echo "ERROR: no world tools/gazebo/worlds/$WORLD.sdf"; exit 1; }
docker image inspect m20_gazebo:latest >/dev/null 2>&1 || { echo "ERROR: build the image first: docker build -f docker/Dockerfile.gazebo -t m20_gazebo:latest docker"; exit 1; }

ROS_ENV="-e ROS_DOMAIN_ID=${GZ_ROS_DOMAIN:-42} -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e PYTHONUNBUFFERED=1"   # GZ_ROS_DOMAIN: isolate from a running stack
# Mesa's d3d12 driver defaults to the FIRST adapter (the Intel iGPU, measured 2026-10-26); name the discrete GPU.
GL_ENV="-e DISPLAY=${DISPLAY:-:0} -e WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0} -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir -e PULSE_SERVER=/mnt/wslg/PulseServer -e MESA_D3D12_DEFAULT_ADAPTER_NAME=${GZ_GPU_ADAPTER:-NVIDIA}"
[ "${GZ_SOFTWARE_GL:-0}" = "1" ] && GL_ENV="$GL_ENV -e LIBGL_ALWAYS_SOFTWARE=1 -e GALLIUM_DRIVER=llvmpipe"
mkdir -p "$HOME/m20_sim/gz_fuel"        # Fuel model cache (props download once)
MOUNTS="-v /usr/lib/wsl:/usr/lib/wsl:ro -v /mnt/wslg:/mnt/wslg -v /tmp/.X11-unix:/tmp/.X11-unix -v $REPO/tools/gazebo:/sim:ro -v $HOME/m20_sim/gz_fuel:/root/.gz/fuel"

# 0. a clean stack: Nav2 nodes left over from a previous run keep FROZEN TF buffers (the planner started every path
# at the Godot robot's last parked pose, measured 2026-10-26), and bringup only adds a second copy of each node.
if [ "$SIM_ONLY" != 1 ]; then
  echo "[gazebo_sim_up] tearing down the previous stack (tools/nav/kill_stack.sh)"
  bash "$REPO/tools/nav/kill_stack.sh" 2>&1 | tail -2
fi
# SIGTERM the glue first (3 s): on exit it publishes zero velocity, so a robot mid-move stops instead of
# coasting on its last /cmd_vel. With --sim-only kill_stack.sh above is skipped, so this is the only stop.
docker stop -t 3 m20_gazebo_bridge >/dev/null 2>&1 || true
docker rm -f m20_gazebo m20_gazebo_bridge >/dev/null 2>&1 || true

# 1. Gazebo: server (physics + sensors) and the ros_gz parameter bridge in one container
echo "[gazebo_sim_up] starting Gazebo ($WORLD, headless$( [ $GUI = 1 ] && echo " + GUI"))"
GZ_ARGS="-r -s --headless-rendering"; [ "$GUI" = 1 ] && GZ_ARGS="-r"
docker run -d --name m20_gazebo --gpus all --network host --ipc host $ROS_ENV $GL_ENV $MOUNTS \
  -e GZ_SIM_RESOURCE_PATH=/sim/models -e GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/humble/lib \
  m20_gazebo:latest "source /opt/ros/humble/setup.bash; \
    gz sim $GZ_ARGS /sim/worlds/$WORLD.sdf > /tmp/gz.log 2>&1 & \
    sleep 8; ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=/sim/ros_gz_bridge.yaml > /tmp/bridge.log 2>&1 & \
    wait" >/dev/null
for i in $(seq 1 40); do
  docker exec m20_gazebo bash -lc "gz topic -l 2>/dev/null | grep -q '/m20/front_lidar/points'" && break; sleep 2
done
docker exec m20_gazebo bash -lc "gz topic -l 2>/dev/null | grep -q '/m20/front_lidar/points'" \
  || { echo "ERROR: Gazebo's lidar topic never appeared; docker exec m20_gazebo cat /tmp/gz.log"; exit 1; }
echo "[gazebo_sim_up] Gazebo up: $(docker exec m20_gazebo bash -lc 'gz topic -l | grep -c /m20/') /m20 topics"

# 2. the glue: /gz/* -> the stack's topics and frames (m20_sim image: has m20_msgs)
echo "[gazebo_sim_up] starting the Gazebo ROS glue (odom=$ODOM)"
docker stop -t 3 m20_gazebo_bridge >/dev/null 2>&1 || true   # same reason as above: zero velocity before removal
docker rm -f m20_gazebo_bridge >/dev/null 2>&1 || true
docker run -d --name m20_gazebo_bridge --network host --ipc host $ROS_ENV \
  -e GZ_ODOM="$ODOM" -e GZ_VMAX="${GZ_VMAX:-0.5}" -e GZ_DEPTH_OBSTACLES="${GZ_DEPTH_OBSTACLES:-1}" \
  -e GZ_CMD_TOPIC="${GZ_CMD_TOPIC:-/cmd_vel}" \
  -v "$REPO/tools/gazebo":/gazebo:ro -v "$REPO/tools/godot":/godot:ro \
  m20_sim:latest bash -lc "source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; python3 /gazebo/gazebo_bridge.py" >/dev/null
sleep 5
if [ "$SIM_ONLY" = 1 ]; then
  echo "[gazebo_sim_up] --sim-only: Gazebo + bridges up. Check: docker exec docker-commander-1 bash -lc 'source /opt/ros/humble/setup.bash; ros2 topic hz /LIDAR/POINTS'"
  exit 0
fi

# 3. the usual stack minus the MuJoCo pieces; no site map yet, so ground-truth localization by default
case "$LOC" in gt) GT_LOC=1 ;; *) GT_LOC=0 ;; esac
echo "[gazebo_sim_up] bringup: localization $LOC (odometry $ODOM)"
M20_EXTERNAL_SIM=1 M20_SCENE="gazebo_$WORLD.xml" M20_GT_LOC=$GT_LOC M20_LOC=$LOC M20_DOMAIN="${GZ_ROS_DOMAIN:-42}" \
M20_NO_GOAL=1 M20_SPLAT_CAMERA=0 M20_NLNAV=0 M20_SITE_MAP="${M20_SITE_MAP:-}" M20_MAX_VEL="${GZ_VMAX:-0.5}" \
bash "$REPO/tools/nav/bringup_plan_a.sh"
SRC="source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=${GZ_ROS_DOMAIN:-42} RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
arm() { docker exec docker-commander-1 bash -lc "$SRC; timeout 15 ros2 service call /m20/set_mode m20_msgs/srv/SetMode '{mode: $1, requester: gazebo_sim_up}'" | grep -c "accepted=True" || true; }
ARMED=$(arm 2)
if [ "$ARMED" != "1" ]; then arm 0 >/dev/null; sleep 1; ARMED=$(arm 2); fi
if [ "$ARMED" != "1" ]; then
  echo "[gazebo_sim_up] ERROR: the commander refused ASSISTED (check /m20/failsafe; docker logs docker-commander-1)"; exit 1
fi
echo "[gazebo_sim_up] armed ASSISTED. World $WORLD, Gazebo in m20_gazebo (logs: docker exec m20_gazebo cat /tmp/gz.log)"
