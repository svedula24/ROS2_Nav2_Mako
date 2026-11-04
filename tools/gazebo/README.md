# Gazebo Harmonic for the M20 stack (branch `gazebo-sim`)

The standard-simulator counterpart of `tools/godot`: Gazebo (gz-sim 8, Harmonic) runs in WSL next to the ROS 2
containers, publishes the same topics and frames the Godot bridge did, and the rest of the stack (commander, Nav2,
mission server, nlnav, the VLFM search, the detector, the eval harness) runs unchanged. Isaac Sim stays the platform for
policy training (rl_training / m20-vla, native Windows).

## Pieces

| File | What |
|---|---|
| `docker/Dockerfile.gazebo` | `m20_gazebo:latest`: `ros:humble-ros-base` + `gz-harmonic` + `ros-humble-ros-gzharmonic` + Cyclone DDS + Mesa |
| `make_m20_sdf.py` | generates `models/m20/model.sdf` from the DeepRobotics URDF (`M20_source.urdf`): one rigid body (box collision, real meshes posed at the standing stance), two hemispherical GPU lidars, an RGB-D camera, an IMU, `VelocityControl` (kinematic planar drive from a Twist), `PosePublisher` (true pose), `OdometryPublisher` (body twist) |
| `worlds/indoor_rooms.sdf` | 16 x 12 m, three rooms with doorways, a low crate (lidar-blind), a cabinet, a pillar, a red box |
| `ros_gz_bridge.yaml` | `parameter_bridge` config: Gazebo topics <-> `/gz/*` ROS topics |
| `gazebo_bridge.py` | the glue (runs in the `m20_sim` image: needs `drdds`/`m20_msgs`): `/gz/*` -> `/odom` + TF, `/odom_true`, `/IMU_DATA`, `/JOINTS_DATA`, `/LIDAR/POINTS` (both lidars merged, z-band), `/camera/{image_raw,depth,camera_info,obstacle_points}`, `/m20/health`; `/cmd_vel` (armed-gated, capped) -> `/gz/cmd_vel`. Drift odometry via `godot_io.DriftOdom` (`GZ_ODOM=drift`) |
| `gazebo_sim_up.sh` | launcher: Gazebo + bridge container, the glue, then `tools/nav/bringup_plan_a.sh` with `M20_EXTERNAL_SIM=1` |

Sensor geometry matches `tools/godot` (`vehicle.json`): camera 640x480, 90 deg hfov, mount (0.376, 0, 0.037);
lidars 360 x 0..90 deg looking up from (+-0.320, 0, -0.013), 32 x 450 beams in the sim (the real Airy: 96 x 900);
base_link 0.43 m above the floor; footprint 0.86 x 0.56 m.

## Run

```bash
docker build -f docker/Dockerfile.gazebo -t m20_gazebo:latest docker        # once, ~4 GB
bash tools/gazebo/gazebo_sim_up.sh --sim-only                               # Gazebo + bridges only (topic checks)
bash tools/gazebo/gazebo_sim_up.sh                                          # + commander / Nav2 (gt localization)
bash tools/gazebo/gazebo_sim_up.sh --gui --world indoor_rooms               # with the Gazebo window (WSLg)
```

Knobs: `GZ_ODOM=truth|drift`, `GZ_VMAX` (0.5), `GZ_CMD_TOPIC` (`/cmd_vel`; Real-time B uses `/cmd_vel_safe`),
`GZ_ROS_DOMAIN` (42; use another to run next to a live stack), `GZ_GPU_ADAPTER` (NVIDIA), `GZ_SOFTWARE_GL=1`.

## Measured (2026-10-26, headless, RTX 5090 through WSLg's Mesa d3d12)

- Gazebo real-time factor 1.00 with both lidars + RGB-D at 10 Hz; `/odom` 50 Hz; lidar cloud ~8.7 k points in the
  z-band; depth 32FC1 with 0 = no return (Godot contract); camera fx 320.0 (90 deg at 640 px).
- Kinematic drive through the glue (armed, `/cmd_vel` at 10 Hz): 0.3 m/s for 4 s moved the true pose 1.18 m
  (expected 1.2); 0.5 rad/s for 4 s turned 1.97 rad (expected 2.0). No yaw trim needed (the Godot M20 under-turned ~50 %).

- Full stack (gt localization, no site map): `m20 goto 2.5 3.5` from the spawn (-4, 0) through two doorways into
  room B: mission success, final true pose (2.28, 3.17) (goal tolerance 0.45 m).

- Three missions in a row on one stack, `m20 goto`: room B (2.5, 3.5) in 18 s, room C (6, -4.5) in 23 s, home
  (-4, 0) in 32 s; final poses (2.27, 3.17), (5.72, -4.78), (-3.60, 0.05).

## Failures on the way (each measured, each fixed)

1. **The lidars saw the robot itself**: 4,992 returns under 0.3 m (the sensors sit inside the 0.86 m body), the
   footprint went lethal, "No valid trajectories out of 231". Fix: robot visuals carry visibility flag 0x2, the lidar
   and camera masks keep 0x1 (`make_m20_sdf.py`). After: nearest return 3.7 m, footprint cost 0.
2. **Leftover Nav2 nodes with frozen TF buffers**: every plan started at the previous robot's last pose (-6.0, 4.8);
   a second bringup only added duplicate action servers ("unknown goal response"). Fix: the launcher runs
   `tools/nav/kill_stack.sh` first.
3. **World geometry**: the x = 0 doorway opened onto the end of the y = 0 wall, two 0.6 m gaps for a 0.56 m robot;
   the scan straight east read 4.0 m where the doorway should give 8 m. Fix: a 1.2 m lobby, 1.8 m doorway.
4. **The planner's start pose froze after the first mission** (every plan started at (2.3, 3.2), the room-B goal,
   while the robot stood in room C; the controller then got "plan with zero length"). The planner's TF lookups went
   stale while the costmap's sensor marks kept updating: the tf2 MessageFilter wait + timeout-timer race the Godot
   bridge had already met. Fix: the glue stamps sensor messages with a TF stamp at least 100 ms OLD (`GZ_SENSOR_LAG`),
   so the transform is always already in every buffer and no filter ever waits. Three missions in a row after.

## Real-time B search on Gazebo (2026-10-26, first run)

`RTB_SIM=gazebo bash tools/realtime_b/realtime_b_up.sh --mode search`, then `m20 search "refrigerator"` (the fridge is
in room B, out of view). Result: `not_found` after 45 s, 15 frontier goals, 81 views scored, 12.4 m, with the robot in
room C and the fridge never in the camera's view. Two causes, both measured on the live exploration map:

- **The 360 deg lidar "explores" rooms the camera never saw.** Through the doorways the scan raytraces big wedges of
  room B (pillar, ladder, fridge all marked), so only 5.6 % of the map stayed unknown and the remaining frontier
  segments were 0.3 and 0.1 m (below the 0.5 m minimum). VLFM's explored area is the depth camera's; ours was the
  lidar's. Fix: in the exploration map only the camera clears unknown space (`M20_EXPLORE_CLEAR=camera`).
- **Goal thrashing** (a new frontier nearly every tick, 15 goals in 45 s): the commit / blacklist / watchdog fixes
  live in the realtime-b worktree's UNCOMMITTED follow-up and are not on this branch yet.

The detector did well on the Fuel props from the first pass: traffic cone 0.83, trash can 0.90, desk 0.69,
office chair 0.64, the low crate as "cardboard box" 0.56.

## Gotchas

- Mesa's d3d12 picks the FIRST adapter (the Intel iGPU): `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` is set by the launcher.
- `OdometryPublisher` also publishes a Pose_V on `/model/<name>/pose` by default, the same topic as `PosePublisher`'s
  true pose: the generator moves it to `/m20/odom_tf`.
- Run WSL docker commands from script files, not inline strings: Git Bash path conversion silently empties
  `-v` mounts (Docker then creates an empty directory on the host).
- The stack's topics are shared: never run this next to the Godot bridge on the same domain.
