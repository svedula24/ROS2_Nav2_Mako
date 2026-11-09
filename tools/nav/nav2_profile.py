#!/usr/bin/env python3
"""The one place the Nav2 parameters get adapted to the M20, for the sim AND the real robot.

Both launchers (tools/nav/bringup_plan_a.sh here, tools/robot/robot_up.sh in m20-sim-to-real) stage a copy of
src/m20_navigation/config/nav2_params.yaml and run this file on it; this file is byte-identical in both repos (a CI
check compares them). Every number has a measured reason, written next to it.

  python3 nav2_profile.py /cfg/nav2_params.yaml --profile sim         # the Godot sim as driven since 2026-10-05
  python3 nav2_profile.py /cfg/nav2_params.yaml --profile robot_shape # the real M20 shape (also the sim rehearsal)
      [--max-vel 0.3] [--max-acc 1.0]
      [--static-map] [--scans scan_front,scan_rear] [--camera-layer] [--footprint-critic]
      [--amcl "x y yaw" --amcl-scan scan_front --amcl-sigma-hit 0.2]
      [--obstacle-scale 1.0] [--min-vel-x 0] [--cam-min-z -0.30]
Prints "profile <name> v<PROFILE_VERSION>: <what changed>" so every run log says which settings drove.

Profiles:
  sim          what [1.6/7] staged for the Godot sim: cruise 0.5-0.7 m/s, footprint 0.86 x 0.56 m + 0.10 padding,
               keep-away inflation 1.0 m (scale 1.5), obstacle weight 0.15, min speeds 0.10 / 0.25, goal tol 0.45.
  robot_shape  the real Lynx M20 Pro: body 0.82 x 0.506 m with leg margin -> 0.90 x 0.60 m footprint + 0.05 padding;
               speed cap 0.3 m/s for first drives; min speeds 0.15 m/s and 0.35 rad/s above the robot's dead zones
               (vy 0.16 m/s, wz 0.15 rad/s measured 2026-09-10); everything else as the sim profile.
"""
import argparse
import sys

import yaml

PROFILE_VERSION = 1

# --- the measured numbers, named once
SIM_FOOTPRINT = [[0.43, 0.28], [0.43, -0.28], [-0.43, -0.28], [-0.43, 0.28]]    # 0.86 x 0.56 m incl. wheels
ROBOT_FOOTPRINT = [[0.45, 0.30], [0.45, -0.30], [-0.45, -0.30], [-0.45, 0.30]]  # 0.82 x 0.506 m body + leg margin
INFLATION_RADIUS_M = 1.0        # keep-away (2026-10-05): the stock 0.55 m at scale 3.0 let laps spend 10-18 % within 0.3 m of boxes
INFLATION_SCALE = 1.5
OBSTACLE_WEIGHT = 0.15          # BaseObstacle 0.02 barely counted; 0.15 makes the controller steer away from cost
GOAL_TOL_M = 0.45               # live-localization value: 0.20 assumes a known-map pose and never declared arrival under jitter
PROGRESS_ALLOWANCE_S = 15.0     # a 180 deg turn takes ~5 s at 0.7 rad/s; 40 s only hid dithering
SIM_MIN_SPEEDS = (0.10, 0.25)   # anti-dither: DWB's "vx 0, wz 0.04" sat 30-160 s; the sim bridge's pivot assist starts at 0.15 rad/s
ROBOT_MIN_SPEEDS = (0.15, 0.35)  # the robot ignores vy < 0.16 m/s and wz < 0.15 rad/s; 0.35 rad/s turns for sure (private repo's numbers)


def _yaml_list(v):
    """Nav2 reads footprint as the STRING "[[x, y], ...]"."""
    return "[" + ", ".join(f"[{x}, {y}]" for x, y in v) + "]"


def apply_vehicle(cfg, vmax, amax, footprint, padding, min_speeds):
    """Speeds, footprint, keep-away costs and the anti-dither minimums, in both costmaps and the controller."""
    c = cfg["controller_server"]["ros__parameters"]
    f = c["FollowPath"]
    f["max_vel_x"] = f["max_speed_xy"] = vmax
    f["acc_lim_x"], f["decel_lim_x"] = amax, -1.5
    f["xy_goal_tolerance"] = c["goal_checker"]["xy_goal_tolerance"] = GOAL_TOL_M
    f["min_speed_xy"], f["min_speed_theta"] = min_speeds
    f["BaseObstacle.scale"] = OBSTACLE_WEIGHT
    c["progress_checker"]["movement_time_allowance"] = PROGRESS_ALLOWANCE_S
    s = cfg["velocity_smoother"]["ros__parameters"]
    s["max_velocity"][0] = vmax
    s["max_accel"][0], s["max_decel"][0] = amax, -1.5
    for k in ("global_costmap", "local_costmap"):
        cm = cfg[k][k]["ros__parameters"]
        cm.pop("robot_radius", None)
        cm["footprint"], cm["footprint_padding"] = _yaml_list(footprint), padding
        cm["inflation_layer"]["inflation_radius"] = INFLATION_RADIUS_M
        cm["inflation_layer"]["cost_scaling_factor"] = INFLATION_SCALE
    return f"vmax {vmax} m/s, accel {amax}/-1.5, footprint {_yaml_list(footprint)} +{padding}, min speeds {min_speeds}"


def apply_footprint_critic(cfg):
    """DWB's BaseObstacle scores only the robot's CENTRE cell: the M20 cut a corner by 6 cm, rode up the wall and tipped
    (2026-10-13). ObstacleFootprint checks the whole footprint along each trajectory, same weight."""
    f = cfg["controller_server"]["ros__parameters"]["FollowPath"]
    f["critics"] = ["ObstacleFootprint" if c == "BaseObstacle" else c for c in f["critics"]]
    f["ObstacleFootprint.scale"] = f.pop("BaseObstacle.scale", OBSTACLE_WEIGHT)
    return "ObstacleFootprint critic"


def apply_static_map(cfg):
    """The global costmap plans on the WHOLE known map from t=0 (static layer) instead of only what the sensors saw:
    the main cause of getting boxed in among furniture."""
    g = cfg["global_costmap"]["global_costmap"]["ros__parameters"]
    g["rolling_window"] = False
    g.pop("width", None); g.pop("height", None)
    g["track_unknown_space"] = True
    g["plugins"] = ["static_layer", "obstacle_layer", "inflation_layer"]
    g["static_layer"] = {"plugin": "nav2_costmap_2d::StaticLayer", "map_subscribe_transient_local": True, "map_topic": "/map"}
    return "static map"


def apply_scans(cfg, names):
    """Replace the single /scan source with one LaserScan source per lidar (the robot's two RoboSense units),
    keeping every other setting of the tested `scan` entry (ranges, height cap, marking/clearing)."""
    for k in ("global_costmap", "local_costmap"):
        ob = cfg[k][k]["ros__parameters"]["obstacle_layer"]
        scan = ob.pop("scan")
        ob["observation_sources"] = " ".join(names)
        for n in names:
            ob[n] = dict(scan, topic="/" + n)
    return "scans " + ",".join(names)


def apply_camera_layer(cfg, min_z=-0.36):
    """SIM ONLY. The Godot lidars look UP from ~0.45 m, so low obstacles come from the depth camera in their own layer
    (in the lidar's layer every upward beam would raytrace-clear the low obstacle it passes over)."""
    for k in ("global_costmap", "local_costmap"):
        cm = cfg[k][k]["ros__parameters"]
        if "camera_layer" not in cm["plugins"]:
            cm["plugins"].insert(cm["plugins"].index("inflation_layer"), "camera_layer")
        cm["camera_layer"] = {
            "plugin": "nav2_costmap_2d::ObstacleLayer", "enabled": True, "observation_sources": "camera",
            "camera": {"topic": "/camera/obstacle_points", "data_type": "PointCloud2",
                       "sensor_frame": "camera_optical_link", "marking": True, "clearing": True,
                       "min_obstacle_height": min_z, "max_obstacle_height": 0.54,
                       "obstacle_max_range": 4.0, "raytrace_max_range": 4.5, "raytrace_min_range": 0.0,
                       "expected_update_rate": 0.0}}
    return f"camera layer (min z {min_z})"


def apply_amcl(cfg, start, scan_topic="scan", sigma_hit=0.1):
    """AMCL on the known map against DRIFTING odometry (no static map->odom then). Tuned 2026-10-14 on circuit4:
    alphas 0.1 (drift ~2 % + 0.002 rad/s), sigma_hit 0.1 m (96 % of endpoints within 0.10 m of the 0.05 m map),
    120 beams. sigma_hit is how far a beam's endpoint may land from the nearest map wall and still count as a hit:
    keep it about twice the map's cell size (0.1 m map -> 0.2)."""
    sx, sy, syaw = start
    cfg["amcl"] = {"ros__parameters": {
        "use_sim_time": False,
        "global_frame_id": "map", "odom_frame_id": "odom", "base_frame_id": "base_link",
        "scan_topic": scan_topic, "map_topic": "map", "first_map_only": False,
        "robot_model_type": "nav2_amcl::DifferentialMotionModel",
        "alpha1": 0.1, "alpha2": 0.1, "alpha3": 0.1, "alpha4": 0.1, "alpha5": 0.1,
        "laser_model_type": "likelihood_field", "max_beams": 120, "laser_min_range": 0.5, "laser_max_range": 12.0,
        "laser_likelihood_max_dist": 2.0, "sigma_hit": float(sigma_hit), "z_hit": 0.5, "z_rand": 0.5, "z_short": 0.05, "z_max": 0.05,
        "lambda_short": 0.1, "do_beamskip": False,
        "min_particles": 500, "max_particles": 2000, "pf_err": 0.05, "pf_z": 0.99, "resample_interval": 1,
        "recovery_alpha_fast": 0.0, "recovery_alpha_slow": 0.0,
        "update_min_d": 0.1, "update_min_a": 0.1,
        "tf_broadcast": True, "transform_tolerance": 1.0, "save_pose_rate": 0.5,
        "set_initial_pose": True, "always_reset_initial_pose": False,
        "initial_pose": {"x": float(sx), "y": float(sy), "z": 0.0, "yaw": float(syaw)},
    }}
    return f"amcl on /{scan_topic} from ({sx}, {sy}, {syaw}), sigma_hit {sigma_hit}"


def apply_keepaway(cfg, obstacle_scale=None, min_vel_x=None, cam_min_z=None):
    """The 2026-10-19 knobs: a stronger footprint critic (the controller grazed a 0.3 m pipe to 7 cm and froze),
    reversing (left off: the searches wandered), and the camera layer's floor cutoff."""
    f = cfg["controller_server"]["ros__parameters"]["FollowPath"]
    out = []
    if obstacle_scale is not None:
        f["ObstacleFootprint.scale"] = float(obstacle_scale); out.append(f"obstacle scale {obstacle_scale}")
    if min_vel_x is not None:
        f["min_vel_x"] = float(min_vel_x); out.append(f"min_vel_x {min_vel_x}")
    if cam_min_z is not None:
        for k in ("global_costmap", "local_costmap"):
            cfg[k][k]["ros__parameters"]["camera_layer"]["camera"]["min_obstacle_height"] = float(cam_min_z)
        out.append(f"camera min z {cam_min_z}")
    return ", ".join(out)


def profile_sim(cfg, vmax=0.7, amax=1.0):
    return [apply_vehicle(cfg, vmax, amax, SIM_FOOTPRINT, 0.10, SIM_MIN_SPEEDS)]


def profile_robot_shape(cfg, vmax=0.3, amax=1.0):
    """DWB stays the controller (the 10/10 was measured with it); min_vel_x 0 in the base params means no reversing."""
    return [apply_vehicle(cfg, vmax, amax, ROBOT_FOOTPRINT, 0.05, ROBOT_MIN_SPEEDS), apply_footprint_critic(cfg)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("params", help="the STAGED nav2_params.yaml (edited in place)")
    ap.add_argument("--profile", choices=("sim", "robot_shape", "none"), default="none",
                    help="none = only the --static-map/--scans/--amcl/... edits")
    ap.add_argument("--max-vel", type=float, default=None)
    ap.add_argument("--max-acc", type=float, default=1.0)
    ap.add_argument("--static-map", action="store_true")
    ap.add_argument("--scans", default="", help="comma list, e.g. scan_front,scan_rear (default: keep /scan)")
    ap.add_argument("--camera-layer", action="store_true")
    ap.add_argument("--footprint-critic", action="store_true",
                    help="ObstacleFootprint instead of BaseObstacle (robot_shape already has it)")
    ap.add_argument("--amcl", default=None, help='"x y yaw" initial pose')
    ap.add_argument("--amcl-scan", default="scan")
    ap.add_argument("--amcl-sigma-hit", type=float, default=0.1, help="about 2x the map cell size")
    ap.add_argument("--obstacle-scale", type=float, default=None)
    ap.add_argument("--min-vel-x", type=float, default=None)
    ap.add_argument("--cam-min-z", type=float, default=None)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.params))
    if a.profile == "sim":
        notes = profile_sim(cfg, a.max_vel if a.max_vel is not None else 0.7, a.max_acc)
    elif a.profile == "robot_shape":
        notes = profile_robot_shape(cfg, a.max_vel if a.max_vel is not None else 0.3, a.max_acc)
    else:
        notes = []
    if a.footprint_critic and "ObstacleFootprint.scale" not in cfg["controller_server"]["ros__parameters"]["FollowPath"]:
        notes.append(apply_footprint_critic(cfg))
    if a.static_map:
        notes.append(apply_static_map(cfg))
    if a.scans:
        notes.append(apply_scans(cfg, [s.strip() for s in a.scans.split(",") if s.strip()]))
    if a.camera_layer:
        notes.append(apply_camera_layer(cfg))
    if a.amcl:
        notes.append(apply_amcl(cfg, a.amcl.split(), a.amcl_scan, a.amcl_sigma_hit))
    k = apply_keepaway(cfg, a.obstacle_scale, a.min_vel_x, a.cam_min_z)
    if k:
        notes.append(k)
    yaml.safe_dump(cfg, open(a.params, "w"), sort_keys=False)
    print(f"  profile {a.profile} v{PROFILE_VERSION}: " + "; ".join(n for n in notes if n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
