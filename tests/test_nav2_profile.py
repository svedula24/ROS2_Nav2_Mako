"""tools/nav/nav2_profile.py: the Nav2 parameter profiles for the sim and the real M20 (pure, no ROS)."""
import os
import sys

import pytest
import yaml

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools", "nav"))
import nav2_profile as prof  # noqa: E402

PARAMS = os.path.join(ROOT, "src", "m20_navigation", "config", "nav2_params.yaml")


@pytest.fixture
def cfg():
    return yaml.safe_load(open(PARAMS))


def follow(cfg):
    return cfg["controller_server"]["ros__parameters"]["FollowPath"]


def test_robot_shape_golden_values(cfg):
    prof.profile_robot_shape(cfg)
    f = follow(cfg)
    assert f["max_vel_x"] == f["max_speed_xy"] == 0.3
    assert (f["min_speed_xy"], f["min_speed_theta"]) == (0.15, 0.35)
    assert f["xy_goal_tolerance"] == 0.45
    assert "ObstacleFootprint" in f["critics"] and "BaseObstacle" not in f["critics"]
    assert f["ObstacleFootprint.scale"] == 0.15 and "BaseObstacle.scale" not in f
    for k in ("global_costmap", "local_costmap"):
        cm = cfg[k][k]["ros__parameters"]
        assert "robot_radius" not in cm
        assert cm["footprint"] == "[[0.45, 0.3], [0.45, -0.3], [-0.45, -0.3], [-0.45, 0.3]]"
        assert cm["footprint_padding"] == 0.05
        assert cm["inflation_layer"]["inflation_radius"] == 1.0 and cm["inflation_layer"]["cost_scaling_factor"] == 1.5
    s = cfg["velocity_smoother"]["ros__parameters"]
    assert s["max_velocity"][0] == 0.3 and s["min_velocity"][0] == -0.2         # recoveries may still back up
    assert f["min_vel_x"] == 0.0                                                 # DWB itself never reverses
    assert cfg["controller_server"]["ros__parameters"]["progress_checker"]["movement_time_allowance"] == 15.0


def test_sim_profile_reproduces_the_launcher_patch(cfg):
    prof.profile_sim(cfg, vmax=0.5)
    f = follow(cfg)
    assert f["max_vel_x"] == 0.5 and (f["min_speed_xy"], f["min_speed_theta"]) == (0.10, 0.25)
    assert f["BaseObstacle.scale"] == 0.15 and "BaseObstacle" in f["critics"]
    cm = cfg["local_costmap"]["local_costmap"]["ros__parameters"]
    assert cm["footprint"] == "[[0.43, 0.28], [0.43, -0.28], [-0.43, -0.28], [-0.43, 0.28]]" and cm["footprint_padding"] == 0.1
    assert cfg["velocity_smoother"]["ros__parameters"]["min_velocity"][0] == -0.2


def test_static_map_and_two_scans(cfg):
    prof.apply_static_map(cfg)
    prof.apply_scans(cfg, ["scan_front", "scan_rear"])
    g = cfg["global_costmap"]["global_costmap"]["ros__parameters"]
    assert g["rolling_window"] is False and "width" not in g and g["track_unknown_space"] is True
    assert g["plugins"] == ["static_layer", "obstacle_layer", "inflation_layer"]
    for k in ("global_costmap", "local_costmap"):
        ob = cfg[k][k]["ros__parameters"]["obstacle_layer"]
        assert ob["observation_sources"] == "scan_front scan_rear" and "scan" not in ob
        assert ob["scan_rear"]["topic"] == "/scan_rear"
        # the tested single-scan settings carry over unchanged (ranges 10 / 12 m, 2.0 m height cap)
        assert (ob["scan_rear"]["obstacle_max_range"], ob["scan_rear"]["raytrace_max_range"],
                ob["scan_rear"]["max_obstacle_height"]) == (10.0, 12.0, 2.0)


def test_camera_layer_is_sim_only_and_amcl_block(cfg):
    prof.apply_camera_layer(cfg)
    cm = cfg["global_costmap"]["global_costmap"]["ros__parameters"]
    assert cm["plugins"].index("camera_layer") == cm["plugins"].index("inflation_layer") - 1
    assert cm["camera_layer"]["camera"]["min_obstacle_height"] == -0.36
    prof.apply_amcl(cfg, ("-2.13", "0.35", "-1.6"), scan_topic="scan_front", sigma_hit=0.2)
    a = cfg["amcl"]["ros__parameters"]
    assert a["scan_topic"] == "scan_front" and a["initial_pose"] == {"x": -2.13, "y": 0.35, "z": 0.0, "yaw": -1.6}
    assert a["sigma_hit"] == 0.2 and a["max_beams"] == 120                    # 0.2 = 2x the 0.1 m map's cell size


def test_keepaway_knobs(cfg):
    prof.profile_robot_shape(cfg)
    assert prof.apply_keepaway(cfg, obstacle_scale=1.0, min_vel_x=0.0) == "obstacle scale 1.0, min_vel_x 0.0"
    assert follow(cfg)["ObstacleFootprint.scale"] == 1.0


def test_cli_edits_in_place_and_reports_the_version(tmp_path, capsys):
    p = tmp_path / "nav2_params.yaml"
    p.write_text(open(PARAMS).read())
    assert prof.main([str(p), "--profile", "robot_shape", "--static-map", "--scans", "scan_front,scan_rear"]) == 0
    out = capsys.readouterr().out
    assert f"profile robot_shape v{prof.PROFILE_VERSION}" in out and "scans scan_front,scan_rear" in out
    cfg = yaml.safe_load(open(p))
    assert follow(cfg)["max_vel_x"] == 0.3 and cfg["global_costmap"]["global_costmap"]["ros__parameters"]["rolling_window"] is False
