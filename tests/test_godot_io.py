"""tools/godot/godot_io.py: the Godot sim's wire formats, frames and command mapping (no ROS)."""
import math
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "godot"))
import godot_io as g   # noqa: E402


def mm_rgba(mm: np.ndarray) -> bytes:
    """uint mm array -> RGBA8 bytes with R,G,B = mm (high byte first), A = 255."""
    mm = mm.astype(np.uint32)
    a = np.stack([(mm >> 16) & 255, (mm >> 8) & 255, mm & 255, np.full_like(mm, 255)], -1)
    return a.astype(np.uint8).tobytes()


def test_simv_header_roundtrip():
    h = g.SIMV.pack(g.SIMV_MAGIC, 640, 480, g.FMT_DEPTH_MM24, 7, 1_791_931_534_339_000)
    assert len(h) == 28
    d = g.parse_simv(h)
    assert (d["w"], d["h"], d["format"], d["seq"]) == (640, 480, 2, 7)
    assert d["unix_us"] == 1_791_931_534_339_000


def test_siml_header_layout_is_104_bytes():
    h = g.SIML.pack(g.SIML_MAGIC, 1, 1, 900, 96, 3, 5, 10, 20, -2.1, 0.6, 0.45, 0.0, 0.0, -0.71, 0.70, 0, 345600)
    assert g.SIML.size == 104 and len(h) == 104
    d = g.parse_siml(h)
    assert (d["index"], d["cols"], d["rings"], d["seq"], d["payload_bytes"]) == (1, 900, 96, 5, 345600)
    assert d["origin"] == (-2.1, 0.6, 0.45) and d["quat"][3] == 0.70


def test_bad_magic_is_rejected():
    try:
        g.parse_simv(b"\0" * 28)
    except ValueError:
        return
    raise AssertionError("bad magic accepted")


def test_mm24_and_depth_decoding():
    mm = np.array([[0, 1, 1234], [65536, 3000, 16_777_215]])
    assert g.mm24(mm_rgba(mm), 2, 3).tolist() == mm.tolist()
    d = g.depth_m(mm_rgba(mm), 2, 3)
    assert d.dtype == np.float32 and d[0, 0] == 0.0 and abs(d[1, 1] - 3.0) < 1e-6


def test_lidar_beam_geometry():
    el = [0.0, 30.0, 60.0]
    dirs = g.lidar_dirs(el, 4)                      # columns at 0, 90, 180, 270 deg CCW from +x
    assert np.allclose(dirs[0, 0], [1, 0, 0])
    assert np.allclose(dirs[0, 1], [0, 1, 0], atol=1e-12)       # 90 deg CCW = +y (left)
    assert np.allclose(dirs[1, 0], [math.cos(math.radians(30)), 0, 0.5])
    mm = np.zeros((3, 4)); mm[0, 1] = 2000; mm[1, 0] = 1000     # 2 m left, 1 m at 30 deg up
    pts = g.lidar_points(mm_rgba(mm), 3, 4, dirs, mount_xyz=(0.32, 0.0, -0.013))
    assert len(pts) == 2                                         # no-return beams dropped
    assert any(np.allclose(p, [0.32, 2.0, -0.013], atol=1e-5) for p in pts)
    assert any(np.allclose(p, [0.32 + math.cos(math.radians(30)), 0, 0.487], atol=1e-5) for p in pts)


def test_depth_points_land_in_front_of_the_camera():
    d = np.zeros((480, 640), np.float32)
    d[240, 320] = 3.0                                            # centre pixel, 3 m
    d[0, 0] = 2.0                                                # top-left pixel
    pts = g.depth_points(d, 320, 320, 319.5, 239.5, mount_xyz=(0.37646, 0, 0.03738), stride=8)
    centre = pts[np.argmin(np.abs(pts[:, 1]) + np.abs(pts[:, 2] - 0.03738))]
    assert abs(centre[0] - (0.37646 + 3.0)) < 1e-3             # straight ahead
    tl = [p for p in pts if p[2] > 0.5][0]                       # top-left pixel: up and to the left
    assert tl[1] > 0 and tl[2] > 0


def test_godot_enu_round_trip_and_axes():
    p = (1.5, 0.4, -2.0)                                         # Godot: x east, y up, z south
    e = g.godot_to_enu(p)
    assert e == (1.5, 2.0, 0.4)                                  # 2 m north, 0.4 m up
    assert g.enu_to_godot(e) == p


def test_quat_rpy_south_facing_spawn():
    # the sim's spawn: facing south in ENU (yaw -90 deg)
    q = (-0.00117, -0.00156, -0.71039, 0.70381)
    r, p, y = g.quat_to_rpy(*q)
    assert abs(math.degrees(y) + 90.5) < 0.2 and abs(r) < 0.01 and abs(p) < 0.01
    # body velocity: moving south while facing south = forward
    v = g.body_velocity(q, (0.0, -0.5, 0.0))
    assert abs(v[0] - 0.5) < 0.01 and abs(v[1]) < 0.02


def test_yaw_trim_boosts_an_under_turning_robot_to_the_target():
    # plant: the sim's M20 turns at ~50 % of the command (measured); first-order lag
    trim, rate, dt = g.YawTrim(), 0.0, 0.1
    for _ in range(40):
        out = trim(0.5, rate, dt)
        rate += (0.5 * out - rate) * 0.6
    assert abs(rate - 0.5) < 0.05


def test_yaw_trim_zero_target_sign_flip_and_clamp():
    trim = g.YawTrim()
    assert trim(0.0, 0.3, 0.1) == 0.0                          # stop means stop, no residual
    for _ in range(50):
        trim(0.5, 0.0, 0.1)                                     # integral winds up (clamped)
    assert trim.integ <= trim.i_max
    first_neg = trim(-0.5, 0.0, 0.1)                            # sign flip: integral reset first
    assert first_neg < 0 and trim.integ <= 0
    assert abs(trim(9.0, -9.0, 0.1)) <= trim.out_max           # output clamped


def test_vel_body_sign_and_caps():
    c = g.vel_body_cmd(0.3, 0.5)
    assert c["cmd"] == "vel_body" and c["forward"] == 0.3 and c["right"] == 0.0
    assert c["yaw_rate_dps"] < 0                                 # CCW (ROS +wz) = negative (sim: + is CW)
    assert abs(c["yaw_rate_dps"] + math.degrees(0.5)) < 1e-9
    c = g.vel_body_cmd(2.0, -9.0, vmax=0.5, wmax=1.5)
    assert c["forward"] == 0.5 and abs(c["yaw_rate_dps"] - math.degrees(1.5)) < 1e-9
    assert g.vel_body_cmd(-2.0, 0)["forward"] == -0.5


# ---- DriftOdom: dead-reckoned odometry for localization (AMCL) tests ----
def _drive(odom, v, yaw_rate, secs, dt=0.02, yaw0=0.0):
    """Feed DriftOdom a robot moving at v m/s, turning at yaw_rate rad/s (the TRUE yaw it would see)."""
    yaw = yaw0
    for _ in range(int(round(secs / dt))):
        yaw += yaw_rate * dt
        pose = odom.step(v, yaw, dt)
    return pose, yaw


def test_drift_odom_without_noise_is_exact_dead_reckoning():
    o = g.DriftOdom(speed_noise=0.0, gyro_bias=0.0, gyro_noise=0.0)
    o.reset(1.0, 2.0, 0.0)
    (x, y, th), _ = _drive(o, 0.5, 0.0, 10.0)                 # straight, 10 s at 0.5 m/s along +x
    assert abs(x - 6.0) < 1e-9 and abs(y - 2.0) < 1e-9 and abs(th) < 1e-12


def test_drift_odom_gyro_bias_grows_heading_error_linearly():
    o = g.DriftOdom(speed_noise=0.0, gyro_bias=0.002, gyro_noise=0.0)
    o.reset(0.0, 0.0, 0.0)
    (_, _, th), true_yaw = _drive(o, 0.3, 0.0, 60.0)          # walking straight for a minute
    assert abs((th - true_yaw) - 0.002 * 60.0) < 1e-6        # 0.12 rad: bias x time


def test_drift_odom_standing_still_does_not_drift():
    o = g.DriftOdom()                                        # full noise + bias
    o.reset(1.0, 2.0, 0.5)
    (x, y, th), true_yaw = _drive(o, 0.0, 0.0, 600.0, yaw0=0.5)   # parked for 10 minutes
    assert (x, y) == (1.0, 2.0) and abs(th - true_yaw) < 1e-12


def test_drift_odom_scale_error_and_reproducible_noise():
    o = g.DriftOdom(speed_noise=0.0, gyro_bias=0.0, gyro_noise=0.0, scale=1.05)
    o.reset(0.0, 0.0, 0.0)
    (x, _, _), _ = _drive(o, 1.0, 0.0, 10.0)
    assert abs(x - 10.5) < 1e-9                              # 5 % long
    runs = []
    for _ in range(2):
        o = g.DriftOdom(seed=7)
        o.reset(0.0, 0.0, 0.0)
        runs.append(_drive(o, 0.5, 0.1, 20.0)[0])
    assert runs[0] == runs[1]                                # same seed -> same drift


def test_drift_odom_follows_true_turns_across_the_pi_wrap():
    o = g.DriftOdom(speed_noise=0.0, gyro_bias=0.0, gyro_noise=0.0)
    o.reset(0.0, 0.0, 3.0)
    (_, _, th), true_yaw = _drive(o, 0.0, 0.5, 2.0, yaw0=3.0)  # turns through +pi
    wrap = lambda a: math.atan2(math.sin(a), math.cos(a))
    assert abs(wrap(th - true_yaw)) < 1e-9
