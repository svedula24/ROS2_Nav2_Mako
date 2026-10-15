"""Godot sim (Bright Robotics Simulator) wire formats, frames and conversions. No ROS.

The sim (stateful-drone-sim, `--api --vehicle=m20`) serves, on 127.0.0.1:
  7799  newline-delimited JSON request/reply (state, telemetry, payload, vel_body, ...)
  7800  colour frames: 28-byte 'SIMV' header + width*height RGBA8
  7802  depth frames:  same header (format 2) + RGBA8 whose R,G,B = z-depth in mm, 0 = none
  7803  lidar sweeps:  104-byte 'SIML' header + rings*cols RGBA8 whose R,G,B = range in mm
All little-endian. Frames: Godot is y-up with north = -z, so ENU = (x, -z, y) (site_frame.gd);
the site's ENU frame is the Nav2 `map` frame here. godot_bridge.py turns these into ROS topics.
"""
import json
import math
import socket
import struct
import threading

import numpy as np

SIMV = struct.Struct("<IIIIIQ")                      # magic, w, h, format, seq, unix_us
SIMV_MAGIC = 0x564D4953
SIML = struct.Struct("<IHHIIIIQQ3d4dII")             # 104 bytes
SIML_MAGIC = 0x4C4D4953
FMT_RGBA8, FMT_DEPTH_MM24 = 0, 2

# camera_optical (x right, y down, z forward) -> camera link / base (x fwd, y left, z up)
R_BASE_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


# ------------------------------------------------------------------ JSON API
class Api:
    """One persistent connection to the JSON API; one request in flight (thread-safe)."""

    def __init__(self, host="127.0.0.1", port=7799, timeout=3.0):
        self.addr, self.timeout = (host, port), timeout
        self._sock, self._buf, self._lock = None, b"", threading.Lock()

    def _connect(self):
        self._sock = socket.create_connection(self.addr, timeout=self.timeout)
        self._buf = b""

    def call(self, msg: dict) -> dict:
        with self._lock:
            for attempt in (0, 1):                     # one reconnect on a dropped socket
                try:
                    if self._sock is None:
                        self._connect()
                    self._sock.sendall((json.dumps(msg) + "\n").encode())
                    while b"\n" not in self._buf:
                        chunk = self._sock.recv(1 << 20)
                        if not chunk:
                            raise ConnectionError("api closed")
                        self._buf += chunk
                    line, self._buf = self._buf.split(b"\n", 1)
                    return json.loads(line)
                except (OSError, ConnectionError, ValueError):
                    self.close()
                    if attempt:
                        raise
        raise ConnectionError("unreachable")

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None


# ------------------------------------------------------------------ binary streams
def recv_exact(sock, n: int) -> bytes:
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("stream closed")
        got += k
    return bytes(buf)


def parse_simv(h: bytes) -> dict:
    magic, w, hh, fmt, seq, us = SIMV.unpack(h[:SIMV.size])
    if magic != SIMV_MAGIC:
        raise ValueError(f"bad SIMV magic {magic:#x}")
    return {"w": w, "h": hh, "format": fmt, "seq": seq, "unix_us": us}


def parse_siml(h: bytes) -> dict:
    (magic, ver, idx, cols, rings, fmt, seq, us, sim_us,
     ox, oy, oz, qx, qy, qz, qw, flags, nbytes) = SIML.unpack(h[:SIML.size])
    if magic != SIML_MAGIC:
        raise ValueError(f"bad SIML magic {magic:#x}")
    return {"version": ver, "index": idx, "cols": cols, "rings": rings, "format": fmt, "seq": seq,
            "unix_us": us, "sim_us": sim_us, "origin": (ox, oy, oz), "quat": (qx, qy, qz, qw),
            "flags": flags, "payload_bytes": nbytes}


def read_simv(sock):
    """-> (header dict, RGBA bytes) for one frame."""
    hdr = parse_simv(recv_exact(sock, SIMV.size))
    return hdr, recv_exact(sock, hdr["w"] * hdr["h"] * 4)


def read_siml(sock):
    hdr = parse_siml(recv_exact(sock, SIML.size))
    return hdr, recv_exact(sock, hdr["payload_bytes"])


def mm24(rgba: bytes, h: int, w: int) -> np.ndarray:
    """RGBA8 whose R,G,B are millimetres (high byte first) -> uint32 mm, shape (h, w)."""
    a = np.frombuffer(rgba, np.uint8).reshape(h, w, 4).astype(np.uint32)
    return (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]


def depth_m(rgba: bytes, h: int, w: int) -> np.ndarray:
    """Depth frame -> float32 metres (z-depth along the optical axis), 0 = no return."""
    return mm24(rgba, h, w).astype(np.float32) * 1e-3


def rgba_to_rgb(rgba: bytes, h: int, w: int) -> np.ndarray:
    return np.frombuffer(rgba, np.uint8).reshape(h, w, 4)[..., :3]


# ------------------------------------------------------------------ geometry
def godot_to_enu(p):
    return (p[0], -p[2], p[1])


def enu_to_godot(p):
    return (p[0], p[2], -p[1])


def quat_to_mat(x, y, z, w) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_to_rpy(x, y, z, w):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def rpy_to_mat(r, p, y) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def lidar_dirs(elevations_deg, cols: int) -> np.ndarray:
    """Unit beam directions in the sensor FLU frame, shape (rings, cols, 3).
    Row r = elevations_deg[r] (row 0 = lowest ring); column c at 360*c/cols deg CCW from +x."""
    el = np.radians(np.asarray(elevations_deg, float))[:, None]
    az = (2 * np.pi * np.arange(cols) / cols)[None, :]
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el) * np.ones_like(az)], -1)


def lidar_points(payload: bytes, rings: int, cols: int, dirs: np.ndarray, mount_xyz, mount_rpy=(0, 0, 0),
                 rmin=0.1, rmax=60.0) -> np.ndarray:
    """One sweep's range image -> float32 (N, 3) points in base_link (sensor pose = URDF mount)."""
    r = mm24(payload, rings, cols).astype(np.float32) * 1e-3
    ok = (r >= rmin) & (r <= rmax)
    p = dirs[ok] * r[ok][:, None]
    if any(mount_rpy):
        p = p @ rpy_to_mat(*mount_rpy).T
    return (p + np.asarray(mount_xyz, float)).astype(np.float32)


def depth_points(depth: np.ndarray, fx, fy, cx, cy, mount_xyz, stride=8, rmax=6.0) -> np.ndarray:
    """Depth image (z metres) -> float32 (N, 3) points in base_link, subsampled every `stride` px.
    Covers low obstacles under the upper-hemisphere lidars; the scan converter's height band then
    drops the floor and the ceiling."""
    d = depth[::stride, ::stride]
    vv, uu = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    ok = (d > 0.05) & (d < rmax)
    z = d[ok]
    p_opt = np.stack([(uu[ok] - cx) / fx * z, (vv[ok] - cy) / fy * z, z], 1)
    return (p_opt @ R_BASE_OPT.T + np.asarray(mount_xyz, float)).astype(np.float32)


# ------------------------------------------------------------------ commands
def vel_body_cmd(vx: float, wz: float, vmax=0.5, wmax=1.5) -> dict:
    """ROS twist (vx m/s forward, wz rad/s CCW) -> the sim's vel_body. The sim's yaw rate is
    degrees per second CLOCKWISE (api.gd:428), so the sign flips. `right` is ignored on the M20."""
    vx = max(-vmax, min(vmax, float(vx)))
    wz = max(-wmax, min(wmax, float(wz)))
    return {"cmd": "vel_body", "forward": vx, "right": 0.0, "yaw_rate_dps": -math.degrees(wz)}


class YawTrim:
    """Closed-loop yaw-rate trim on top of the sim's own skid-steer controller.

    Measured 2026-10-13: the sim's M20 turns at only 44-57 % of the commanded yaw rate (its yaw loop's
    integrator is slow), so DWB's predicted arcs ran wide and the robot drifted > 2 m off its path.
    Mako's own locomotion bridge trims the same way (PI on the gyro, bridge_node.py SimVendorSDK).
      out = ff * wz + kp * (wz - gyro) + ki * integral(wz - gyro)
    The integral resets when the target stops or changes sign, and is clamped."""

    def __init__(self, ff=1.8, kp=0.8, ki=1.5, i_max=1.0, out_max=2.5):
        self.ff, self.kp, self.ki, self.i_max, self.out_max = ff, kp, ki, i_max, out_max
        self.integ, self.last_target = 0.0, 0.0

    def __call__(self, wz: float, gyro_z: float, dt: float) -> float:
        if abs(wz) < 1e-3 or wz * self.last_target < 0:
            self.integ = 0.0
        self.last_target = wz
        if abs(wz) < 1e-3:
            return 0.0
        err = wz - gyro_z
        self.integ = max(-self.i_max, min(self.i_max, self.integ + err * dt))
        out = self.ff * wz + self.kp * err + self.ki * self.integ
        return max(-self.out_max, min(self.out_max, out))


class DriftOdom:
    """Dead-reckoned odometry that drifts like a real robot's, for testing localization (AMCL) against the sim.

    Same model as tools/mujoco_sim.py's odometry:
      heading += true yaw change + gyro_bias·dt + N(0, gyro_noise)     a gyro: right rate, a small bias, noise
      distance = forward speed · scale · (1 + N(0, speed_noise)) · dt   legs/encoders: slip noise + scale error
    Heading error accumulates (bias · time while moving), and turns that into position error that grows with
    distance, which is what a localizer has to correct. Deterministic for a given seed. The pose starts wherever
    reset() puts it.
    Standing still (speed < STILL_V and turn rate < STILL_W) adds no bias or noise: a zero-velocity update, as real
    state estimators do. Without it the heading drifts ~7 deg/min while parked, and AMCL can't fix that because it
    only updates after the robot moves (update_min_d / update_min_a)."""

    STILL_V, STILL_W = 0.02, 0.02              # m/s, rad/s

    def __init__(self, seed=0, speed_noise=0.02, gyro_bias=0.002, gyro_noise=0.0005, scale=1.0):
        self.rng = np.random.default_rng(seed)
        self.speed_noise, self.gyro_bias, self.gyro_noise, self.scale = speed_noise, gyro_bias, gyro_noise, scale
        self.pose = None                       # (x, y, yaw) in the odom frame
        self._prev_true_yaw = None

    def reset(self, x, y, yaw):
        self.pose, self._prev_true_yaw = (float(x), float(y), float(yaw)), float(yaw)

    def step(self, v_forward, true_yaw, dt):
        """Advance by dt seconds given the body's forward speed and the TRUE yaw now. -> (x, y, yaw)."""
        x, y, th = self.pose
        dyaw = math.atan2(math.sin(true_yaw - self._prev_true_yaw), math.cos(true_yaw - self._prev_true_yaw))
        self._prev_true_yaw = true_yaw
        if abs(v_forward) < self.STILL_V and abs(dyaw) < self.STILL_W * max(dt, 1e-6):   # standing still
            th = math.atan2(math.sin(th + dyaw), math.cos(th + dyaw))
            self.pose = (x, y, th)
            return self.pose
        th = th + dyaw + self.gyro_bias * dt + (self.rng.normal(0.0, self.gyro_noise) if self.gyro_noise else 0.0)
        th = math.atan2(math.sin(th), math.cos(th))
        v = v_forward * self.scale * (1.0 + (self.rng.normal(0.0, self.speed_noise) if self.speed_noise else 0.0))
        self.pose = (x + v * math.cos(th) * dt, y + v * math.sin(th) * dt, th)
        return self.pose


def body_velocity(q, vel_enu):
    """World (ENU) linear velocity -> base_link frame."""
    return quat_to_mat(*q).T @ np.asarray(vel_enu, float)
