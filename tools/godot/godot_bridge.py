#!/usr/bin/env python3
"""ROS 2 bridge to the Godot sim (Bright Robotics Simulator, `--api --vehicle=m20`).

Stands in for tools/mujoco_sim.py + the locomotion bridge, so the rest of the stack (commander,
Nav2, mission server, m20_nlnav, detector, Viser) runs unchanged against the Godot world.

Published (stamped on this process's wall clock, the same as TF; map = the site's ENU frame):
  /odom (odom -> base_link) + TF odom->base_link   ground truth (the M20_GT_LOC role: bringup
                                                   publishes a static identity map->odom)
  /odom_true (map -> base_link_true, full 3D)      for the detector, Viser and nlnav
  /IMU_DATA, /JOINTS_DATA (drdds)                  the commander's tip-over failsafe; Viser's legs
  /LIDAR/POINTS (PointCloud2 xyz, base_link)       both hemispherical lidars (-> /scan)
  /camera/obstacle_points (PointCloud2, base_link) the depth camera as points (GODOT_DEPTH_OBSTACLES=1): the
                                                   lidars only look UP from ~0.45 m, so low obstacles come
                                                   from the camera. Bringup gives these their own costmap
                                                   layer, so the lidar's raytracing can't clear them; floor
                                                   points are kept (they clear free floor, never mark).
  /camera/image_raw, /camera/depth, /camera/camera_info   one stamp per frame pair (detector pairs on it)
  /m20/health (HealthReport, node m20_godot_bridge)       ERROR while the sim is unreachable
Consumed:
  /cmd_vel -> vel_body (the sim mixes and steers itself), ONLY while /m20/mode says armed; zero when
  disarmed or when /cmd_vel goes stale. Capped at GODOT_VMAX (0.5 m/s: the M20 can flip on furniture).
  The yaw rate is trimmed on the measured gyro (godot_io.YawTrim): the sim's M20 turns at ~50 % of the
  command, which sent DWB's arcs wide and made it drop the path.

Env: GODOT_HOST (127.0.0.1), GODOT_API_PORT (7799), GODOT_VIDEO_PORT (7800), GODOT_DEPTH_PORT (7802),
GODOT_LIDAR_PORT (7803), GODOT_POSE_HZ (50), GODOT_CAM_HZ (10), GODOT_VMAX (0.5), GODOT_WMAX (1.5),
GODOT_CLOUD_ZBAND (-0.45,0.70: base_link z kept in /LIDAR/POINTS),
GODOT_DEPTH_OBSTACLES (1).
"""
import array
import os
import socket
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from drdds.msg import ImuData, JointsData
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from m20_msgs.msg import HealthReport, RobotMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import godot_io as gio   # noqa: E402

HOST = os.environ.get("GODOT_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("GODOT_API_PORT", 7799))
VIDEO_PORT = int(os.environ.get("GODOT_VIDEO_PORT", 7800))
DEPTH_PORT = int(os.environ.get("GODOT_DEPTH_PORT", 7802))
LIDAR_PORT = int(os.environ.get("GODOT_LIDAR_PORT", 7803))
POSE_HZ = float(os.environ.get("GODOT_POSE_HZ", 50))
CAM_HZ = float(os.environ.get("GODOT_CAM_HZ", 10))
VMAX = float(os.environ.get("GODOT_VMAX", 0.5))
WMAX = float(os.environ.get("GODOT_WMAX", 1.5))
DEPTH_OBSTACLES = os.environ.get("GODOT_DEPTH_OBSTACLES", "1") == "1"
CMD_STALE_S = 0.5
# Only points the scan converter can use (pointcloud_to_laserscan keeps -0.36..0.54 m around
# base_link), with margin: the full sweeps are ~155k points, this is ~10x fewer.
ZBAND = tuple(float(v) for v in os.environ.get("GODOT_CLOUD_ZBAND", "-0.45,0.70").split(","))


def u8(arr) -> array.array:
    """numpy -> array('B'). rclpy (Humble) checks a bytes/list value element by element in Python;
    an array.array('B') takes the fast path. At camera/lidar sizes that is the whole CPU budget."""
    return array.array("B", np.ascontiguousarray(arr).tobytes())


def cloud_msg(points: np.ndarray, frame: str, stamp) -> PointCloud2:
    m = PointCloud2()
    m.header.stamp, m.header.frame_id = stamp, frame
    m.height, m.width = 1, len(points)
    m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
    m.is_bigendian, m.point_step, m.is_dense = False, 12, True
    m.row_step = 12 * len(points)
    m.data = u8(np.ascontiguousarray(points, np.float32))
    return m


class GodotBridge(Node):
    def __init__(self):
        super().__init__("m20_godot_bridge")
        self.api_pose = gio.Api(HOST, API_PORT)        # pose loop
        self.api_cmd = gio.Api(HOST, API_PORT)         # commands (separate socket: no head-of-line wait)
        self.payload = self._wait_payload()
        cam = self.payload["camera"]
        self.K = cam["intrinsics"]
        self.cam_mount = self.payload["mounts"][cam["body"]]["xyz"]
        self.lidars = []
        for L in self.payload["lidars"]:
            m = self.payload["mounts"][L["body"]]
            self.lidars.append({"dirs": gio.lidar_dirs(L["elevations_deg"], L["cols"]), "xyz": m["xyz"],
                                "rpy": m["rpy"], "range": L["range_m"]})

        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.true_pub = self.create_publisher(Odometry, "/odom_true", 10)
        self.imu_pub = self.create_publisher(ImuData, "/IMU_DATA", 10)
        self.jd_pub = self.create_publisher(JointsData, "/JOINTS_DATA", 10)
        self.cloud_pub = self.create_publisher(PointCloud2, "/LIDAR/POINTS", 10)
        self.cam_cloud_pub = self.create_publisher(PointCloud2, "/camera/obstacle_points", 5)
        self.img_pub = self.create_publisher(Image, "/camera/image_raw", 5)
        self.depth_pub = self.create_publisher(Image, "/camera/depth", 5)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", 5)
        self.health_pub = self.create_publisher(HealthReport, "/m20/health", 10)
        self.tf_bc = TransformBroadcaster(self)
        self._static_tf()

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(RobotMode, "/m20/mode", self.on_mode, latched)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd_vel, 10)
        self.armed, self.cmd, self.cmd_t, self.sent_zero = False, (0.0, 0.0), 0.0, True
        self.gyro_z, self.trim, self.last_tick = 0.0, gio.YawTrim(), time.monotonic()
        self.sim_ok, self.sim_err = True, ""
        self.lock = threading.Lock()
        self.latest_lidar = {}                         # index -> points (in the z band)
        self.frames = {"c": {}, "d": {}}               # seq -> (header, pixels): newest few only
        self.sweeps = {}                               # lidar index -> newest (header, payload)

        self.create_timer(0.1, self.cmd_tick)           # 10 Hz, inside the sim's 1 s watchdog
        self.create_timer(1.0, self.heartbeat)
        for fn in (self.pose_loop, self.camera_loop, self.lidar_loop):
            threading.Thread(target=fn, daemon=True).start()
        self.get_logger().info(f"Godot bridge up: api {HOST}:{API_PORT}, {len(self.lidars)} lidars, "
                               f"camera {self.K['width']}x{self.K['height']}, depth obstacles={DEPTH_OBSTACLES}")

    # ---------------------------------------------------------------- setup
    def _wait_payload(self):
        while rclpy.ok():
            try:
                r = self.api_pose.call({"cmd": "payload"})
                if r.get("ok") and r.get("lidars") is not None:
                    return r
            except (OSError, ConnectionError, ValueError) as e:
                self.get_logger().warn(f"waiting for the Godot sim on {HOST}:{API_PORT} ({e})", throttle_duration_sec=5)
            time.sleep(1.0)

    def _static_tf(self):
        self.static_bc = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = "base_link", "camera_optical_link"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (float(v) for v in self.cam_mount)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = -0.5, 0.5, -0.5, 0.5
        self.static_bc.sendTransform(t)

    # ---------------------------------------------------------------- commands
    def on_mode(self, m: RobotMode):
        self.armed = bool(m.armed)

    def on_cmd_vel(self, m: Twist):
        self.cmd, self.cmd_t = (m.linear.x, m.angular.z), time.monotonic()

    def cmd_tick(self):
        live = self.armed and time.monotonic() - self.cmd_t < CMD_STALE_S
        if not live and self.sent_zero:
            return                                     # idle: let the sim's own watchdog hold it
        vx, wz = self.cmd if live else (0.0, 0.0)
        now = time.monotonic()
        wz = max(-WMAX, min(WMAX, wz))
        # the sim's M20 under-turns (~50 % of the command); trim on the measured gyro rate
        wz_out = self.trim(wz, self.gyro_z, now - self.last_tick)
        self.last_tick = now
        try:
            self.api_cmd.call(gio.vel_body_cmd(vx, wz_out, VMAX, self.trim.out_max))
            self.sent_zero = not live
        except (OSError, ConnectionError, ValueError) as e:
            self._sim_down(f"vel_body failed: {e}")

    # ---------------------------------------------------------------- pose / imu / joints
    def pose_loop(self):
        period = 1.0 / POSE_HZ
        while rclpy.ok():
            t0 = time.monotonic()
            try:
                t = self.api_pose.call({"cmd": "telemetry"})
                self.publish_state(t)
                self.sim_ok = True
            except (OSError, ConnectionError, ValueError, KeyError) as e:
                self._sim_down(f"telemetry failed: {e}")
                time.sleep(1.0)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def publish_state(self, t: dict):
        now = self.get_clock().now().to_msg()
        b = t["base_enu"]
        (px, py, pz), q = b["t"], b["q"]
        roll, pitch, yaw = gio.quat_to_rpy(*q)
        v_body = gio.body_velocity(q, t["vel_enu"])
        imu = t.get("imu") or {}
        gyro = imu.get("gyro", [0.0, 0.0, 0.0])
        self.gyro_z = float(gyro[2])
        accel = imu.get("accel", [0.0, 0.0, 9.81])

        od = Odometry()
        od.header.stamp, od.header.frame_id, od.child_frame_id = now, "odom", "base_link"
        od.pose.pose.position.x, od.pose.pose.position.y = float(px), float(py)
        od.pose.pose.orientation.z, od.pose.pose.orientation.w = float(np.sin(yaw / 2)), float(np.cos(yaw / 2))
        od.twist.twist.linear.x, od.twist.twist.angular.z = float(v_body[0]), float(gyro[2])
        self.odom_pub.publish(od)
        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = now, "odom", "base_link"
        tf.transform.translation.x, tf.transform.translation.y = float(px), float(py)
        tf.transform.rotation.z, tf.transform.rotation.w = od.pose.pose.orientation.z, od.pose.pose.orientation.w
        self.tf_bc.sendTransform(tf)

        ot = Odometry()
        ot.header.stamp, ot.header.frame_id, ot.child_frame_id = now, "map", "base_link_true"
        ot.pose.pose.position.x, ot.pose.pose.position.y, ot.pose.pose.position.z = float(px), float(py), float(pz)
        o = ot.pose.pose.orientation
        o.x, o.y, o.z, o.w = (float(v) for v in q)
        self.true_pub.publish(ot)

        im = ImuData()
        im.header.stamp = now
        d = im.data
        d.roll, d.pitch, d.yaw = float(roll), float(pitch), float(yaw)
        d.omega_x, d.omega_y, d.omega_z = (float(v) for v in gyro)
        d.acc_x, d.acc_y, d.acc_z = (float(v) for v in accel)
        self.imu_pub.publish(im)

        joints = t.get("joints") or []
        if len(joints) >= 16:
            jd = JointsData()
            jd.header.stamp = now
            for i in range(16):
                e, j = jd.data.joints_data[i], joints[i]
                e.position, e.velocity, e.torque = float(j["q"]), float(j["dq"]), float(j["tau"])
            self.jd_pub.publish(jd)

    # ---------------------------------------------------------------- stream readers
    # Readers ONLY drain the sockets and keep the newest frame: the sim sends from its main thread,
    # so a slow reader would stall Godot's rendering AND physics (measured: 103 -> 16 fps, real-time
    # factor 0.5). Processing and publishing happen in separate threads, on the newest data only.
    def _reader(self, port, read, store):
        while rclpy.ok():
            try:
                s = socket.create_connection((HOST, port), timeout=10)
                while rclpy.ok():
                    store(*read(s))
            except (OSError, ConnectionError, ValueError) as e:
                self.get_logger().warn(f"stream :{port}: {e}; reconnecting", throttle_duration_sec=10)
                time.sleep(2.0)

    def _keep_colour(self, h, px):
        with self.lock:
            self.frames["c"][h["seq"]] = (h, px)
            self._trim(self.frames["c"])

    def _keep_depth(self, h, px):
        with self.lock:
            self.frames["d"][h["seq"]] = (h, px)
            self._trim(self.frames["d"])

    def _keep_sweep(self, h, px):
        with self.lock:
            self.sweeps[h["index"]] = (h, px)

    @staticmethod
    def _trim(d, keep=4):
        while len(d) > keep:
            d.pop(min(d))

    def camera_loop(self):
        for port, read, store in ((VIDEO_PORT, gio.read_simv, self._keep_colour),
                                  (DEPTH_PORT, gio.read_simv, self._keep_depth)):
            threading.Thread(target=self._reader, args=(port, read, store), daemon=True).start()
        last_seq = -1
        while rclpy.ok():
            time.sleep(1.0 / CAM_HZ)
            with self.lock:
                both = sorted(set(self.frames["c"]) & set(self.frames["d"]))
                if not both or both[-1] == last_seq:
                    continue
                last_seq = both[-1]
                ch, cpx = self.frames["c"][last_seq]
                dh, dpx = self.frames["d"][last_seq]
            self.publish_camera(ch, cpx, dh, dpx)

    def publish_camera(self, ch, cpx, dh, dpx):
        w, h = ch["w"], ch["h"]
        # Stamped on THIS process's clock (the same one as /odom and TF), not the sim's capture
        # time: Windows' clock runs ~0.15 s ahead of WSL's, and future-stamped clouds made the
        # costmaps drop every observation (TF can't extrapolate forward).
        stamp = self.get_clock().now().to_msg()
        img = Image()
        img.header.stamp, img.header.frame_id = stamp, "camera_optical_link"
        img.height, img.width, img.encoding, img.step = h, w, "rgb8", w * 3
        img.data = u8(gio.rgba_to_rgb(cpx, h, w))
        depth = gio.depth_m(dpx, dh["h"], dh["w"])
        dm = Image()
        dm.header = img.header
        dm.height, dm.width, dm.encoding, dm.step = dh["h"], dh["w"], "32FC1", dh["w"] * 4
        dm.data = u8(depth)
        info = CameraInfo()
        info.header = img.header
        info.width, info.height, info.distortion_model = w, h, "plumb_bob"
        fx, fy, cx, cy = self.K["fx"], self.K["fy"], self.K["cx"], self.K["cy"]
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.d = [0.0] * 5
        self.img_pub.publish(img)
        self.depth_pub.publish(dm)
        self.info_pub.publish(info)
        if DEPTH_OBSTACLES:
            pts = gio.depth_points(depth, fx, fy, cx, cy, self.cam_mount, stride=4, rmax=4.5)
            self.cam_cloud_pub.publish(cloud_msg(pts, "base_link", stamp))

    # ---------------------------------------------------------------- lidar
    def lidar_loop(self):
        threading.Thread(target=self._reader, args=(LIDAR_PORT, gio.read_siml, self._keep_sweep), daemon=True).start()
        seen = {}
        while rclpy.ok():
            time.sleep(0.1)                            # 10 Hz, the lidars' rate
            with self.lock:
                fresh = {i: hp for i, hp in self.sweeps.items() if seen.get(i) != hp[0]["seq"]}
            if not fresh:
                continue
            for i, (hdr, px) in fresh.items():
                seen[i] = hdr["seq"]
                if i >= len(self.lidars):
                    continue
                L = self.lidars[i]
                pts = gio.lidar_points(px, hdr["rings"], hdr["cols"], L["dirs"], L["xyz"], L["rpy"],
                                       rmin=L["range"][0], rmax=L["range"][1])
                self.latest_lidar[i] = pts[(pts[:, 2] > ZBAND[0]) & (pts[:, 2] < ZBAND[1])]
            self.cloud_pub.publish(cloud_msg(np.concatenate(list(self.latest_lidar.values())), "base_link",
                                             self.get_clock().now().to_msg()))

    # ---------------------------------------------------------------- health
    def _sim_down(self, why):
        self.sim_ok, self.sim_err = False, why
        self.get_logger().error(why, throttle_duration_sec=5)

    def heartbeat(self):
        h = HealthReport()
        h.header.stamp = self.get_clock().now().to_msg()
        h.node_name = self.get_name()
        h.status = HealthReport.STATUS_OK if self.sim_ok else HealthReport.STATUS_ERROR
        h.message = "bridging the Godot sim" if self.sim_ok else self.sim_err
        self.health_pub.publish(h)


def main():
    rclpy.init()
    node = GodotBridge()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
