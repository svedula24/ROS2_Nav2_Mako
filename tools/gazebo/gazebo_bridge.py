#!/usr/bin/env python3
"""ROS 2 glue between Gazebo (ros_gz_bridge's /gz/* topics) and the M20 stack: the same contract as
tools/godot/godot_bridge.py, so commander, Nav2, mission server, nlnav, the search and the detector run unchanged.
Runs in the autonomy container (needs m20_msgs). Published (wall-clock stamps, like TF; map = Gazebo's world frame):
  /odom (odom -> base_link) + TF odom->base_link   GZ_ODOM=truth (default) | drift (godot_io.DriftOdom, for AMCL)
  /odom_true (map -> base_link_true, full 3D)      the true world pose of base_link (detector, Viser, nlnav)
  /IMU_DATA, /JOINTS_DATA                          the commander's tip-over failsafe; joints = the standing stance
  /LIDAR/POINTS (PointCloud2 xyz, base_link)       both lidars merged, z-band filtered (-> /scan)
  /camera/obstacle_points (PointCloud2, base_link) the depth camera as points (GZ_DEPTH_OBSTACLES=1)
  /camera/image_raw, /camera/depth, /camera/camera_info   one stamp per frame pair; depth 32FC1 m, 0 = no return
  /m20/health
Consumed: /cmd_vel (GZ_CMD_TOPIC; Real-time B uses /cmd_vel_safe) -> /gz/cmd_vel ONLY while /m20/mode says armed,
  zero when disarmed or stale, capped at GZ_VMAX / GZ_WMAX. Gazebo's VelocityControl is exact: no yaw trim.
Env: GZ_ODOM (truth|drift), GZ_ODOM_SEED, GZ_ODOM_SPEED_NOISE, GZ_ODOM_GYRO_BIAS, GZ_ODOM_SCALE, GZ_VMAX (0.5),
  GZ_WMAX (1.5), GZ_CLOUD_ZBAND (-0.45,0.70), GZ_DEPTH_OBSTACLES (1), GZ_CMD_TOPIC (/cmd_vel),
  GZ_CAM_MOUNT (0.37646,0,0.03738), GZ_LIDAR_MOUNTS (front/rear xyz), GZ_BASE_Z (0.43: model origin -> base_link).
"""
import array
import collections
import math
import os
import sys
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2, PointField
from tf2_msgs.msg import TFMessage
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from drdds.msg import ImuData, JointsData
from m20_msgs.msg import HealthReport, RobotMode

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "godot"))
import godot_io as gio  # noqa: E402  (DriftOdom, depth_points, quat helpers: no Godot in them)

ODOM_MODE = os.environ.get("GZ_ODOM", "truth")
VMAX = float(os.environ.get("GZ_VMAX", 0.5))
WMAX = float(os.environ.get("GZ_WMAX", 1.5))
ZBAND = tuple(float(v) for v in os.environ.get("GZ_CLOUD_ZBAND", "-0.45,0.70").split(","))
DEPTH_OBSTACLES = os.environ.get("GZ_DEPTH_OBSTACLES", "1") == "1"
CMD_TOPIC = os.environ.get("GZ_CMD_TOPIC", "/cmd_vel")
CAM_MOUNT = tuple(float(v) for v in os.environ.get("GZ_CAM_MOUNT", "0.37646,0,0.03738").split(","))
LIDAR_MOUNTS = {"front": (0.32028, 0.0, -0.013), "rear": (-0.32028, 0.0, -0.013)}
BASE_Z = float(os.environ.get("GZ_BASE_Z", 0.43))
CMD_STALE_S = 0.5
RESEED_JUMP_M = 0.3
# Sensor messages carry a TF stamp this much OLDER than the newest one (GZ_SENSOR_LAG, s). With the newest stamp the
# costmaps' tf2 MessageFilters still sometimes wait (the TF arrives on another topic a few ms later), and that wait +
# timeout timer race froze the planner's TF buffer twice in Gazebo on 2026-10-26 (every plan started at the robot's old
# pose) and twice on Godot in October. A stamp 100 ms old is always already in the buffer: no wait, no timer, no race.
# Cost: marks lag by v * lag (5 cm at 0.5 m/s).
SENSOR_LAG_S = float(os.environ.get("GZ_SENSOR_LAG", 0.1))
STANCE = [0.0, -0.7, 1.4, 0.0] * 2 + [0.0, 0.7, -1.4, 0.0] * 2       # 16 joints, hipx/hipy/knee/wheel per leg


def u8(arr) -> array.array:
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


def cloud_xyz(m: PointCloud2) -> np.ndarray:
    """PointCloud2 (any float32 x/y/z layout) -> (N, 3) float32, non-finite rows dropped."""
    off = {f.name: f.offset for f in m.fields}
    buf = np.frombuffer(bytes(m.data), dtype=np.uint8).reshape(-1, m.point_step)
    xyz = np.stack([buf[:, off[k]:off[k] + 4].copy().view(np.float32)[:, 0] for k in "xyz"], 1)
    return xyz[np.isfinite(xyz).all(1)]


def image_np(m: Image, dtype, channels):
    a = np.frombuffer(bytes(m.data), dtype=dtype)
    return a.reshape(m.height, m.width, channels) if channels > 1 else a.reshape(m.height, m.width)


class GazeboBridge(Node):
    def __init__(self):
        super().__init__("m20_gazebo_bridge")
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
        self.gz_cmd_pub = self.create_publisher(Twist, "/gz/cmd_vel", 10)
        self.tf_bc = TransformBroadcaster(self)
        self._static_tf()

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(RobotMode, "/m20/mode", self.on_mode, latched)
        self.create_subscription(Twist, CMD_TOPIC, self.on_cmd_vel, 10)
        self.create_subscription(TFMessage, "/gz/pose", self.on_pose, 50)
        self.create_subscription(Odometry, "/gz/odometry", self.on_gz_odom, 50)
        self.create_subscription(Imu, "/gz/imu", self.on_imu, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, "/gz/lidar_front", lambda m: self.on_lidar("front", m), qos_profile_sensor_data)
        self.create_subscription(PointCloud2, "/gz/lidar_rear", lambda m: self.on_lidar("rear", m), qos_profile_sensor_data)
        self.create_subscription(Image, "/gz/image", self.on_image, qos_profile_sensor_data)
        self.create_subscription(Image, "/gz/depth", self.on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/gz/camera_info", self.on_info, qos_profile_sensor_data)

        self.armed, self.cmd, self.cmd_t, self.sent_zero = False, (0.0, 0.0), 0.0, True
        self.gyro, self.accel, self.v_forward = (0.0, 0.0, 0.0), (0.0, 0.0, 9.81), 0.0
        self.state_stamp, self.last_pose_t = None, 0.0
        self.stamp_hist = collections.deque()       # (monotonic, stamp) of published TFs, oldest first
        self.K = None
        self.drift = (gio.DriftOdom(seed=int(os.environ.get("GZ_ODOM_SEED", 0)),
                                    speed_noise=float(os.environ.get("GZ_ODOM_SPEED_NOISE", 0.02)),
                                    gyro_bias=float(os.environ.get("GZ_ODOM_GYRO_BIAS", 0.002)),
                                    scale=float(os.environ.get("GZ_ODOM_SCALE", 1.0)))
                      if ODOM_MODE == "drift" else None)
        self._last_true_xy, self._last_odom_t = None, None
        self.lock = threading.Lock()
        self.latest_lidar = {}
        self.frames = {"c": {}, "d": {}}            # gz stamp (ns) -> message
        self.create_timer(0.1, self.cmd_tick)
        self.create_timer(1.0, self.heartbeat)
        self.get_logger().info(f"Gazebo bridge up: odom={ODOM_MODE}, cmd in {CMD_TOPIC}, depth obstacles={DEPTH_OBSTACLES}")

    def _static_tf(self):
        self.static_bc = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = "base_link", "camera_optical_link"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (float(v) for v in CAM_MOUNT)
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
            return
        vx, wz = self.cmd if live else (0.0, 0.0)
        t = Twist()
        t.linear.x = float(max(-VMAX, min(VMAX, vx)))
        t.angular.z = float(max(-WMAX, min(WMAX, wz)))
        self.gz_cmd_pub.publish(t)
        self.sent_zero = not live

    # ---------------------------------------------------------------- pose / imu / joints
    def on_gz_odom(self, m: Odometry):
        self.v_forward = float(m.twist.twist.linear.x)

    def on_imu(self, m: Imu):
        self.gyro = (m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z)
        self.accel = (m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z)

    def on_pose(self, m: TFMessage):
        """gz PosePublisher: the model's world pose (the model origin sits on the floor, base_link BASE_Z above)."""
        tr = None
        for t in m.transforms:
            if t.child_frame_id in ("m20", "m20/base_link") or t.child_frame_id.endswith("base_link"):
                tr = t
        if tr is None and m.transforms:
            tr = m.transforms[0]
        if tr is None:
            return
        q = (tr.transform.rotation.x, tr.transform.rotation.y, tr.transform.rotation.z, tr.transform.rotation.w)
        R = gio.quat_to_mat(*q)
        p = np.array([tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z])
        if tr.child_frame_id == "m20":                        # model origin -> base_link
            p = p + R @ np.array([0.0, 0.0, BASE_Z])
        self.publish_state(p, q, R)

    def publish_state(self, p, q, R):
        now = self.get_clock().now().to_msg()
        px, py, pz = (float(v) for v in p)
        roll, pitch, yaw = gio.quat_to_rpy(*q)
        ox, oy, oyaw = self._odom_pose(px, py, yaw, self.v_forward)
        od = Odometry()
        od.header.stamp, od.header.frame_id, od.child_frame_id = now, "odom", "base_link"
        od.pose.pose.position.x, od.pose.pose.position.y = float(ox), float(oy)
        od.pose.pose.orientation.z, od.pose.pose.orientation.w = float(math.sin(oyaw / 2)), float(math.cos(oyaw / 2))
        od.twist.twist.linear.x, od.twist.twist.angular.z = float(self.v_forward), float(self.gyro[2])
        self.odom_pub.publish(od)
        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = now, "odom", "base_link"
        tf.transform.translation.x, tf.transform.translation.y = float(ox), float(oy)
        tf.transform.rotation.z, tf.transform.rotation.w = od.pose.pose.orientation.z, od.pose.pose.orientation.w
        self.tf_bc.sendTransform(tf)
        self.state_stamp = now                      # sensors may use this stamp from here on (TF covers it)
        self.stamp_hist.append((time.monotonic(), now))
        while len(self.stamp_hist) > 2 and time.monotonic() - self.stamp_hist[1][0] >= SENSOR_LAG_S:
            self.stamp_hist.popleft()               # keep the newest stamp that is at least SENSOR_LAG_S old at [0]

        ot = Odometry()
        ot.header.stamp, ot.header.frame_id, ot.child_frame_id = now, "map", "base_link_true"
        ot.pose.pose.position.x, ot.pose.pose.position.y, ot.pose.pose.position.z = px, py, pz
        o = ot.pose.pose.orientation
        o.x, o.y, o.z, o.w = (float(v) for v in q)
        self.true_pub.publish(ot)

        im = ImuData()
        im.header.stamp = now
        d = im.data
        d.roll, d.pitch, d.yaw = float(roll), float(pitch), float(yaw)
        d.omega_x, d.omega_y, d.omega_z = (float(v) for v in self.gyro)
        d.acc_x, d.acc_y, d.acc_z = (float(v) for v in self.accel)
        self.imu_pub.publish(im)
        jd = JointsData()
        jd.header.stamp = now
        for i in range(16):
            jd.data.joints_data[i].position = float(STANCE[i])
        self.jd_pub.publish(jd)
        self.last_pose_t = time.monotonic()

    def _odom_pose(self, px, py, yaw, v_forward):
        if self.drift is None:
            return px, py, yaw
        now = time.monotonic()
        jumped = (self._last_true_xy is not None and
                  math.hypot(px - self._last_true_xy[0], py - self._last_true_xy[1])
                  > RESEED_JUMP_M + 1.0 * (now - self._last_odom_t))
        if self.drift.pose is None or jumped:
            self.drift.reset(px, py, yaw)
            self.get_logger().info(f"odometry (drift) seeded at the true pose ({px:.2f}, {py:.2f}, {yaw:.2f})"
                                   + (" after a sim reset" if jumped else ""))
            pose = self.drift.pose
        else:
            pose = self.drift.step(v_forward, yaw, now - self._last_odom_t)
        self._last_true_xy, self._last_odom_t = (px, py), now
        return pose

    def sensor_stamp(self):
        """A TF stamp at least SENSOR_LAG_S old (see the constant), else the newest, else now."""
        if self.stamp_hist and time.monotonic() - self.stamp_hist[0][0] >= SENSOR_LAG_S:
            return self.stamp_hist[0][1]
        return self.state_stamp or self.get_clock().now().to_msg()

    # ---------------------------------------------------------------- lidar
    def on_lidar(self, name, m: PointCloud2):
        pts = cloud_xyz(m) + np.asarray(LIDAR_MOUNTS[name], np.float32)
        with self.lock:
            self.latest_lidar[name] = pts[(pts[:, 2] > ZBAND[0]) & (pts[:, 2] < ZBAND[1])]
            if len(self.latest_lidar) < 2 and name == "front":
                return                                   # wait for the rear once, then publish on every front sweep
            merged = np.concatenate(list(self.latest_lidar.values()))
        if name == "front":
            self.cloud_pub.publish(cloud_msg(merged, "base_link", self.sensor_stamp()))

    # ---------------------------------------------------------------- camera
    def on_info(self, m: CameraInfo):
        self.K = (m.k[0], m.k[4], m.k[2], m.k[5], m.width, m.height)

    @staticmethod
    def _key(m):
        return m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec

    def on_image(self, m: Image):
        self._pair("c", m)

    def on_depth(self, m: Image):
        self._pair("d", m)

    def _pair(self, kind, m):
        with self.lock:
            self.frames[kind][self._key(m)] = m
            for d in self.frames.values():
                while len(d) > 4:
                    d.pop(min(d))
            both = sorted(set(self.frames["c"]) & set(self.frames["d"]))
            if not both:
                return
            k = both[-1]
            c, d = self.frames["c"].pop(k), self.frames["d"].pop(k)
        self.publish_camera(c, d)

    def publish_camera(self, c: Image, d: Image):
        if self.K is None:
            return
        stamp = self.sensor_stamp()
        img = Image()
        img.header.stamp, img.header.frame_id = stamp, "camera_optical_link"
        img.height, img.width, img.encoding, img.step = c.height, c.width, "rgb8", c.width * 3
        img.data = c.data if c.encoding == "rgb8" else u8(image_np(c, np.uint8, 3)[:, :, :3])
        depth = image_np(d, np.float32, 1).copy()
        depth[~np.isfinite(depth)] = 0.0                 # Godot contract: 0 = no return
        dm = Image()
        dm.header = img.header
        dm.height, dm.width, dm.encoding, dm.step = d.height, d.width, "32FC1", d.width * 4
        dm.data = u8(depth)
        fx, fy, cx, cy, w, h = self.K
        info = CameraInfo()
        info.header = img.header
        info.width, info.height, info.distortion_model = w, h, "plumb_bob"
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.d = [0.0] * 5
        self.img_pub.publish(img)
        self.depth_pub.publish(dm)
        self.info_pub.publish(info)
        if DEPTH_OBSTACLES:
            pts = gio.depth_points(depth, fx, fy, cx, cy, CAM_MOUNT, stride=4, rmax=4.5)
            self.cam_cloud_pub.publish(cloud_msg(pts, "base_link", stamp))

    # ---------------------------------------------------------------- health
    def heartbeat(self):
        ok = time.monotonic() - self.last_pose_t < 2.0
        h = HealthReport()
        h.header.stamp = self.get_clock().now().to_msg()
        h.node_name = self.get_name()
        h.status = HealthReport.STATUS_OK if ok else HealthReport.STATUS_ERROR
        h.message = "bridging Gazebo" if ok else "no /gz/pose from Gazebo for 2 s"
        self.health_pub.publish(h)


def main():
    rclpy.init()
    node = GazeboBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
