#!/usr/bin/env python3
"""Fake the real M20's localization report in the sim: the robot's NOS computer pushes its map pose about twice a
second (1002/7 LocationStatus), late and a little noisy. This node turns the sim's true pose (/odom_true, map ->
base_link_true) into that report on /m20/vendor_pose (PoseStamped, frame map) so map_tf_adapter.py and Nav2 run
exactly as they will on the robot.

Numbers: 2 Hz and ~0.1 s late (measured on Dog1, 2026-09-10 / 2026-10-16); jitter 1 cm (assumed: the robot's
localization is lidar-based; two consecutive reports then differ by ~0.014 m per axis, so a 0.10 m map->odom step
limit still catches the first bridge's stutter of 0.15 m per report at 0.3 m/s). Measure the real jitter on Dog1
(step 3 of the walk) and set VENDOR_POSE_NOISE_M to it.
Env: VENDOR_POSE_HZ (2), VENDOR_POSE_DELAY_S (0.1), VENDOR_POSE_NOISE_M (0.01). The stamp is the TRUE time of the
pose (delay = publish later), the way the robot stamps its own report.
"""
import collections
import os
import random

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

HZ = float(os.environ.get("VENDOR_POSE_HZ", 2.0))
DELAY = float(os.environ.get("VENDOR_POSE_DELAY_S", 0.1))
NOISE = float(os.environ.get("VENDOR_POSE_NOISE_M", 0.01))


class VendorPoseSim(Node):
    def __init__(self):
        super().__init__("vendor_pose_sim")
        self.recent = collections.deque(maxlen=200)         # (t_seconds, Odometry): ~4 s of truth at 50 Hz
        self.create_subscription(Odometry, "/odom_true", self.on_true, qos_profile_sensor_data)
        self.pub = self.create_publisher(PoseStamped, "/m20/vendor_pose", 10)
        self.create_timer(1.0 / HZ, self.tick)
        self.get_logger().info(f"vendor pose sim: {HZ} Hz, {DELAY} s late, {NOISE} m noise -> /m20/vendor_pose")

    def on_true(self, m: Odometry):
        self.recent.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m))

    def tick(self):
        if not self.recent:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        # the newest truth sample at least DELAY old: what the robot would be reporting right now
        old = [s for s in self.recent if s[0] <= now - DELAY]
        if not old:
            return
        _, m = old[-1]
        p = PoseStamped()
        p.header.stamp, p.header.frame_id = m.header.stamp, "map"
        p.pose = m.pose.pose
        p.pose.position.x += random.gauss(0.0, NOISE)
        p.pose.position.y += random.gauss(0.0, NOISE)
        self.pub.publish(p)


def main():
    rclpy.init()
    rclpy.spin(VendorPoseSim())


if __name__ == "__main__":
    main()
