#!/usr/bin/env python3
"""map -> odom from the robot's own localization report, the job AMCL does when it runs.

Nav2 needs two transforms: odom -> base_link (smooth, 10-50 Hz, from odometry) and map -> odom (slow, the
correction between where odometry thinks the robot is and where the map says it is). The M20 reports its map pose
about twice a second (/m20/vendor_pose, PoseStamped in `map`, stamped with the time the pose was true). At each NEW
report this node looks up odom -> base_link AT THAT STAMP in TF and sets

    T_map_odom = T_map_base(report) * inverse(T_odom_base(report stamp))

then re-publishes that transform at 10 Hz, stamped TRANSFORM_TOLERANCE_S into the future like AMCL does: a lookup at
a sensor stamp a few ms newer than the last broadcast would otherwise fail (measured 2026-11-09: the eval's 50 Hz
lookups at /odom_true stamps succeeded twice in a 7 s drive with "now" stamps). Recomputing on
every tick from the CACHED report (what the first bridge did) made the map pose stand still for half a second and
jump ~0.2 m at 0.4 m/s; comparing odometry from the same instant as the report is what keeps it continuous.

The same file will run on the robot against the bridge's LocationStatus; keep it free of sim-only imports.
"""
import math

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformBroadcaster, TransformListener

TRANSFORM_TOLERANCE_S = 0.2        # the stamp lead on map->odom (AMCL's default transform_tolerance is 1.0 s)


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def compose(a, b):
    """a then b: (x, y, yaw) triples."""
    ax, ay, at = a
    bx, by, bt = b
    c, s = math.cos(at), math.sin(at)
    return ax + c * bx - s * by, ay + s * bx + c * by, math.atan2(math.sin(at + bt), math.cos(at + bt))


def invert(a):
    ax, ay, at = a
    c, s = math.cos(at), math.sin(at)
    return -(c * ax + s * ay), -(-s * ax + c * ay), -at


class MapTfAdapter(Node):
    def __init__(self):
        super().__init__("map_tf_adapter")
        self.tf = Buffer()
        self._listener = TransformListener(self.tf, self)
        self.bc = TransformBroadcaster(self)
        self.map_odom = None                                  # (x, y, yaw) once the first report has been matched
        self.n_updates, self.max_step = 0, 0.0
        self.create_subscription(PoseStamped, "/m20/vendor_pose", self.on_report, 10)
        self.create_timer(0.1, self.tick)
        self.create_timer(10.0, self.report)

    def on_report(self, p: PoseStamped):
        try:
            t = self.tf.lookup_transform("odom", "base_link", Time.from_msg(p.header.stamp))
        except Exception as e:                                # odometry not yet in the buffer at that stamp
            self.get_logger().warn(f"no odom->base_link at the report's stamp: {e}", throttle_duration_sec=5.0)
            return
        odom_base = (t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation))
        map_base = (p.pose.position.x, p.pose.position.y, yaw_of(p.pose.orientation))
        new = compose(map_base, invert(odom_base))
        if self.map_odom is not None:
            self.max_step = max(self.max_step, math.hypot(new[0] - self.map_odom[0], new[1] - self.map_odom[1]))
        self.map_odom, self.n_updates = new, self.n_updates + 1

    def tick(self):                                           # single-threaded executor: no overlap with on_report
        mo = self.map_odom
        if mo is None:
            return
        tf = TransformStamped()
        stamp = (self.get_clock().now() + Duration(seconds=TRANSFORM_TOLERANCE_S)).to_msg()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = stamp, "map", "odom"
        tf.transform.translation.x, tf.transform.translation.y = mo[0], mo[1]
        tf.transform.rotation.z, tf.transform.rotation.w = math.sin(mo[2] / 2), math.cos(mo[2] / 2)
        self.bc.sendTransform(tf)

    def report(self):
        self.get_logger().info(f"map->odom: {self.n_updates} reports matched, largest step {self.max_step:.3f} m")


def main():
    rclpy.init()
    rclpy.spin(MapTfAdapter())


if __name__ == "__main__":
    main()
