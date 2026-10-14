#!/usr/bin/env python3
"""Record keyframes for the offline semantic map (the "mapping run").

Saves a keyframe whenever the camera has moved >= KF_DIST m or turned >= KF_ANGLE deg since the last one:
  <out>/kf_00001.jpg           colour (rgb8 -> JPEG)
  <out>/kf_00001_depth.npy     depth, float16 metres (0 = no return)
  <out>/index.jsonl            one line per keyframe: id, stamp, T_map_cam (4x4, OpenCV optical frame)
  <out>/camera.json            intrinsics (from /camera/camera_info)
The camera pose comes from TF (map -> camera_optical_link) at the image stamp, so the object-map builder
needs no mount maths and works for any camera. Colour and depth are paired on identical header stamps.

Env: SEMMAP_OUT (required), KF_DIST (0.4), KF_ANGLE (15), CAM_TOPIC (/camera/image_raw),
DEPTH_TOPIC (/camera/depth), INFO_TOPIC (/camera/camera_info), CAM_FRAME (camera_optical_link).
"""
import json
import math
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from frozen_frames import FrozenFrameGuard          # same directory (the script's dir is on sys.path)

OUT = os.environ["SEMMAP_OUT"]
KF_DIST = float(os.environ.get("KF_DIST", 0.4))
KF_ANGLE = math.radians(float(os.environ.get("KF_ANGLE", 15)))
CAM_FRAME = os.environ.get("CAM_FRAME", "camera_optical_link")


def tf_to_mat(t):
    q, p = t.transform.rotation, t.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, (p.x, p.y, p.z)
    return T


def moved_enough(Ta, Tb):
    if Ta is None:
        return True
    d = np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])
    cosang = (np.trace(Ta[:3, :3].T @ Tb[:3, :3]) - 1) / 2
    return d >= KF_DIST or math.acos(max(-1.0, min(1.0, cosang))) >= KF_ANGLE


class Recorder(Node):
    def __init__(self):
        super().__init__("semmap_keyframe_recorder")
        os.makedirs(OUT, exist_ok=True)
        self.tf = Buffer()
        self._tfl = TransformListener(self.tf, self)
        self.pending, self.last_T, self.n, self.info_saved = {}, None, 0, False
        self.frozen = FrozenFrameGuard()
        self.index = open(os.path.join(OUT, "index.jsonl"), "a")
        self.create_subscription(Image, os.environ.get("CAM_TOPIC", "/camera/image_raw"),
                                 lambda m: self.on_frame("c", m), qos_profile_sensor_data)
        self.create_subscription(Image, os.environ.get("DEPTH_TOPIC", "/camera/depth"),
                                 lambda m: self.on_frame("d", m), qos_profile_sensor_data)
        self.create_subscription(CameraInfo, os.environ.get("INFO_TOPIC", "/camera/camera_info"),
                                 self.on_info, qos_profile_sensor_data)
        self.get_logger().info(f"recording keyframes to {OUT} (every {KF_DIST} m / {math.degrees(KF_ANGLE):.0f} deg)")

    def on_info(self, m):
        if not self.info_saved:
            json.dump({"width": m.width, "height": m.height, "fx": m.k[0], "fy": m.k[4], "cx": m.k[2], "cy": m.k[5],
                       "frame": CAM_FRAME}, open(os.path.join(OUT, "camera.json"), "w"), indent=1)
            self.info_saved = True

    def on_frame(self, kind, m):
        key = (m.header.stamp.sec, m.header.stamp.nanosec)
        slot = self.pending.setdefault(key, {})
        slot[kind] = m
        if len(self.pending) > 8:
            self.pending.pop(min(self.pending))
        if "c" in slot and "d" in slot:
            self.pending.pop(key)
            self.consider(slot["c"], slot["d"])

    def consider(self, cm, dm):
        try:
            T = tf_to_mat(self.tf.lookup_transform("map", CAM_FRAME, Time.from_msg(cm.header.stamp)))
        except Exception:
            return
        if not moved_enough(self.last_T, T):
            return
        if self.frozen.is_frozen(cm.data):               # see frozen_frames.py
            self.get_logger().warn("camera image unchanged while the robot moved: frozen frame, not saved",
                                   throttle_duration_sec=5.0)
            return
        self.frozen.saved(cm.data)
        self.last_T = T
        self.n += 1
        kid = f"kf_{self.n:05d}"
        rgb = np.frombuffer(bytes(cm.data), np.uint8).reshape(cm.height, cm.width, 3)
        cv2.imwrite(os.path.join(OUT, kid + ".jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        depth = np.frombuffer(bytes(dm.data), np.float32).reshape(dm.height, dm.width)
        np.save(os.path.join(OUT, kid + "_depth.npy"), depth.astype(np.float16))
        stamp = cm.header.stamp.sec + cm.header.stamp.nanosec * 1e-9
        self.index.write(json.dumps({"id": kid, "stamp": stamp, "T_map_cam": T.round(5).tolist()}) + "\n")
        self.index.flush()
        if self.n % 25 == 0:
            self.get_logger().info(f"{self.n} keyframes")


def main():
    rclpy.init()
    rclpy.spin(Recorder())


if __name__ == "__main__":
    main()
