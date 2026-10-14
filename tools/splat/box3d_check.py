#!/usr/bin/env python3
"""Headless check of the detection boxes Viser draws in the MAIN view: grab the live object map
(/detections/objects), render the photoreal splat from viewpoints near the objects, and call the
viewer's own draw_objects() (projected 3D boxes in the robot camera's style, hidden when the
splat is in front). Writes a contact sheet. Run in m20_splat with --gpus all on the demo's
ROS domain:  python3 -u box3d_check.py /out/box3d_check.png
"""
import json, math, os, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splat_viewer import load_world, look_at_c2w, draw_objects, NO_DEPTH


def grab_objects(timeout=10.0):
    import rclpy
    from std_msgs.msg import String
    rclpy.init()
    n = rclpy.create_node("box3d_check")
    got = []
    n.create_subscription(String, "/detections/objects", lambda m: got.append(m.data), 10)
    t0 = time.time()
    while not got and time.time() - t0 < timeout:
        rclpy.spin_once(n, timeout_sec=0.2)
    return json.loads(got[-1])["objects"] if got else []


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/out/box3d_check.png"
    objs = grab_objects()
    print(f"[box3d] {len(objs)} confirmed objects ({sum(o.get('size') is not None for o in objs)} with 3D size)")
    scene, g, render = load_world()
    w, h, fov = 960, 540, math.radians(60)
    tiles = []
    for o in sorted([o for o in objs if o.get("size")], key=lambda o: -o["n"])[:9]:
        c = np.array(o["xyz"])
        d = -c[:2] / max(np.linalg.norm(c[:2]), 1e-6)     # view from the room-centre side
        eye = np.array([c[0] + 3.0 * d[0], c[1] + 3.0 * d[1], 1.4])
        c2w = look_at_c2w(eye, c)
        rgb, depth, alpha = render(c2w, fov, w, h)
        depth = np.where(alpha > 0.5, depth, NO_DEPTH).astype(np.float32)
        draw_objects(rgb, depth, c2w, fov, objs, (np.array([eye[0], eye[1], 0.46]), None),
                     min_score=0.5, box_scale=0.7, style_scale=0.8)   # = Viser defaults
        tiles.append(cv2.cvtColor(cv2.resize(rgb, (640, 360)), cv2.COLOR_RGB2BGR))
    if not tiles:
        # a check with no input FAILS (ENGINEERING_RULES 1): no objects = detector down or wrong domain
        print("[box3d] FAIL: no objects with a 3D size on /detections/objects (is the detector running?)")
        os._exit(1)
    cols = 3
    rows = [np.concatenate(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols])), 1)
            for i in range(0, len(tiles), cols)]
    cv2.imwrite(out, np.concatenate(rows, 0))
    print(f"[box3d] wrote {out}")
    os._exit(0)


if __name__ == "__main__":
    main()
