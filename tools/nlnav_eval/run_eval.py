#!/usr/bin/env python3
"""Run the NL-navigation eval cases against the live stack (runs INSIDE docker-commander-1: needs rclpy + m20_msgs).
Driven by eval.sh; direct use:
  python3 run_eval.py --cases cases.yaml --out run.json [--repeat 3] [--drive] [--tag T] [--config '{json}']
                      [--object-map object_map.json] [--only name,name]
Plan-level cases: NavigateNL with dry_run=true from the start pose (the robot is reset first). Drive-level
cases (--drive): dry_run=false; ground truth from /odom_true (the sim's bridge), tilt and footprint ages
sampled during the drive, /m20/failsafe watched, "stop" sent after `stop_after_s` where the case says so.
Judgement: judge.py. Output: one JSON with every attempt (status, items, candidates, latency, judge verdict).
"""
import argparse
import json
import math
import os
import socket
import sys
import threading
import time

import yaml
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PolygonStamped
from m20_msgs.action import NavigateNL
from m20_msgs.msg import FailsafeStatus, RobotMode
from m20_msgs.srv import SetMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge import judge  # noqa: E402

SIM = ("127.0.0.1", 7799)


def sim(cmd: dict):
    s = socket.create_connection(SIM, timeout=10)
    s.sendall((json.dumps(cmd) + "\n").encode())
    b = b""
    while not b.endswith(b"\n"):
        b += s.recv(1 << 20)
    s.close()
    return json.loads(b)


def rpy(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class Eval(Node):
    def __init__(self):
        super().__init__("nlnav_eval")
        self.lock = threading.Lock()
        self.pose = None                      # (x, y, yaw, roll, pitch, t)
        self.max_tilt = 0.0
        self.fp_age_max = 0.0
        self.failsafe = ""
        self.watch = False
        self.create_subscription(Odometry, "/odom_true", self.on_odom, qos_profile_sensor_data)
        self.create_subscription(PolygonStamped, "/local_costmap/published_footprint", self.on_fp, 10)
        self.create_subscription(FailsafeStatus, "/m20/failsafe", self.on_fs, 10)
        self.mode = None
        self.create_subscription(RobotMode, "/m20/mode", lambda m: setattr(self, "mode", m),
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.client = ActionClient(self, NavigateNL, "/m20/nlnav/navigate")
        self.set_mode = self.create_client(SetMode, "/m20/set_mode")

    def on_odom(self, m):
        r, p, y = rpy(m.pose.pose.orientation)
        with self.lock:
            self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, y, r, p, time.time())
            if self.watch:
                self.max_tilt = max(self.max_tilt, abs(r), abs(p))

    def on_fp(self, m):
        age = time.time() - (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
        with self.lock:
            if self.watch:
                self.fp_age_max = max(self.fp_age_max, age)

    def on_fs(self, m):
        if m.active:                                          # OR of FS_* flags; 0 = none
            with self.lock:
                if self.watch:
                    self.failsafe = f"0x{m.active:x} sev {m.severity} {m.detail}".strip()

    def start_watch(self):
        with self.lock:
            self.watch, self.max_tilt, self.fp_age_max, self.failsafe = True, 0.0, 0.0, ""

    def stop_watch(self):
        with self.lock:
            self.watch = False
            return {"max_tilt_rad": self.max_tilt, "max_footprint_age_s": self.fp_age_max, "failsafe": self.failsafe}

    def xy(self):
        with self.lock:
            return (self.pose[0], self.pose[1]) if self.pose else None

    def spin_for(self, secs):
        t = time.time()
        while time.time() - t < secs:
            rclpy.spin_once(self, timeout_sec=0.05)

    def say(self, text, dry_run, pick=-1, cancel_after=None, feedback=None):
        g = NavigateNL.Goal()
        g.text, g.dry_run, g.pick_id = text, dry_run, pick
        fut = self.client.send_goal_async(g, feedback_callback=feedback)
        t = time.time()
        while not fut.done() and time.time() - t < 10:
            rclpy.spin_once(self, timeout_sec=0.05)
        gh = fut.result()
        if gh is None or not gh.accepted:
            return None, None
        res_f = gh.get_result_async()
        t0 = time.time()
        while not res_f.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.time() - t0 > 600:
                gh.cancel_goal_async()
        return res_f.result().result, time.time() - t0

    def arm(self):
        """Make sure the commander will accept missions: leave an armed robot alone; otherwise ask for ASSISTED
        and return the commander's own reason if it refuses (e.g. 'denied: active failsafes 0x20')."""
        if self.mode is not None and self.mode.armed:
            return f"already armed (mode {self.mode.mode})"
        if not self.set_mode.wait_for_service(timeout_sec=3):
            return "no set_mode service"
        req = SetMode.Request(); req.mode, req.requester = 2, "nlnav_eval"       # 2 = ASSISTED
        f = self.set_mode.call_async(req)
        t = time.time()
        while not f.done() and time.time() - t < 5:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not f.done():
            return "set_mode timeout"
        r = f.result()
        return "armed ASSISTED" if r.accepted else f"REFUSED: {r.reason}"


def result_dict(r, wall):
    items = []
    for it in r.items:
        p = it.goal.pose.position
        q = it.goal.pose.orientation
        lab = it.label
        oid = int(lab.split("#")[-1]) if "#" in lab and lab.split("#")[-1].strip().isdigit() else -1
        items.append({"label": lab, "id": oid, "x": round(p.x, 3), "y": round(p.y, 3), "yaw": round(rpy(q)[2], 3)})
    cands = [{"id": c.object_id, "label": c.cls, "score": round(float(c.score), 3), "why": c.why} for c in r.candidates[:5]]
    return {"status": r.status, "success": bool(r.success), "interpretation": r.interpretation, "items": items,
            "candidates": cands, "latency_ms": round(float(r.latency_ms), 1), "detail": r.detail, "wall_s": round(wall, 2)}


def reset_robot(node, start):
    sim({"cmd": "reset"})
    for _ in range(100):                                      # wait for the GT pose to show the spawn
        node.spin_for(0.1)
        xy = node.xy()
        if xy and math.hypot(xy[0] - start[0], xy[1] - start[1]) < 0.3:
            break
    node.spin_for(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--drive", action="store_true")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--config", default="{}")
    ap.add_argument("--object-map", default="")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    spec = yaml.safe_load(open(a.cases))
    start = spec.get("start_pose", [0, 0, 0])
    cases = [c for c in spec["cases"] if bool(c.get("drive")) == a.drive]
    if a.only:
        keep = set(a.only.split(","))
        cases = [c for c in cases if c["name"] in keep]
    objects = {}
    if a.object_map and os.path.exists(a.object_map):
        for o in json.load(open(a.object_map))["objects"]:
            objects[int(o["id"])] = (o["xyz"][0], o["xyz"][1])

    rclpy.init()
    node = Eval()
    if not node.client.wait_for_server(timeout_sec=15):
        print("ERROR: NL navigation action not available"); sys.exit(1)
    node.spin_for(1.0)
    run = {"tag": a.tag, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "drive": a.drive, "config": json.loads(a.config),
           "start_pose": start, "cases": []}
    print(f"[eval] {len(cases)} {'drive' if a.drive else 'plan'}-level cases x{a.repeat}; resetting the robot to the spawn")
    reset_robot(node, start)
    if a.drive:
        armed = node.arm()
        print("[eval] arm:", armed)
        if armed.startswith("REFUSED") or armed.startswith("no ") or armed.endswith("timeout"):
            print("[eval] cannot drive: fix the commander state (tools/dev/m20 status) and rerun"); sys.exit(3)

    for c in cases:
        rec = {"name": c["name"], "text": c["text"], "expect": c["expect"], "attempts": []}
        for k in range(a.repeat):
            att = {"t_start": time.time()}
            if a.drive:
                reset_robot(node, start)
                node.spin_for(0.5)
                node.start_watch()
                stop_after = c["expect"].get("stop_after_s")
                if stop_after:
                    # send the command; after stop_after_s send "stop" as a second goal (the node cancels the
                    # running mission); the first goal then returns canceled. Drift is measured over the next 2 s.
                    t0 = time.time()
                    g = NavigateNL.Goal(); g.text, g.dry_run, g.pick_id = c["text"], False, -1
                    fut = node.client.send_goal_async(g)
                    while not fut.done():
                        rclpy.spin_once(node, timeout_sec=0.05)
                    gh = fut.result()
                    res_f = gh.get_result_async()
                    while time.time() - t0 < stop_after:
                        rclpy.spin_once(node, timeout_sec=0.05)
                    xy_at_stop = node.xy()
                    r2, _ = node.say("stop", False)
                    while not res_f.done():
                        rclpy.spin_once(node, timeout_sec=0.05)
                    r = res_f.result().result
                    node.spin_for(2.0)
                    xy_after = node.xy()                     # roll-out: stop command -> +2 s
                    node.spin_for(1.0)
                    xy_settle = node.xy()                    # +2 s -> +3 s: must be still
                    att.update(result_dict(r, time.time() - t0))
                    att["stop_result"] = r2.status if r2 else None
                    att["stop_drift_m"] = round(math.hypot(xy_after[0] - xy_at_stop[0], xy_after[1] - xy_at_stop[1]), 3) if xy_at_stop and xy_after else None
                    att["settle_m"] = round(math.hypot(xy_settle[0] - xy_after[0], xy_settle[1] - xy_after[1]), 3) if xy_after and xy_settle else None
                else:
                    r, wall = node.say(c["text"], False)
                    if r is None:
                        att.update({"status": "rejected", "items": [], "candidates": [], "latency_ms": None, "detail": "goal rejected"})
                    else:
                        att.update(result_dict(r, wall))
                node.spin_for(0.5)
                att.update(node.stop_watch())
                xy = node.xy()
                att["final_xy"] = [round(xy[0], 3), round(xy[1], 3)] if xy else None
            else:
                r, wall = node.say(c["text"], True)
                if r is None:
                    att.update({"status": "rejected", "items": [], "candidates": [], "latency_ms": None, "detail": "goal rejected"})
                else:
                    att.update(result_dict(r, wall))
            att["t_end"] = time.time()
            att["pass"], att["why"] = judge(c["expect"], att, objects)
            rec["attempts"].append(att)
            print(f"[eval] {c['name']:<26} {'PASS' if att['pass'] else 'FAIL'}  {att['status']:<14} "
                  f"{(att.get('latency_ms') or 0):6.0f} ms  {att['why'][:70]}", flush=True)
        run["cases"].append(rec)
    json.dump(run, open(a.out, "w"), indent=1)
    n_pass = sum(all(x["pass"] for x in c["attempts"]) for c in run["cases"])
    print(f"[eval] {n_pass}/{len(run['cases'])} cases pass -> {a.out}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
