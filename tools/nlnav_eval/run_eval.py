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
        self.fp_last = {}                     # costmap -> (last stamp seen, monotonic time it last advanced)
        self.failsafe = ""
        self.watch = False
        self.create_subscription(Odometry, "/odom_true", self.on_odom, qos_profile_sensor_data)
        for k in ("local", "global"):         # both costmaps froze independently on 2026-10-13
            self.create_subscription(PolygonStamped, f"/{k}_costmap/published_footprint",
                                     (lambda k: lambda m: self.on_fp(k, m))(k), 10)
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

    def on_fp(self, costmap, m):
        """Staleness = how long the footprint's stamp has NOT advanced, timed on this process's monotonic clock
        (not wall clock minus stamp: WSL steps its wall clock, which once produced a -499 ms latency)."""
        stamp, now = (m.header.stamp.sec, m.header.stamp.nanosec), time.monotonic()
        with self.lock:
            last = self.fp_last.get(costmap)
            if last is None or stamp != last[0]:
                self.fp_last[costmap] = (stamp, now)
            elif self.watch:
                self.fp_age_max = max(self.fp_age_max, now - last[1])

    def _fp_staleness_now(self):
        """Also counts a footprint that stopped arriving altogether (no message = its stamp never advances)."""
        now = time.monotonic()
        return max((now - adv for _, adv in self.fp_last.values()), default=0.0)

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
            age = max(self.fp_age_max, self._fp_staleness_now())
            return {"max_tilt_rad": self.max_tilt, "max_footprint_age_s": round(age, 2), "failsafe": self.failsafe}

    def xy(self):
        with self.lock:
            return (self.pose[0], self.pose[1]) if self.pose else None

    def spin_for(self, secs):
        t = time.monotonic()
        while time.monotonic() - t < secs:
            rclpy.spin_once(self, timeout_sec=0.05)

    def send(self, text, dry_run, pick=-1, timeout_s=10.0):
        """-> accepted goal handle, or None (no answer within timeout_s, or rejected)."""
        g = NavigateNL.Goal()
        g.text, g.dry_run, g.pick_id = text, dry_run, pick
        fut = self.client.send_goal_async(g)
        t = time.monotonic()
        while not fut.done() and time.monotonic() - t < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.05)
        gh = fut.result() if fut.done() else None
        return gh if gh is not None and gh.accepted else None

    def wait_result(self, gh, limit_s, grace_s=5.0):
        """-> the action result, or None if it doesn't finish within limit_s (then cancel ONCE and wait up to
        grace_s for the cancelled result, which is also returned as None: the attempt timed out either way)."""
        res_f = gh.get_result_async()
        t = time.monotonic()
        while not res_f.done() and time.monotonic() - t < limit_s:
            rclpy.spin_once(self, timeout_sec=0.05)
        if res_f.done():
            return res_f.result().result
        gh.cancel_goal_async()
        t = time.monotonic()
        while not res_f.done() and time.monotonic() - t < grace_s:
            rclpy.spin_once(self, timeout_sec=0.05)
        return None

    def say(self, text, dry_run, pick=-1, limit_s=600.0):
        """-> (result | None, wall_s, outcome) with outcome "ok" | "rejected" | "timeout"."""
        t0 = time.monotonic()
        gh = self.send(text, dry_run, pick)
        if gh is None:
            return None, time.monotonic() - t0, "rejected"
        r = self.wait_result(gh, limit_s)
        return r, time.monotonic() - t0, ("ok" if r is not None else "timeout")

    def arm(self):
        """Make sure the commander will accept missions: leave an armed robot alone; otherwise ask for ASSISTED
        and return the commander's own reason if it refuses (e.g. 'denied: active failsafes 0x20')."""
        if self.mode is not None and self.mode.armed:
            return f"already armed (mode {self.mode.mode})"
        if not self.set_mode.wait_for_service(timeout_sec=3):
            return "no set_mode service"
        req = SetMode.Request(); req.mode, req.requester = 2, "nlnav_eval"       # 2 = ASSISTED
        f = self.set_mode.call_async(req)
        t = time.monotonic()
        while not f.done() and time.monotonic() - t < 5:
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


def failed_attempt(status, detail):
    """Same shape as a real result, so the judge fails it on status and the report stays uniform."""
    return {"status": status, "success": False, "interpretation": "", "items": [], "candidates": [],
            "latency_ms": None, "detail": detail}


def reset_robot(node, start):
    """-> True once the ground-truth pose is within 0.3 m of the start (up to 10 s), else False."""
    sim({"cmd": "reset"})
    for _ in range(100):                                      # wait for the GT pose to show the spawn
        node.spin_for(0.1)
        xy = node.xy()
        if xy and math.hypot(xy[0] - start[0], xy[1] - start[1]) < 0.3:
            node.spin_for(1.0)
            return True
    return False


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
    if not reset_robot(node, start):
        print(f"[eval] WARNING: the robot did not reach the start pose {start[:2]}; plan cases depend on it")
    if a.drive:
        armed = node.arm()
        print("[eval] arm:", armed)
        if armed.startswith("REFUSED") or armed.startswith("no ") or armed.endswith("timeout"):
            print("[eval] cannot drive: fix the commander state (tools/dev/m20 status) and rerun"); sys.exit(3)

    for c in cases:
        rec = {"name": c["name"], "text": c["text"], "expect": c["expect"], "attempts": []}
        for k in range(a.repeat):
            att = {"t_start": time.time()}           # wall clock on purpose: joined to the node's log stamps
            if a.drive and not reset_robot(node, start):
                att.update(failed_attempt("reset_failed", f"robot did not reach the start pose {start[:2]}"))
            elif a.drive:
                node.spin_for(0.5)
                node.start_watch()
                stop_after = c["expect"].get("stop_after_s")
                if stop_after:
                    # send the command; after stop_after_s send "stop" as a second goal (the node cancels the
                    # running mission); the first goal then returns canceled. Drift is measured over the next 2 s.
                    t0 = time.monotonic()
                    gh = node.send(c["text"], False)
                    if gh is None:
                        att.update(failed_attempt("rejected", "command goal rejected or unanswered"))
                    else:
                        node.spin_for(max(0.0, stop_after - (time.monotonic() - t0)))
                        xy_at_stop = node.xy()
                        r2, _, out2 = node.say("stop", False, limit_s=30.0)
                        r = node.wait_result(gh, limit_s=60.0)      # the stopped mission must end promptly
                        if r is None:
                            att.update(failed_attempt("timeout", "command still running 60 s after stop"))
                        else:
                            node.spin_for(2.0)
                            xy_after = node.xy()                     # roll-out: stop command -> +2 s
                            node.spin_for(1.0)
                            xy_settle = node.xy()                    # +2 s -> +3 s: must be still
                            att.update(result_dict(r, time.monotonic() - t0))
                            att["stop_drift_m"] = round(math.hypot(xy_after[0] - xy_at_stop[0], xy_after[1] - xy_at_stop[1]), 3) if xy_at_stop and xy_after else None
                            att["settle_m"] = round(math.hypot(xy_settle[0] - xy_after[0], xy_settle[1] - xy_after[1]), 3) if xy_after and xy_settle else None
                        att["stop_result"] = r2.status if r2 is not None else out2
                else:
                    r, wall, out = node.say(c["text"], False)
                    att.update(result_dict(r, wall) if r is not None else failed_attempt(out, f"command goal {out}"))
                node.spin_for(0.5)
                att.update(node.stop_watch())
                xy = node.xy()
                att["final_xy"] = [round(xy[0], 3), round(xy[1], 3)] if xy else None
            else:
                r, wall, out = node.say(c["text"], True, limit_s=60.0)
                att.update(result_dict(r, wall) if r is not None else failed_attempt(out, f"dry-run goal {out}"))
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
