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
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PolygonStamped, PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from m20_msgs.action import NavigateNL, RunMission
from m20_msgs.msg import FailsafeStatus, MissionItem, RobotMode
from m20_msgs.srv import SetMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge import judge  # noqa: E402
import route  # noqa: E402  (tools/missions/route.py, copied beside this file by eval.sh)

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
    def __init__(self, action="/m20/nlnav/navigate", truth_topic="/odom_true"):
        super().__init__("nlnav_eval")
        self.lock = threading.Lock()
        self.pose = None                      # (x, y, yaw, roll, pitch, t)
        self.max_tilt = 0.0
        self.fp_age_max = 0.0
        self.fp_last = {}                     # costmap -> (last stamp seen, monotonic time it last advanced)
        self.failsafe = ""
        self.watch = False
        # "truth": the sim's /odom_true (Odometry), or on the robot its own position report /m20/vendor_pose
        # (PoseStamped): the best reference we have there, so the robot's localization error reads as the adapter's
        # lag, not as a map error
        truth_type = PoseStamped if truth_topic == "/m20/vendor_pose" else Odometry
        self.create_subscription(truth_type, truth_topic, self.on_odom, qos_profile_sensor_data)
        for k in ("local", "global"):         # both costmaps froze independently on 2026-10-13
            self.create_subscription(PolygonStamped, f"/{k}_costmap/published_footprint",
                                     (lambda k: lambda m: self.on_fp(k, m))(k), 10)
        self.create_subscription(FailsafeStatus, "/m20/failsafe", self.on_fs, 10)
        self.mode = None
        self.create_subscription(RobotMode, "/m20/mode", lambda m: setattr(self, "mode", m),
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # --action /m20/vlmnav/navigate = Real-time B (same action type; the result's detail carries the episode)
        self.client = ActionClient(self, NavigateNL, action)
        self.path_m, self._path_last = 0.0, None         # distance actually driven while watching (true pose)
        self.set_mode = self.create_client(SetMode, "/m20/set_mode")
        # localization check (SCAFFOLD for AMCL vs ground truth): the pose the stack BELIEVES (TF map->base_link,
        # which Nav2 plans from) against /odom_true at the same stamp. With ground-truth localization the two are
        # the same by construction (error 0); with AMCL this becomes the localization error during every drive.
        self.tf = Buffer()
        self._tfl = TransformListener(self.tf, self)
        self.loc_err, self.loc_yaw_err, self.loc_n = 0.0, 0.0, 0
        self.odom_err = 0.0                   # raw odometry drift: TF odom->base_link vs truth (odom is seeded at truth)
        self.initial_pose = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 1)
        # waypoint walks (cases with `route:`): straight to the mission server, no NL layer. Closest obstacle the
        # scans saw and the largest map->odom step (the stutter the first robot bridge had) are measured too.
        self.mission = ActionClient(self, RunMission, "/m20/mission/run")
        for topic in ("/scan", "/scan_front", "/scan_rear"):
            self.create_subscription(LaserScan, topic, self.on_scan, qos_profile_sensor_data)
        self.min_scan, self.map_odom_prev, self.map_odom_step = math.inf, None, 0.0
        self.leg_ends = {}                    # mission item label -> (true xy, wall time) when the server moved past it

    def on_scan(self, m):
        if self.watch:
            good = [r for r in m.ranges if m.range_min <= r < m.range_max]
            if good:
                with self.lock:
                    self.min_scan = min(self.min_scan, min(good))

    def publish_initial_pose(self, start):
        m = PoseWithCovarianceStamped()
        m.header.frame_id, m.header.stamp = "map", self.get_clock().now().to_msg()
        m.pose.pose.position.x, m.pose.pose.position.y = float(start[0]), float(start[1])
        yaw = float(start[2]) if len(start) > 2 else 0.0
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        cov = [0.0] * 36
        cov[0], cov[7], cov[35] = 0.05 ** 2, 0.05 ** 2, math.radians(3) ** 2     # placed at a known spot
        m.pose.covariance = cov
        self.initial_pose.publish(m)

    def on_odom(self, m):
        pose = getattr(m.pose, "pose", m.pose)                # Odometry.pose.pose or PoseStamped.pose
        r, p, y = rpy(pose.orientation)
        with self.lock:
            self.pose = (pose.position.x, pose.position.y, y, r, p, time.time())
            if self.watch:
                self.max_tilt = max(self.max_tilt, abs(r), abs(p))
                xy = (pose.position.x, pose.position.y)
                if self._path_last is not None:
                    self.path_m += math.hypot(xy[0] - self._path_last[0], xy[1] - self._path_last[1])
                self._path_last = xy
        if self.watch:
            self._sample_localization(m, pose, y)

    def _sample_localization(self, m, pose, true_yaw):
        try:
            t = self.tf.lookup_transform("map", "base_link", Time.from_msg(m.header.stamp))
        except Exception:
            return                                            # TF not at that stamp yet: skip this sample
        dx = t.transform.translation.x - pose.position.x
        dy = t.transform.translation.y - pose.position.y
        dyaw = rpy(t.transform.rotation)[2] - true_yaw
        dyaw = math.degrees(math.atan2(math.sin(dyaw), math.cos(dyaw)))   # wrap to [-180, 180]
        with self.lock:
            self.loc_err = max(self.loc_err, math.hypot(dx, dy))
            self.loc_yaw_err = max(self.loc_yaw_err, abs(dyaw))
            self.loc_n += 1
        try:                                                  # odometry alone, before any map correction
            o = self.tf.lookup_transform("odom", "base_link", Time.from_msg(m.header.stamp))
        except Exception:
            return
        e = math.hypot(o.transform.translation.x - pose.position.x,
                       o.transform.translation.y - pose.position.y)
        mo = (t.transform.translation.x - o.transform.translation.x, t.transform.translation.y - o.transform.translation.y)
        with self.lock:
            self.odom_err = max(self.odom_err, e)
            if self.map_odom_prev is not None:                # how far the correction jumped between two samples
                self.map_odom_step = max(self.map_odom_step, math.hypot(mo[0] - self.map_odom_prev[0], mo[1] - self.map_odom_prev[1]))
            self.map_odom_prev = mo

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
            self.loc_err, self.loc_yaw_err, self.loc_n, self.odom_err = 0.0, 0.0, 0, 0.0
            self.path_m, self._path_last = 0.0, None
            self.min_scan, self.map_odom_prev, self.map_odom_step, self.leg_ends = math.inf, None, 0.0, {}

    def stop_watch(self):
        with self.lock:
            self.watch = False
            age = max(self.fp_age_max, self._fp_staleness_now())
            out = {"max_tilt_rad": self.max_tilt, "max_footprint_age_s": round(age, 2), "failsafe": self.failsafe,
                   "path_len_m": round(self.path_m, 2)}
            if self.loc_n:
                out.update(loc_err_max_m=round(self.loc_err, 3), loc_yaw_err_max_deg=round(self.loc_yaw_err, 2),
                           odom_err_max_m=round(self.odom_err, 3), loc_samples=self.loc_n,
                           map_odom_step_max_m=round(self.map_odom_step, 3))
            if self.min_scan < math.inf:
                out["min_scan_m"] = round(self.min_scan, 3)
            return out

    def run_mission(self, wps, cancel_after_s=None, limit_s=600.0):
        """Drive a waypoint list through the mission server. -> attempt fields: status done | canceled | failed |
        rejected | timeout, legs (goal, true end pose, error, seconds), and the roll-out after a cancel."""
        goal = RunMission.Goal()
        for w in wps:
            it = MissionItem(); it.type = MissionItem.TYPE_GOTO; it.label = w.label
            it.goal.header.frame_id = "map"
            it.goal.pose.position.x, it.goal.pose.position.y = w.x, w.y
            it.goal.pose.orientation.z, it.goal.pose.orientation.w = route.quaternion_z(w.yaw)
            goal.items.append(it)
            if w.wait_s > 0:
                wt = MissionItem(); wt.type = MissionItem.TYPE_WAIT; wt.wait_seconds = float(w.wait_s); wt.label = w.label + "_wait"
                goal.items.append(wt)
        labels = [it.label for it in goal.items]
        last = {"i": 0}

        def fb(msg):                                          # the server moved to item i: item i-1 just ended
            i = msg.feedback.current_index
            if i > last["i"]:
                xy = self.xy()
                with self.lock:
                    for k in range(last["i"], i):
                        self.leg_ends.setdefault(labels[k], (xy, time.time()))
                last["i"] = i
        t0, t0_wall = time.monotonic(), time.time()
        fut = self.mission.send_goal_async(goal, feedback_callback=fb)
        while not fut.done() and time.monotonic() - t0 < 10:
            rclpy.spin_once(self, timeout_sec=0.05)
        gh = fut.result() if fut.done() else None
        if gh is None or not gh.accepted:
            return failed_attempt("rejected", "mission rejected or unanswered (armed? ASSISTED?)")
        out = {}
        if cancel_after_s:
            self.spin_for(cancel_after_s)
            xy_at_cancel = self.xy()
            gh.cancel_goal_async()
            r = self.wait_result(gh, limit_s=60.0)
            self.spin_for(2.0)
            xy_after = self.xy()
            self.spin_for(1.0)
            xy_settle = self.xy()
            out["stop_drift_m"] = round(math.hypot(xy_after[0] - xy_at_cancel[0], xy_after[1] - xy_at_cancel[1]), 3)
            out["settle_m"] = round(math.hypot(xy_settle[0] - xy_after[0], xy_settle[1] - xy_after[1]), 3)
        else:
            r = self.wait_result(gh, limit_s)
        if r is None:
            return dict(failed_attempt("timeout", f"mission still running after {limit_s:.0f} s"), **out)
        end_xy, t_end = self.xy(), time.time()
        with self.lock:
            if r.success:                                     # the last item ends with the result itself
                for lab in labels:
                    self.leg_ends.setdefault(lab, (end_xy, t_end))
            ends = dict(self.leg_ends)
        legs, t_prev = [], None
        for w in wps:
            if w.label not in ends:                           # canceled or failed before this leg: nothing to measure
                continue
            xy, t = ends[w.label]
            legs.append({"label": w.label, "goal": [w.x, w.y], "end": [round(xy[0], 3), round(xy[1], 3)],
                         "err_m": round(math.hypot(xy[0] - w.x, xy[1] - w.y), 3),
                         "t_s": round(t - (t_prev if t_prev else t0_wall), 1)})
            t_prev = t
        status = "done" if r.success else ("canceled" if "cancel" in r.detail else "failed")
        return dict(out, status=status, success=bool(r.success), detail=r.detail, completed=int(r.completed),
                    items=[{"label": w.label, "id": -1, "x": w.x, "y": w.y, "yaw": w.yaw} for w in wps], legs=legs,
                    wall_s=round(time.monotonic() - t0, 2))

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
    out = {"status": r.status, "success": bool(r.success), "interpretation": r.interpretation, "items": items,
           "candidates": cands, "latency_ms": round(float(r.latency_ms), 1), "detail": r.detail, "wall_s": round(wall, 2)}
    if r.detail.startswith("{"):                         # Real-time B: the episode summary (decisions, moves, log)
        try:
            out["episode"] = json.loads(r.detail)
        except ValueError:
            pass
    return out


def failed_attempt(status, detail):
    """Same shape as a real result, so the judge fails it on status and the report stays uniform."""
    return {"status": status, "success": False, "interpretation": "", "items": [], "candidates": [],
            "latency_ms": None, "detail": detail}


def reset_robot(node, start):
    """-> True once the ground-truth pose is within 0.3 m of the start (up to 10 s), else False.
    Then tells AMCL the robot is at the spawn (/initialpose), like an operator after placing the robot: a sim reset
    is a teleport that odometry can't see. With ground-truth localization nothing listens and this is a no-op."""
    sim({"cmd": "reset"})
    for _ in range(100):                                      # wait for the GT pose to show the spawn
        node.spin_for(0.1)
        xy = node.xy()
        if xy and math.hypot(xy[0] - start[0], xy[1] - start[1]) < 0.3:
            node.spin_for(1.0)
            node.publish_initial_pose(start)
            node.spin_for(2.0)                                # AMCL re-initializes on its next scan
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
    ap.add_argument("--action", default="/m20/nlnav/navigate",
                    help="NavigateNL action to drive: /m20/nlnav/navigate (offline) or /m20/vlmnav/navigate (Real-time B)")
    ap.add_argument("--truth-topic", default="/odom_true", help="/odom_true (sim) or /m20/vendor_pose (the robot's own report)")
    ap.add_argument("--no-reset", action="store_true", help="no sim to teleport: the operator places the robot; start_pose is only logged")
    a = ap.parse_args()
    spec = yaml.safe_load(open(a.cases))
    start = spec.get("start_pose", [0, 0, 0])
    waypoints = spec.get("kind") == "waypoints"          # every case has `route:`; all drive-level, no NL action
    if waypoints:
        a.drive = True
    cases = [c for c in spec["cases"] if bool(c.get("drive", waypoints)) == a.drive]
    if a.only:
        keep = set(a.only.split(","))
        cases = [c for c in cases if c["name"] in keep]
    objects = {}
    if a.object_map and os.path.exists(a.object_map):
        for o in json.load(open(a.object_map))["objects"]:
            objects[int(o["id"])] = (o["xyz"][0], o["xyz"][1])

    rclpy.init()
    node = Eval(a.action, a.truth_topic)
    # the robot (--no-reset): no sim API, nobody to teleport; "reset" = wait a second and have a pose
    reset = (lambda node, start: (node.spin_for(1.0), node.xy() is not None)[1]) if a.no_reset else reset_robot
    if waypoints:
        if not node.mission.wait_for_server(timeout_sec=15):
            print("ERROR: mission server /m20/mission/run not available"); sys.exit(1)
    elif not node.client.wait_for_server(timeout_sec=15):
        print(f"ERROR: action {a.action} not available"); sys.exit(1)
    node.spin_for(1.0)
    run = {"tag": a.tag, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "drive": a.drive, "config": json.loads(a.config),
           "start_pose": start, "action": a.action, "truth_topic": a.truth_topic, "cases": []}
    print(f"[eval] {len(cases)} {'drive' if a.drive else 'plan'}-level cases x{a.repeat}; resetting the robot to the spawn")
    if not reset(node, start):
        print(f"[eval] WARNING: the robot did not reach the start pose {start[:2]}; plan cases depend on it")
    if a.drive:
        armed = node.arm()
        print("[eval] arm:", armed)
        if armed.startswith("REFUSED") or armed.startswith("no ") or armed.endswith("timeout"):
            print("[eval] cannot drive: fix the commander state (tools/dev/m20 status) and rerun"); sys.exit(3)

    for c in cases:
        rec = {"name": c["name"], "text": c.get("text", c.get("route", "")), "expect": c["expect"], "attempts": []}
        if c.get("known_issue"):
            rec["known_issue"] = c["known_issue"]
        for k in range(a.repeat):
            att = {"t_start": time.time()}           # wall clock on purpose: joined to the node's log stamps
            if a.drive and not reset(node, start):
                att.update(failed_attempt("reset_failed", f"robot did not reach the start pose {start[:2]}"))
            elif a.drive:
                node.spin_for(0.5)
                node.start_watch()
                stop_after = c["expect"].get("stop_after_s")
                if "route" in c:
                    wps = route.fill_auto_yaw(route.load_route(os.path.join(os.path.dirname(a.cases), c["route"])))
                    att.update(node.run_mission(wps, cancel_after_s=c["expect"].get("cancel_after_s")))
                elif stop_after:
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
                with node.lock:
                    att["final_yaw"] = round(node.pose[2], 4) if node.pose else None
                att["start_xy"] = [round(start[0], 3), round(start[1], 3)]
            else:
                r, wall, out = node.say(c["text"], True, limit_s=60.0)
                att.update(result_dict(r, wall) if r is not None else failed_attempt(out, f"dry-run goal {out}"))
            att["t_end"] = time.time()
            att["pass"], att["why"] = judge(c["expect"], att, objects, run["config"].get("localization"))
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
