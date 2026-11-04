"""m20_vlmnav search: VLFM-style object search on an unseen site (no maps given). Action /m20/vlfm/navigate
(NavigateNL type; the text is the TARGET PHRASE, e.g. "refrigerator").

Every tick the camera frame is scored by the image-text matcher ("Seems like there is a {target} ahead.") and painted
onto the VALUE MAP (value_map.py, VLFM's math); the exploration costmap (Nav2's global costmap, M20_EXPLORE_MAP) gives
the FRONTIERS (frontiers.py); the best frontier by value, with VLFM's hysteresis, becomes a Nav2 goal through the
mission server. The live OWLv2 detector (tools/splat/detector_node.py, told the target through /detections/query)
ends the search: a confirmed object of the target class -> approach pose (approach.py) -> drive -> done.

Inputs:  /camera/image_raw (rgb8), /camera/depth (32FC1 m), /odom (the believed pose; map->odom is identity here),
         /m20/mode, /detections/objects (JSON), /global_costmap/get_costmap (service)
Params:  itm_url (http://127.0.0.1:8091), log_dir, max_seconds (300), score_hz (2), plan_hz (1), fov_deg (90),
         fresh (True: clear the exploration + object maps first), look_around (True: VLFM's initial full turn),
         empty_ticks (5: not_found only after this many frontier-less ticks in a row), stuck_s / stuck_m (30 s / 1.0 m:
         no progress = the robot stayed within stuck_m of its whole last stuck_s; the goal is given up, 3 such = "stuck"),
         commit_s (10: a goal is kept at least this long before a better-valued frontier may replace it)
Outputs: /detections/query (the target phrase), RunMission goals, per episode <log_dir>/<stamp>/{steps.jsonl,
         summary.json, map_NNN.png, objects_seen.json}
"""
import base64
import io
import json
import math
import os
import threading
import time
import urllib.request

import numpy as np
import rclpy
from PIL import Image as PILImage
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

from m20_msgs.action import NavigateNL, RunMission
from m20_msgs.msg import MissionItem, RobotMode
from nav2_msgs.action import Spin
from nav2_msgs.srv import ClearEntireCostmap, GetCostmap
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, String

from m20_nlnav.approach import approach_pose
from m20_nlnav.world import grid_from_costmap

from . import behaviors
from . import frontiers as fr
from . import progress
from .episode import terminal_action
from .instruction import MOVES, OBJECT, classify
from .targets import same_class
from .value_map import PROMPT, ValueMap, depth_row_from_image

STATUS = {"done": "done", "not_found": "not_found", "timeout": "error", "canceled": "canceled", "mode": "vetoed",
          "crash": "error", "no_map": "error", "itm_error": "error", "stuck": "error",
          "not_understood": "not_understood", "blocked": "error", "move_error": "error"}
MAX_STUCK = 3                       # give up the episode after this many goals abandoned for no progress
ITM_MAX_FAILURES = 5




def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def jpeg_b64(rgb, quality=85):
    buf = io.BytesIO()
    PILImage.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def goto_item(label, x, y, yaw):
    it = MissionItem()
    it.type, it.label = MissionItem.TYPE_GOTO, label
    it.goal.header.frame_id = "map"
    it.goal.pose.position.x, it.goal.pose.position.y = float(x), float(y)
    it.goal.pose.orientation.z, it.goal.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return it


class SearchNode(Node):
    def __init__(self):
        super().__init__("m20_vlfm_search")
        for name, default in (("itm_url", "http://127.0.0.1:8091"), ("log_dir", os.path.expanduser("~/m20_sim/vlfm")),
                              ("max_seconds", 300.0), ("score_hz", 2.0), ("plan_hz", 1.0), ("fov_deg", 90.0),
                              ("max_depth", 5.0), ("min_depth", 0.5), ("fusion", "max_confidence"), ("fresh", True),
                              ("look_around", True), ("empty_ticks", 5), ("stuck_s", 30.0), ("stuck_m", 1.0),
                              ("commit_s", 10.0), ("target_min_n", 5), ("target_min_score", 0.45),
                              ("move_speed", 0.3)):
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value
        self.itm_url, self.log_dir, self.max_seconds = p("itm_url"), p("log_dir"), float(p("max_seconds"))
        self.fresh = bool(p("fresh"))          # every search starts from nothing: exploration map + object map cleared
        self.look_around = bool(p("look_around"))   # VLFM's initial full turn: every nearby frontier gets a value first
        self.empty_ticks = int(p("empty_ticks"))    # not_found only after this many ticks in a row without a frontier
        # A robot that stayed within stuck_m of everywhere it was during the last stuck_s has made no progress: the
        # goal is given up and blacklisted, and after MAX_STUCK such goals the episode ends "stuck". Nav2 keeps a
        # boxed-in robot "running" through recoveries ("No valid trajectories", follow_path aborting, back-ups and
        # spins that move it 0.3-0.5 m) until the episode clock runs out otherwise (two 300 s timeouts, 2026-10-19).
        self.stuck_s, self.stuck_m = float(p("stuck_s")), float(p("stuck_m"))
        # a goal is kept at least commit_s before a better-valued frontier may replace it (a retarget of the same
        # frontier is always allowed): our scorer is noisier than VLFM's setting, and while Nav2 spins through
        # recoveries the values near the goal repaint, the 0.01 hysteresis lets go and the search thrashed
        # (54 switches in 180 s standing still)
        self.commit_s = float(p("commit_s"))
        # the detector confirms an object at 3 sightings; the SEARCH ends only on a firmer one (a filing cabinet was
        # "whiteboard" at 0.58 after 3 sightings; real targets reach n 20-30 and 0.7-0.9 within seconds)
        self.target_min_n, self.target_min_score = int(p("target_min_n")), float(p("target_min_score"))
        self.move_speed = float(p("move_speed"))    # m/s for a dispatched forward move (stage 1's executor speed)
        self.score_dt, self.plan_dt = 1.0 / float(p("score_hz")), 1.0 / float(p("plan_hz"))
        self.fov, self.max_depth, self.min_depth, self.fusion = math.radians(float(p("fov_deg"))), float(p("max_depth")), float(p("min_depth")), p("fusion")

        self.cb = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self.rgb, self.depth, self.frame_t = None, None, 0.0
        self.pose = None                       # believed (x, y, yaw)
        self.true_pose = None                  # evals only
        self.mode_ok = False
        self.detections = {"frame": [], "objects": []}
        self.det_fresh = True                  # False from a reset until the detector has published an EMPTY map
        self.running = False
        self.goal_lock = threading.Lock()
        self.active = None                     # mission goal handle in flight

        self.create_subscription(Image, "/camera/image_raw", self.on_rgb, qos_profile_sensor_data, callback_group=self.cb)
        self.create_subscription(Image, "/camera/depth", self.on_depth, qos_profile_sensor_data, callback_group=self.cb)
        self.create_subscription(Odometry, "/odom", self.on_odom, qos_profile_sensor_data, callback_group=self.cb)
        self.create_subscription(Odometry, "/odom_true", self.on_true, qos_profile_sensor_data, callback_group=self.cb)
        self.create_subscription(String, "/detections/objects", self.on_det, 10, callback_group=self.cb)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(RobotMode, "/m20/mode", self.on_mode, latched, callback_group=self.cb)
        self.query_pub = self.create_publisher(String, "/detections/query", latched)
        self.reset_pub = self.create_publisher(Empty, "/detections/reset", 10)
        self.get_costmap = self.create_client(GetCostmap, "/global_costmap/get_costmap", callback_group=self.cb)
        self.clear_costmap = self.create_client(ClearEntireCostmap, "/global_costmap/clear_entirely_global_costmap",
                                                callback_group=self.cb)
        self.mission = ActionClient(self, RunMission, "/m20/mission/run", callback_group=self.cb)
        self.behaviors = behaviors.make_clients(self, self.cb)    # the executor shared with the VLM node
        self.spin = self.behaviors["turn"]
        self.active_move = None                # behavior goal handle of a dispatched move in flight
        self.spinning = None                   # (goal handle, result future) of the initial turn while it runs
        self.server = ActionServer(self, NavigateNL, "/m20/vlfm/navigate", execute_callback=self.execute,
                                   goal_callback=self.on_goal, cancel_callback=lambda _g: CancelResponse.ACCEPT,
                                   callback_group=self.cb)
        self.get_logger().info(f"VLFM-style search up: /m20/vlfm/navigate, ITM at {self.itm_url}, logs in {self.log_dir}")

    # ---------- inputs ----------
    def on_goal(self, g):
        if not g.text.strip():
            return GoalResponse.REJECT
        with self.goal_lock:
            if self.running:
                self.get_logger().warn("goal rejected: a search is already running (m20 search stop first)")
                return GoalResponse.REJECT
            self.running = True
        return GoalResponse.ACCEPT

    def on_rgb(self, m):
        if m.encoding not in ("rgb8", "bgr8"):
            return
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step // 3, 3)[:, :m.width]
        with self.lock:
            self.rgb, self.frame_t = (a[:, :, ::-1] if m.encoding == "bgr8" else a).copy(), time.monotonic()

    def on_depth(self, m):
        if m.encoding != "32FC1":
            return
        with self.lock:
            self.depth = np.frombuffer(m.data, np.float32).reshape(m.height, m.step // 4)[:, :m.width].copy()

    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        with self.lock:
            self.pose = (p.x, p.y, yaw_of(q))

    def on_true(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        with self.lock:
            self.true_pose = (round(p.x, 3), round(p.y, 3), round(yaw_of(q), 4))

    def on_mode(self, m):
        self.mode_ok = m.armed and m.mode in (RobotMode.MODE_ASSISTED, RobotMode.MODE_AUTONOMOUS)

    def on_det(self, m):
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        with self.lock:
            self.detections = d
            if not d.get("objects"):
                self.det_fresh = True          # the detector's map is empty: anything from now on is this episode's

    # ---------- services ----------
    def costmap(self):
        """-> (cost uint8 HxW, resolution, (ox, oy)) from the global costmap service, or None."""
        if not self.get_costmap.wait_for_service(timeout_sec=0.5):
            return None
        f = self.get_costmap.call_async(GetCostmap.Request())
        t = time.monotonic()
        while not f.done() and time.monotonic() - t < 2.0:
            time.sleep(0.01)
        if not f.done() or f.result() is None:
            return None
        m = f.result().map; md = m.metadata
        cost = np.array(m.data, np.uint8).reshape(md.size_y, md.size_x)
        return cost, float(md.resolution), (md.origin.position.x, md.origin.position.y)

    def score(self, rgb, target):
        body = json.dumps({"frame": jpeg_b64(rgb), "text": PROMPT.format(target=target)}).encode()
        req = urllib.request.Request(self.itm_url + "/score", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=3.0) as r:
                return float(json.loads(r.read())["cosine"])
        except Exception as e:
            self.get_logger().warn(f"ITM request failed: {type(e).__name__}: {e}")
            return None

    @staticmethod
    def _wait(fut, timeout):
        return behaviors.wait(fut, timeout)

    # ---------- driving ----------
    def send_goto(self, label, x, y, yaw):
        """Start a one-item mission; -> goal handle or None."""
        self.cancel_goto()
        if not self.mission.wait_for_server(timeout_sec=3.0):
            return None
        g = RunMission.Goal(); g.items = [goto_item(label, x, y, yaw)]
        send = self.mission.send_goal_async(g)
        if not self._wait(send, 5.0):
            return None
        mgh = send.result()
        if mgh is None or not mgh.accepted:
            return None
        self.active = mgh
        self._result = mgh.get_result_async()
        return mgh

    def goto_state(self):
        """running | reached | failed | none"""
        if self.active is None:
            return "none"
        if not self._result.done():
            return "running"
        r = self._result.result()
        return "reached" if r.result.success else "failed"

    def cancel_goto(self):
        if self.active is not None:
            try:
                self.active.cancel_goal_async()
                self._wait(self._result, 3.0)
            except Exception:
                pass
            self.active = None

    # ---------- the search ----------
    def execute(self, gh):
        req, res = gh.request, NavigateNL.Result()
        # the front door: a move sequence is executed by dead reckoning, an object phrase is searched for, anything
        # else is refused at once (a search for a sentence that names no object wandered 104 m)
        instr = classify(req.text)
        target = instr.phrase if instr.kind == OBJECT else " ".join(req.text.lower().split())
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        out = os.path.join(self.log_dir, stamp)
        log = {"target": target, "kind": instr.kind, "end": None, "t0": time.monotonic(), "steps": [], "frontier_goals": 0,
               "scores": 0, "approach": None, "path_m": 0.0, "last_xy": None, "moves": [m.text() for m in instr.seq]}
        self.get_logger().info(f"{instr.kind}: {target!r} -> {out}")
        try:                                                   # everything that can raise: `running` must clear
            os.makedirs(out, exist_ok=True)
            if instr.kind == MOVES:
                self._run_moves(gh, req, instr.seq, log)
            elif instr.kind == OBJECT:
                self._run(gh, req, target, log, out)
            else:
                log["end"] = "not_understood"
        except Exception as e:
            self.get_logger().error(f"search crashed: {type(e).__name__}: {e}")
            log["end"] = log["end"] or "crash"
        finally:
            self.cancel_spin(); self.cancel_goto(); self.cancel_behavior()
            with self.goal_lock:
                self.running = False
        status = STATUS.get(log["end"], "error")
        with self.lock:
            objs = self.detections.get("objects", [])
        summary = {"target": target, "kind": log["kind"], "moves": log["moves"], "status": status, "end_reason": log["end"],
                   "seconds": round(time.monotonic() - log["t0"], 1),
                   "decisions": len(log["steps"]), "frontier_goals": log["frontier_goals"], "scores": log["scores"],
                   "path_m": round(log["path_m"], 2), "approach": log["approach"], "objects_seen": len(objs)}
        try:                                                   # a logging failure must not leave the goal non-terminal
            with open(os.path.join(out, "steps.jsonl"), "w") as f:
                for s in log["steps"]:
                    f.write(json.dumps(s) + "\n")
            json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
            json.dump(objs, open(os.path.join(out, "objects_seen.json"), "w"), indent=1)
        except Exception as e:
            self.get_logger().error(f"episode log not written: {type(e).__name__}: {e}")
        res.status, res.success = status, status == "done"
        res.interpretation = f"{target}: {log['end']}" + (f" at {log['approach']['object_xy']}" if log["approach"] else "")
        res.tier, res.latency_ms = 3, 0.0                      # 3 = VLFM-style search
        res.detail = json.dumps({**summary, "log": stamp})
        self.get_logger().info(f"search end: {summary}")
        act = terminal_action(status, gh.is_cancel_requested)
        if act == "canceled":
            res.status = "canceled"; gh.canceled()
        elif act == "succeed":
            gh.succeed()
        else:
            gh.abort()
        return res

    def start_fresh(self):
        """No maps given: forget the exploration map (back to all-unknown) and the detector's objects, then let the
        depth camera fill the first cells (about a second) so the first frontiers are real."""
        with self.lock:
            self.det_fresh = False             # ignore the detector until it has published an empty map (a stale
        self.reset_pub.publish(Empty())        # message in flight could resurrect the last episode's target)
        if self.clear_costmap.wait_for_service(timeout_sec=1.0):
            self._wait(self.clear_costmap.call_async(ClearEntireCostmap.Request()), 3.0)
        else:
            self.get_logger().warn("clear_entirely_global_costmap not available: the exploration map keeps its history")
        with self.lock:
            self.detections = {"frame": [], "objects": []}
        time.sleep(1.5)

    # ---------- dead-reckoned moves (the dispatcher's MOVES branch; the same behaviors stage 1 drives) ----------
    def _run_moves(self, gh, req, mvs, log):
        """Execute the moves one after the other with behaviors.execute_move (stage 1's executor). A move the
        behavior aborts (an obstacle) ends the episode "blocked"; the server missing or refusing -> "move_error"."""
        if not req.dry_run and not self.mode_ok:
            log["end"] = "mode"; return

        def should_stop():
            if gh.is_cancel_requested:
                return "canceled"
            if not self.mode_ok:
                return "mode"
            return None

        def on_active(bgh):
            self.active_move = bgh
        for i, mv in enumerate(mvs):
            step = {"t": round(time.monotonic() - log["t0"], 1), "i": i, "move": mv.text()}
            with self.lock:
                step["true_pose"] = self.true_pose
            if req.dry_run:
                step["result"] = "dry_run"; log["steps"].append(step); continue
            ok, detail = behaviors.execute_move(self, self.behaviors, mv, self.move_speed, should_stop, on_active)
            step["result"] = detail
            with self.lock:
                step["true_pose_after"] = self.true_pose
            log["steps"].append(step)
            self.get_logger().info(f"  move {i + 1}/{len(mvs)} {mv.text()}: {detail}")
            if not ok:
                if detail in ("canceled", "mode"):
                    log["end"] = detail
                elif detail.startswith("behavior status"):
                    log["end"] = "blocked"
                else:
                    log["end"] = "move_error"
                return
        log["end"] = "done"

    def cancel_behavior(self):
        """A crash path's cleanup: a dispatched move still in flight is cancelled (fire-and-forget)."""
        if self.active_move is not None:
            try:
                self.active_move.cancel_goal_async()
            except Exception:
                pass
            self.active_move = None

    def start_spin(self, yaw=2 * math.pi, allow_s=25.0):
        """Start a full turn on the spot (Nav2 Spin behavior) and return at once; the loop keeps scoring views."""
        if not self.spin.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("spin behavior unavailable: no look-around"); return
        g = Spin.Goal(); g.target_yaw = float(yaw)
        g.time_allowance.sec = int(allow_s)
        send = self.spin.send_goal_async(g)
        if not self._wait(send, 3.0) or send.result() is None or not send.result().accepted:
            self.get_logger().warn("spin rejected: no look-around"); return
        self.spinning = (send.result(), send.result().get_result_async(), time.monotonic() + allow_s)

    def spin_running(self):
        if self.spinning is None:
            return False
        _h, res, deadline = self.spinning
        if res.done() or time.monotonic() > deadline:
            self.cancel_spin(); return False
        return True

    def cancel_spin(self):
        if self.spinning is not None:
            h, res, _d = self.spinning
            self.spinning = None
            if not res.done():
                try:
                    h.cancel_goal_async(); self._wait(res, 3.0)
                except Exception:
                    pass

    def _run(self, gh, req, target, log, out):
        if not req.dry_run and not self.mode_ok:               # unarmed: nothing moves, not even the look-around
            log["end"] = "mode"; return
        if self.fresh:
            self.start_fresh()
        if self.look_around and not req.dry_run:
            self.start_spin()
            log["look_around"] = True
        cm = self.costmap()
        if cm is None:
            log["end"] = "no_map"; return
        cost, res_m, origin = cm
        vm = ValueMap(cost.shape[1], cost.shape[0], res_m, origin, self.fov, self.max_depth, self.min_depth, self.fusion)
        sel = fr.Selector()
        goal_xy, goal_frontier_value = None, None
        next_score, next_plan, next_png, n_png = 0.0, 0.0, 0.0, 0
        empty = 0                                              # ticks in a row without a frontier
        trail, stuck_n, goal_t = [], 0, 0.0                    # (time, xy) over the last stuck_s; goals abandoned; last send
        itm_failures = 0                                       # scores in a row the matcher did not answer
        q = String(); q.data = target
        self.query_pub.publish(q)
        while True:
            now = time.monotonic()
            if gh.is_cancel_requested:
                log["end"] = "canceled"; return
            if not req.dry_run and not self.mode_ok:
                log["end"] = "mode"; return
            if now - log["t0"] > self.max_seconds:
                log["end"] = "timeout"; return
            with self.lock:
                rgb, depth, pose, tp = self.rgb, self.depth, self.pose, self.true_pose
                dets = self.detections if self.det_fresh else {"objects": []}
            if pose is not None:                               # path length from the believed pose
                if log["last_xy"] is not None:
                    log["path_m"] += math.hypot(pose[0] - log["last_xy"][0], pose[1] - log["last_xy"][1])
                log["last_xy"] = pose[:2]
            # 1. score the view and paint the value map
            if now >= next_score and rgb is not None and depth is not None and pose is not None:
                next_score = now + self.score_dt
                s = self.score(rgb, target)
                if s is not None:
                    vm.update(s, pose[0], pose[1], pose[2], depth_row_from_image(depth, self.min_depth, self.max_depth))
                    log["scores"] += 1
                    log["last_score"] = s
                    itm_failures = 0
                else:                                          # a dead matcher means a blind search: stop instead
                    itm_failures += 1
                    if itm_failures >= ITM_MAX_FAILURES:
                        log["end"] = "itm_error"; return
            # 2. plan: detected -> approach; else best frontier
            if now < next_plan or pose is None:
                time.sleep(0.05); continue
            next_plan = now + self.plan_dt
            step = {"t": round(now - log["t0"], 1), "pose": [round(v, 3) for v in pose], "true_pose": tp,
                    "score": round(log.get("last_score", -1.0), 4)}
            found = [o for o in dets.get("objects", []) if same_class(str(o.get("cls", "")), target)
                     and o.get("n", 0) >= self.target_min_n and o.get("score", 0) >= self.target_min_score]
            if found:
                o = max(found, key=lambda o: o.get("score", 0))
                step["detected"] = {"xy": o["xyz"][:2], "n": o.get("n"), "score": o.get("score"),
                                    "itm": round(log.get("last_score", -1.0), 4)}      # the view's cosine, for calibration
                if req.dry_run:
                    log["steps"].append(step); log["end"] = "done"; return
                self.cancel_spin(); self.cancel_goto()
                # Let the cancel land before the approach goal: a NavigateToPose sent while the previous one is still
                # being aborted trips Nav2's BT action client ("unknown goal response, ignoring" -> "invalid status
                # value" -> the new goal fails at once; the approach ended "not_found" 0.6 s after the detection on
                # Gazebo, 2026-10-26).
                t_cancel = time.monotonic()
                while self.goto_state() == "running" and time.monotonic() - t_cancel < 3.0:
                    time.sleep(0.1)
                cm = self.costmap()
                grid = grid_from_costmap(cm[0].shape[1], cm[0].shape[0], cm[1], cm[2], cm[0].flatten().tolist()) if cm else None
                size = o.get("size") or [0.8, 0.8]
                radius = max(0.2, max(size[:2]) / 2)
                ap, note = approach_pose((o["xyz"][0], o["xyz"][1]), float(radius), grid, from_xy=pose[:2])
                if ap is None:
                    step["approach"] = f"no approach pose ({note})"; log["steps"].append(step)
                    log["end"] = "not_found"; return
                log["approach"] = {"object_xy": [round(o["xyz"][0], 2), round(o["xyz"][1], 2)], "pose": [round(v, 2) for v in ap], "note": note}
                step["approach"] = log["approach"]; log["steps"].append(step)
                self.get_logger().info(f"  {target} detected at ({o['xyz'][0]:.2f}, {o['xyz'][1]:.2f}): approaching")
                mgh = self.send_goto(f"{target} approach", *ap)
                if mgh is None:
                    log["end"] = "crash"; return
                trail = []                                     # a new goal: its own progress window
                while self.goto_state() == "running":
                    if gh.is_cancel_requested or not self.mode_ok:
                        log["end"] = "canceled" if gh.is_cancel_requested else "mode"; return
                    t_now = time.monotonic()
                    if t_now - log["t0"] > self.max_seconds:
                        log["end"] = "timeout"; return
                    with self.lock:
                        p_now = self.pose
                    if p_now is not None:                      # a boxed-in approach is given up like a frontier is
                        trail, stuck = self._tick_trail(trail, t_now, p_now[:2])
                        if stuck:
                            self.get_logger().warn(f"  no progress in {self.stuck_s:.0f} s on the approach: giving up")
                            self.cancel_goto(); log["approach"]["stuck"] = True
                            log["end"] = "stuck"; return
                    time.sleep(0.1)
                log["end"] = "done" if self.goto_state() == "reached" else "not_found"
                self._png(vm, cost, origin, res_m, [], None, pose, os.path.join(out, f"map_{n_png:03d}.png"))
                return
            if self.spin_running():                            # the look-around: paint first, choose after
                step["phase"] = "look_around"; log["steps"].append(step); continue
            # frontiers from the current exploration map
            cm = self.costmap()
            if cm is not None and cm[0].shape == cost.shape:
                cost, res_m, origin = cm
            elif cm is not None:
                self.get_logger().warn(f"costmap shape changed {cost.shape} -> {cm[0].shape}: keeping the old grid")
            fs = fr.extract(cost, res_m, origin)
            for f in fs:
                f.value = vm.read(f.xy[0], f.xy[1])
            state = self.goto_state()
            # progress watchdog (believed pose; drift is slow next to the window): stuck = the robot has stayed
            # within stuck_m of every point of its last stuck_s, WHATEVER goals were sent meanwhile (a boxed-in
            # robot keeps "running" through recoveries while better-valued frontiers are accepted every commit_s:
            # a window that restarted on every send would never fill). Only a REACHED goal restarts it: short goals
            # reached one after the other inside stuck_m are progress, not a box.
            if state == "reached":
                trail = []
            trail, stuck = self._tick_trail(trail, now, pose[:2])
            if goal_xy is not None and stuck:
                stuck_n += 1
                step["stuck"] = [round(goal_xy[0], 2), round(goal_xy[1], 2), stuck_n]
                self.get_logger().warn(f"  no progress in {self.stuck_s:.0f} s (goal {goal_xy[0]:.2f}, {goal_xy[1]:.2f}): "
                                       f"giving it up ({stuck_n}/{MAX_STUCK})")
                self.cancel_goto(); sel.fail(goal_xy)
                state, goal_xy, trail = "failed", None, []
                if stuck_n >= MAX_STUCK:
                    log["steps"].append(step); log["end"] = "stuck"; return
            step["frontiers"] = len(fs)
            step["top"] = [[round(f.xy[0], 2), round(f.xy[1], 2), round(f.value, 3)] for f in sorted(fs, key=lambda f: -f.value)[:3]]
            if not fs:
                # one empty read can be a costmap mid-update; the site counts as explored only after empty_ticks in a row
                empty += 1
                log["steps"].append(step)
                if empty >= self.empty_ticks:
                    log["end"] = "not_found"; return
                continue
            empty = 0
            if state == "running" and goal_xy is not None:
                # The goal frontier's waypoint drifts as cells fill in (segments merge and split), so first find it
                # again (nearest within TRACK_RADIUS_M) and follow it; a cancel + re-send only when the waypoint
                # itself moved more than STICK_RADIUS_M. VLFM's hysteresis then decides whether to keep it: the
                # frontier's value must not have dropped by more than 0.01 since the last tick. (Without the
                # tracking, every small shift read as "a different frontier" and the search thrashed: 41 switches
                # in 270 s on values that differ by 0.002.)
                same = min((f for f in fs if math.dist(f.xy, goal_xy) <= fr.TRACK_RADIUS_M),
                           key=lambda f: math.dist(f.xy, goal_xy), default=None)
                if same is not None:
                    sel.track(same.xy)
                    i = sel.choose(fs, pose[:2], commit=False)         # a peek: nothing counted until it is sent
                    pick = fs[i]
                    if pick is not same and math.dist(pick.xy, same.xy) > fr.STICK_RADIUS_M:
                        if progress.switch_allowed(now, goal_t, self.commit_s):   # the current goal had its chance
                            step["switch"] = [round(pick.xy[0], 2), round(pick.xy[1], 2), round(pick.value, 3)]
                            sent = self._go(pick, pose, log)
                            if sent:
                                sel.commit(fs, i, pose[:2]); goal_xy, goal_t = sent, now
                        # declined: the selector still holds `same` at the last tick's value; judged again next tick
                    else:
                        # kept (or retargeted): commit so last_value becomes THIS tick's value, as VLFM's hysteresis
                        # compares tick to tick; it counts no try, the pick being within STICK_RADIUS_M of last_xy
                        sel.commit(fs, i, pose[:2])
                        if math.dist(same.xy, goal_xy) > fr.STICK_RADIUS_M:      # a retarget is never gated
                            step["retarget"] = [round(same.xy[0], 2), round(same.xy[1], 2)]
                            goal_xy = self._go(same, pose, log) or goal_xy
                else:                                          # the goal frontier is gone (consumed or merged away)
                    i = sel.choose(fs, pose[:2])
                    pick = fs[i]
                    step["switch"] = [round(pick.xy[0], 2), round(pick.xy[1], 2), round(pick.value, 3)]
                    sent = self._go(pick, pose, log)
                    goal_xy, goal_t = (sent, now) if sent else (goal_xy, goal_t)
                step["goal"] = [round(goal_xy[0], 2), round(goal_xy[1], 2)]
            else:
                if goal_xy is not None and state == "failed":
                    # Nav2 gave up on it (behind glass, an unreachable gap): never send that frontier again
                    step["goal_failed"] = [round(goal_xy[0], 2), round(goal_xy[1], 2)]
                    sel.fail(goal_xy)
                elif goal_xy is not None and state == "reached" and \
                        any(math.dist(f.xy, goal_xy) <= fr.STICK_RADIUS_M for f in fs):
                    # arrived and the frontier is still there (unknown space the sensors can't resolve): same
                    step["goal_persists"] = [round(goal_xy[0], 2), round(goal_xy[1], 2)]
                    sel.fail(goal_xy)
                i = sel.choose(fs, pose[:2])
                pick = fs[i]
                goal_xy, goal_t = self._go(pick, pose, log), now
                if goal_xy is None:                            # the mission server did not take it: try again next tick
                    step["goal_rejected"] = [round(pick.xy[0], 2), round(pick.xy[1], 2)]
                else:
                    step["goal"] = [round(goal_xy[0], 2), round(goal_xy[1], 2), round(pick.value, 3)]
            log["steps"].append(step)
            if now >= next_png:
                next_png = now + 10.0
                self._png(vm, cost, origin, res_m, fs, goal_xy, pose, os.path.join(out, f"map_{n_png:03d}.png")); n_png += 1

    def _tick_trail(self, trail, now, xy):
        """The progress watchdog's one step: add the believed xy, drop what is older than stuck_s, judge.
        -> (trail, stuck). Stuck = the trail spans the window and never left stuck_m of where the robot is now."""
        trail.append((now, xy))
        trail = progress.prune(trail, now, self.stuck_s)
        return trail, progress.is_stuck(trail, now, xy, self.stuck_s, self.stuck_m)

    def _go(self, f, pose, log):
        """Send a frontier as the mission goal. -> its xy, or None if the mission server did not accept it."""
        yaw = math.atan2(f.xy[1] - pose[1], f.xy[0] - pose[0])
        if self.send_goto(f"frontier {f.xy[0]:.1f},{f.xy[1]:.1f}", f.xy[0], f.xy[1], yaw) is None:
            self.get_logger().warn(f"  frontier goal ({f.xy[0]:.2f}, {f.xy[1]:.2f}) not accepted by the mission server")
            return None
        log["frontier_goals"] += 1
        self.get_logger().info(f"  frontier goal ({f.xy[0]:.2f}, {f.xy[1]:.2f}) value {f.value:.3f}, {f.length_m} m long")
        return f.xy

    @staticmethod
    def _png(vm, cost, origin, res_m, fs, goal_xy, pose, path):
        """Value map (dark = unobserved, warm = high value) with the exploration map's walls, frontiers and the goal.
        Cosines sit in a narrow band (~0.12-0.40), so the colour is stretched over the observed min..max."""
        try:
            h, w = cost.shape
            seen = vm.conf > 0
            lo, hi = (vm.value[seen].min(), vm.value[seen].max()) if seen.any() else (0.0, 1.0)
            v = np.clip((vm.value - lo) / max(hi - lo, 1e-3), 0.0, 1.0)
            img = np.zeros((h, w, 3), np.uint8)
            img[..., 0] = np.where(seen, 40 + 215 * v, 20); img[..., 1] = np.where(seen, 40 + 120 * v, 20); img[..., 2] = np.where(seen, 90 - 60 * v, 30)
            img[(cost >= 253) & (cost != 255)] = (0, 0, 0)
            for f in fs:
                img[f.cells[:, 0], f.cells[:, 1]] = (255, 255, 255)
                r, c = vm.cell(f.xy[0], f.xy[1])
                img[max(r - 2, 0):r + 3, max(c - 2, 0):c + 3] = (0, 255, 0)
            if goal_xy is not None:
                r, c = vm.cell(goal_xy[0], goal_xy[1])
                img[max(r - 3, 0):r + 4, max(c - 3, 0):c + 4] = (255, 0, 255)
            r, c = vm.cell(pose[0], pose[1])
            img[max(r - 3, 0):r + 4, max(c - 3, 0):c + 4] = (0, 150, 255)
            PILImage.fromarray(img[::-1]).resize((w * 2, h * 2), PILImage.NEAREST).save(path)
        except Exception:
            pass


def main():
    rclpy.init()
    node = SearchNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
