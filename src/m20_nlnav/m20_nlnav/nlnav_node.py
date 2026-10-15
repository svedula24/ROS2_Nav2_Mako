"""m20_nlnav: natural-language navigation node, offline phase (docs/NL_NAVIGATION_DESIGN.md; the
"NL Nav Architecture" plan: offline semantic map for known sites).

Serves m20_msgs/action/NavigateNL on /m20/nlnav/navigate:
  text --LLM (llm.py, structured output only)--> query --semantic.py (plain code, object map)--> goal poses
  --ComputePathToPose reachability--> ONE RunMission on /m20/mission/run (m20_missions stays the sole
  driver, ADR-014; the Commander can still veto, ADR-005). The language layer never commands velocity and
  never picks the goal itself: the LLM only fills the query schema.

Inputs:  the site's object map (param object_map: format v1 JSON, tools/semmap/build_object_map.py;
           re-read when it changes)
         the robot pose ("here"), param pose_source:
           odom_true = nav_msgs/Odometry on param odom_topic (default /odom_true: the sim's ground truth)
           tf        = TF map -> base_link, the pose Nav2 believes (AMCL); use this whenever localization is real
         Nav2's global costmap (get_costmap service per command; the topic as fallback)
         scene yaml `waypoints:` (param scene_yaml) = named places ("start", ...)
Params:  llm (ollama | claude | json; env NLNAV_LLM overrides), text_encoder (owl | label),
         threshold (OWLv2 probability to count as a match), reach_check (bool)
Outputs: /m20/nlnav/markers (candidates, target, goals), /m20/health
"""
import math
import os
import threading
import time

import rclpy
import rclpy.time
import yaml
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy,
                       qos_profile_sensor_data)

from geometry_msgs.msg import Point
from nav2_msgs.action import ComputePathToPose
from nav2_msgs.srv import GetCostmap
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import Marker, MarkerArray
from m20_msgs.action import NavigateNL, RunMission
from m20_msgs.msg import HealthReport, MissionItem, NLCandidate

from . import llm, semantic
from .object_map import ObjectMap
from .world import grid_from_costmap, grid_from_msg, waypoints_from_scene


def yaw_of(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class NLNavNode(Node):
    def __init__(self):
        super().__init__("m20_nlnav")
        for name, default in (("scene_yaml", ""), ("object_map", ""), ("odom_topic", "/odom_true"),
                              ("costmap_topic", "/global_costmap/costmap"),
                              ("costmap_service", "/global_costmap/get_costmap"),
                              ("llm", os.environ.get("NLNAV_LLM", "ollama")), ("text_encoder", "owl"),
                              ("threshold", 0.10), ("reach_check", True),
                              ("min_label_score", 0.4), ("min_object_score", 0.0),
                              ("pose_source", "odom_true"), ("map_frame", "map"), ("base_frame", "base_link")):
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value
        self.scene_yaml, self.map_path = p("scene_yaml"), p("object_map")
        self.threshold, self.reach_check = float(p("threshold")), bool(p("reach_check"))
        # confident labels only: below min_label_score the word becomes "object" (embedding kept); below
        # min_object_score the object is dropped. See ObjectMap.confident.
        self.min_label_score, self.min_object_score = float(p("min_label_score")), float(p("min_object_score"))
        self.backend = llm.make_backend(p("llm"))
        self.encoder_kind, self._encoder, self.enc_lock = p("text_encoder"), None, threading.Lock()

        self.cb = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self.pose = None
        self.grid, self.grid_t = None, 0.0
        self.waypoints, self.scene_mtime = {}, None
        self.omap, self.map_mtime = ObjectMap(), None
        self.active = None                     # (RunMission goal handle) of OUR running mission

        self.pose_source = p("pose_source")
        if self.pose_source == "tf":           # believed pose: same source as Nav2's planner/controller
            import tf2_ros
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            self.frames = (p("map_frame"), p("base_frame"))
            self.create_timer(0.1, self.on_tf_tick, callback_group=self.cb)
        elif self.pose_source == "odom_true":
            self.create_subscription(Odometry, p("odom_topic"), self.on_odom, qos_profile_sensor_data,
                                     callback_group=self.cb)
        else:
            raise ValueError(f"pose_source must be odom_true or tf, got {self.pose_source!r}")
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(OccupancyGrid, p("costmap_topic"), self.on_costmap, latched,
                                 callback_group=self.cb)
        self.get_costmap = self.create_client(GetCostmap, p("costmap_service"), callback_group=self.cb)
        self.planner = ActionClient(self, ComputePathToPose, "/compute_path_to_pose", callback_group=self.cb)
        self.markers = self.create_publisher(MarkerArray, "/m20/nlnav/markers", 10)
        self.mission = ActionClient(self, RunMission, "/m20/mission/run", callback_group=self.cb)
        self.server = ActionServer(self, NavigateNL, "/m20/nlnav/navigate",
                                   execute_callback=self.execute,
                                   goal_callback=lambda g: GoalResponse.ACCEPT if g.text.strip()
                                   else GoalResponse.REJECT,
                                   cancel_callback=lambda gh: CancelResponse.ACCEPT,
                                   callback_group=self.cb)
        self.health_pub = self.create_publisher(HealthReport, "/m20/health", 10)
        self.create_timer(1.0, self.heartbeat, callback_group=self.cb)
        self.load_waypoints()
        self.load_object_map()
        self.get_logger().info(
            f"NL navigation (offline phase) up: llm={self.backend.name}, encoder={self.encoder_kind}, "
            f"{len(self.omap.objects)} objects from {self.map_path or '(no object map)'}, "
            f"{len(self.waypoints)} named places, pose from {self.pose_source}")
        threading.Thread(target=self.warm_up, daemon=True).start()

    def warm_up(self):
        """Load the LLM and the text encoder now, so the first instruction doesn't pay a 15-20 s cold start."""
        t = time.time()
        try:
            self.encoder()("chair")
            if hasattr(self.backend, "warm"):
                self.backend.warm()                      # weights into memory...
                # ...and one real parse: the first generation also pays the system prompt's KV cache
                # and the JSON-schema grammar (~20 s cold, measured), which `warm` alone doesn't touch.
                llm.parse("stop", self.backend, sorted({w.name for w in self.waypoints.values()} | {"start"}),
                          timeout_s=60.0)
            self.get_logger().info(f"warm-up done in {time.time() - t:.1f} s")
        except Exception as e:                 # not fatal: the first query just pays the load instead
            self.get_logger().warn(f"warm-up failed ({type(e).__name__}: {e})")

    # ---------- inputs ----------
    def on_odom(self, m):
        q = m.pose.pose.orientation
        with self.lock:
            self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_of(q))

    def on_tf_tick(self):
        try:                                   # latest available map -> base_link (Time() = newest)
            t = self.tf_buffer.lookup_transform(self.frames[0], self.frames[1], rclpy.time.Time())
        except Exception:                      # not yet localized / TF not up: keep the last pose (or None)
            return
        tr, q = t.transform.translation, t.transform.rotation
        with self.lock:
            self.pose = (tr.x, tr.y, yaw_of(q))

    def on_costmap(self, m):
        g = grid_from_msg(m.info.width, m.info.height, m.info.resolution,
                          (m.info.origin.position.x, m.info.origin.position.y), m.data)
        with self.lock:
            self.grid, self.grid_t = g, time.monotonic()

    def load_waypoints(self):
        if not self.scene_yaml or not os.path.exists(self.scene_yaml):
            return
        mt = os.path.getmtime(self.scene_yaml)
        if mt != self.scene_mtime:
            with open(self.scene_yaml) as f:
                self.waypoints = waypoints_from_scene(yaml.safe_load(f) or {})
            self.scene_mtime = mt

    def load_object_map(self):
        if not self.map_path or not os.path.exists(self.map_path):
            return
        mt = os.path.getmtime(self.map_path)
        if mt != self.map_mtime:
            raw = ObjectMap.load(self.map_path)
            self.omap = raw.confident(self.min_label_score, self.min_object_score)
            self.map_mtime = mt
            labeled = sum(o.label != "object" for o in self.omap.objects)
            self.get_logger().info(f"object map: {len(self.omap.objects)} of {len(raw.objects)} objects kept, "
                                   f"{labeled} with a label >= {self.min_label_score} ({self.map_path})")

    def encoder(self):
        with self.enc_lock:                    # warm-up thread and a first query may race to load it
            if self._encoder is None:
                self._encoder = semantic.OwlTextEncoder() if self.encoder_kind == "owl" else semantic.LabelEncoder()
        return self._encoder

    def fetch_grid(self, timeout_s: float = 2.0):
        """Current global costmap from Nav2's service, or the last topic message, or None."""
        if self.get_costmap.wait_for_service(timeout_sec=0.5):
            fut = self.get_costmap.call_async(GetCostmap.Request())
            t = time.time()
            while not fut.done() and time.time() - t < timeout_s:
                time.sleep(0.01)
            if fut.done() and fut.result() is not None:
                m = fut.result().map
                md = m.metadata
                return grid_from_costmap(md.size_x, md.size_y, md.resolution,
                                         (md.origin.position.x, md.origin.position.y), m.data)
        with self.lock:
            return self.grid

    def reachable(self, goal, timeout_s: float = 3.0) -> bool:
        """Nav2's planner finds a path to (x, y, yaw)? Unknown (no planner) counts as yes."""
        if not self.planner.wait_for_server(timeout_sec=0.5):
            return True
        g = ComputePathToPose.Goal()
        g.goal.header.frame_id = "map"
        g.goal.pose.position.x, g.goal.pose.position.y = float(goal[0]), float(goal[1])
        g.goal.pose.orientation.z, g.goal.pose.orientation.w = math.sin(goal[2] / 2), math.cos(goal[2] / 2)
        g.use_start = False
        send = self.planner.send_goal_async(g)
        t = time.time()
        while not send.done() and time.time() - t < timeout_s:
            time.sleep(0.01)
        if not send.done() or not send.result().accepted:
            return True
        res = send.result().get_result_async()
        while not res.done() and time.time() - t < timeout_s:
            time.sleep(0.01)
        return bool(res.done() and len(res.result().result.path.poses) > 0)

    # ---------- action ----------
    def execute(self, gh):
        req = gh.request
        t0 = time.monotonic()                            # latency: monotonic (WSL steps its wall clock)
        res = NavigateNL.Result()
        res.tier = 1                                           # 1 = LLM-parsed (tier 0 retired)
        self._fb(gh, "resolving")
        try:
            self.load_waypoints()
            self.load_object_map()
        except Exception as e:                 # half-written / wrong-version file: answer, don't kill the goal
            return self._finish(gh, res, t0, "error", detail=f"can't load the site files ({type(e).__name__}: {e})")
        places = sorted({w.name for w in self.waypoints.values()} | {"start"})
        try:
            q = llm.Query([], True) if llm.is_stop(req.text) else llm.parse(req.text, self.backend, places)
        except llm.ParserUnavailable as e:
            return self._finish(gh, res, t0, "error", detail=str(e))
        except llm.NotUnderstood as e:
            return self._finish(gh, res, t0, "not_understood", detail=f"{e}. Please rephrase.")
        if q.stop:
            stopped = self.cancel_active()
            return self._finish(gh, res, t0, "stopped", success=True,
                                detail="stopped" if stopped else "nothing of mine was running")
        with self.lock:
            pose = self.pose
        t1 = time.monotonic()
        grid = self.fetch_grid()
        t2 = time.monotonic()
        try:                                   # e.g. text_encoder:=owl in an image without torch
            r = semantic.resolve(q, self.omap, self.encoder(), self.waypoints, pose=pose, grid=grid,
                                 threshold=self.threshold, pick_id=req.pick_id,
                                 reach_fn=self.reachable if self.reach_check else None)
        except Exception as e:
            self.get_logger().error(f"resolve failed: {type(e).__name__}: {e}")
            return self._finish(gh, res, t0, "error", detail=f"resolver failed ({type(e).__name__}: {e})")
        self.get_logger().info(f"timing: llm {1e3 * (t1 - t0):.0f} ms, costmap {1e3 * (t2 - t1):.0f} ms, "
                               f"resolve+reach {1e3 * (time.monotonic() - t2):.0f} ms")
        res.interpretation = r.interpretation
        res.candidates = [self._cand(c) for c in r.candidates]
        res.items = [self._item(s) for s in r.steps]
        self.publish_markers(r)
        if r.status != "plan":
            return self._finish(gh, res, t0, r.status, detail=r.detail)
        if req.dry_run:
            return self._finish(gh, res, t0, "plan", success=True,
                                detail=("dry run: not driving" + (f"; {r.detail}" if r.detail else "")))
        res.latency_ms = 1e3 * (time.monotonic() - t0)
        res.detail = r.detail
        return self._drive(gh, res)

    def _drive(self, gh, res):
        self.cancel_active()                       # a new command replaces our previous one
        if not self.mission.wait_for_server(timeout_sec=3.0):
            return self._finish(gh, res, None, "error", detail="mission server not running")
        goal = RunMission.Goal()
        goal.items = list(res.items)
        latest = {}
        send = self.mission.send_goal_async(goal, feedback_callback=lambda m: latest.update(fb=m.feedback))
        while not send.done():
            time.sleep(0.02)
        mgh = send.result()
        if mgh is None or not mgh.accepted:
            return self._finish(gh, res, None, "error",
                                detail="mission rejected: arm the robot (ASSISTED/AUTONOMOUS) first")
        self.active = mgh
        result_f = mgh.get_result_async()
        while not result_f.done():
            if gh.is_cancel_requested:
                mgh.cancel_goal_async()
                t = time.time()
                while not result_f.done() and time.time() - t < 5.0:
                    time.sleep(0.05)
                self.active = None
                gh.canceled()
                res.status, res.detail = "canceled", "canceled by the caller"
                return res
            f = latest.get("fb")
            if f is not None:
                self._fb(gh, "navigating", f.current_index, f.current_label, f.status)
            time.sleep(0.2)
        if self.active is mgh:
            self.active = None
        wrapped = result_f.result()
        mres = wrapped.result
        ok = bool(mres.success)
        status = "done" if ok else ("canceled" if wrapped.status == 5 else "error")   # 5 = CANCELED ('stop' / newer)
        detail = mres.detail + (f"; {res.detail}" if res.detail and ok else "")
        return self._finish(gh, res, None, status, success=ok, detail=detail)

    def cancel_active(self) -> bool:
        mgh, self.active = self.active, None
        if mgh is None:
            return False
        mgh.cancel_goal_async()
        return True

    def _finish(self, gh, res, t0, status, success=False, detail=""):
        res.status, res.success, res.detail = status, success, detail
        if t0 is not None:
            res.latency_ms = 1e3 * (time.monotonic() - t0)
        self.get_logger().info(f"'{gh.request.text}' -> {status}: {res.interpretation or ''} {detail}".strip())
        gh.succeed()
        return res

    def _fb(self, gh, stage, idx=0, label="", status=""):
        f = NavigateNL.Feedback()
        f.stage, f.current_index, f.current_label, f.status = stage, int(idx), label, status
        gh.publish_feedback(f)

    @staticmethod
    def _item(step) -> MissionItem:
        it = MissionItem()
        it.type = MissionItem.TYPE_GOTO
        it.label = step.label
        it.goal.header.frame_id = "map"
        x, y, yaw = step.goal
        it.goal.pose.position.x, it.goal.pose.position.y = float(x), float(y)
        it.goal.pose.orientation.z, it.goal.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return it

    @staticmethod
    def _cand(c) -> NLCandidate:
        m = NLCandidate()
        m.object_id, m.cls, m.score, m.why = int(c.obj.id), c.obj.label, float(c.score), c.why
        m.position = Point(x=float(c.obj.xyz[0]), y=float(c.obj.xyz[1]), z=float(c.obj.xyz[2]))
        return m

    def heartbeat(self):
        """Liveness for docker/healthcheck.py (HEALTH_NODE=m20_nlnav); WARN while blind."""
        h = HealthReport()
        h.header.stamp = self.get_clock().now().to_msg()
        h.node_name = self.get_name()
        with self.lock:
            blind = [n for n, v in (("object map", self.omap.objects or None), ("pose", self.pose)) if v is None]
        h.status = HealthReport.STATUS_WARN if blind else HealthReport.STATUS_OK
        h.message = ("no " + ", no ".join(blind)) if blind else "serving /m20/nlnav/navigate"
        self.health_pub.publish(h)

    # ---------- markers (Foxglove / RViz) ----------
    def publish_markers(self, r):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        stamp = self.get_clock().now().to_msg()

        def mk(ns, i, typ, rgba):
            m = Marker()
            m.header.frame_id, m.header.stamp = "map", stamp
            m.ns, m.id, m.type, m.action = ns, i, typ, Marker.ADD
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            m.pose.orientation.w = 1.0
            return m

        for i, c in enumerate(r.candidates):
            m = mk("candidates", i, Marker.SPHERE, (1.0, 0.85, 0.1, 0.8))
            m.pose.position = Point(x=float(c.obj.xyz[0]), y=float(c.obj.xyz[1]), z=float(c.obj.xyz[2]))
            m.scale.x = m.scale.y = m.scale.z = 0.3
            arr.markers.append(m)
        for i, s in enumerate(r.steps):
            if s.obj is not None:
                m = mk("targets", i, Marker.CUBE, (1.0, 0.0, 1.0, 0.5))
                m.pose.position = Point(x=float(s.obj.xyz[0]), y=float(s.obj.xyz[1]), z=float(s.obj.xyz[2]))
                sz = s.obj.size or (0.4, 0.4, 0.4)
                m.scale.x, m.scale.y, m.scale.z = (float(v) for v in sz)
                arr.markers.append(m)
            m = mk("goals", i, Marker.ARROW, (0.1, 0.9, 0.3, 0.9))
            x, y, yaw = s.goal
            m.pose.position = Point(x=float(x), y=float(y), z=0.1)
            m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            m.scale.x, m.scale.y, m.scale.z = 0.6, 0.12, 0.12
            arr.markers.append(m)
        self.markers.publish(arr)


def main():
    rclpy.init()
    node = NLNavNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    ex.spin()


if __name__ == "__main__":
    main()
