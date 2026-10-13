"""m20_nlnav: natural-language navigation node (docs/NL_NAVIGATION_DESIGN.md).

Serves m20_msgs/action/NavigateNL on /m20/nlnav/navigate: text -> parse -> resolve against what
the robot has seen -> goal poses -> ONE RunMission on /m20/mission/run (m20_missions stays the
sole driver, ADR-014; the Commander can still veto, ADR-005). The language layer only chooses
goal poses: it never commands velocity.

Phase 1 = tier 0 only (exact names + synonyms, tier0.py). Text it can't fully parse is answered
not_found, never guessed; the LLM/VLM tiers plug in where tier0.parse() returns None.

Inputs:  /detections/objects (std_msgs/String JSON, confirmed object map; detector_node.py)
         odom (nav_msgs/Odometry, param odom_topic, default /odom_true in the sim)
         Nav2's global costmap for approach-pose checks: fetched per command from the
           /global_costmap/get_costmap service (the topic only publishes on change, and its
           latched copy doesn't reliably reach late subscribers across containers here);
           /global_costmap/costmap (OccupancyGrid) is the fallback
         scene yaml `waypoints:` (param scene_yaml; re-read when it changes)
Outputs: /m20/nlnav/markers (visualization_msgs/MarkerArray: candidates, target, goals)
"""
import math
import os
import threading
import time

import rclpy
import yaml
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy,
                       qos_profile_sensor_data)

from geometry_msgs.msg import Point
from nav2_msgs.srv import GetCostmap
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from m20_msgs.action import NavigateNL, RunMission
from m20_msgs.msg import HealthReport, MissionItem, NLCandidate

from . import tier0
from .resolve import resolve
from .world import (DETECTOR_CLASSES, World, grid_from_costmap, grid_from_msg, objects_from_json,
                    waypoints_from_scene)

HELP = ("I can follow simple commands so far: 'go to the printer', 'go to the nearest chair', "
        "'go to stop 3', 'go back to the start', 'stop', and chains with 'then'")


def yaw_of(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class NLNavNode(Node):
    def __init__(self):
        super().__init__("m20_nlnav")
        self.declare_parameter("scene_yaml", "")
        self.declare_parameter("min_score", 0.5)               # = Viser's default Min confidence
        self.declare_parameter("odom_topic", "/odom_true")
        self.declare_parameter("objects_topic", "/detections/objects")
        self.declare_parameter("costmap_topic", "/global_costmap/costmap")
        self.declare_parameter("costmap_service", "/global_costmap/get_costmap")
        p = lambda n: self.get_parameter(n).value
        self.min_score = float(p("min_score"))
        self.scene_yaml = p("scene_yaml")

        self.cb = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self.objects_json = None
        self.pose = None
        self.grid, self.grid_t = None, 0.0
        self.waypoints, self.scene_mtime = {}, None
        self.active = None                     # (RunMission goal handle) of OUR running mission

        self.create_subscription(String, p("objects_topic"), self.on_objects, 10, callback_group=self.cb)
        self.create_subscription(Odometry, p("odom_topic"), self.on_odom, qos_profile_sensor_data,
                                 callback_group=self.cb)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(OccupancyGrid, p("costmap_topic"), self.on_costmap, latched,
                                 callback_group=self.cb)
        self.get_costmap = self.create_client(GetCostmap, p("costmap_service"), callback_group=self.cb)
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
        self.get_logger().info(f"NL navigation up: /m20/nlnav/navigate (NavigateNL); "
                               f"{len(self.waypoints)} waypoints from {self.scene_yaml or '(none)'}")

    # ---------- inputs ----------
    def on_objects(self, m):
        with self.lock:
            self.objects_json = m.data

    def on_odom(self, m):
        q = m.pose.pose.orientation
        with self.lock:
            self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_of(q))

    def on_costmap(self, m):
        g = grid_from_msg(m.info.width, m.info.height, m.info.resolution,
                          (m.info.origin.position.x, m.info.origin.position.y), m.data)
        with self.lock:
            self.grid, self.grid_t = g, time.monotonic()

    def load_waypoints(self):
        if not self.scene_yaml or not os.path.exists(self.scene_yaml):
            return
        mt = os.path.getmtime(self.scene_yaml)
        if mt == self.scene_mtime:
            return
        with open(self.scene_yaml) as f:
            self.waypoints = waypoints_from_scene(yaml.safe_load(f) or {})
        self.scene_mtime = mt

    def fetch_grid(self, timeout_s: float = 2.0):
        """Current global costmap from Nav2's service, or None."""
        if not self.get_costmap.wait_for_service(timeout_sec=0.5):
            return None
        fut = self.get_costmap.call_async(GetCostmap.Request())
        t = time.time()
        while not fut.done() and time.time() - t < timeout_s:
            time.sleep(0.01)
        if not fut.done() or fut.result() is None:
            return None
        m = fut.result().map
        md = m.metadata
        return grid_from_costmap(md.size_x, md.size_y, md.resolution,
                                 (md.origin.position.x, md.origin.position.y), m.data)

    def snapshot(self) -> World:
        self.load_waypoints()
        grid = self.fetch_grid()
        with self.lock:
            every = objects_from_json(self.objects_json) if self.objects_json else []
            if grid is None and self.grid is not None:
                self.get_logger().warn("get_costmap unavailable: using the last costmap topic message")
                grid = self.grid
            return World(objects=[o for o in every if o.score >= self.min_score],
                         low_conf=[o for o in every if o.score < self.min_score],
                         min_score=self.min_score, waypoints=dict(self.waypoints), pose=self.pose,
                         grid=grid)

    # ---------- action ----------
    def execute(self, gh):
        req = gh.request
        t0 = time.time()
        res = NavigateNL.Result()
        res.tier = 0
        self._fb(gh, "resolving")
        world = self.snapshot()
        known = sorted(set(DETECTOR_CLASSES) | set(world.classes()))
        cmd = tier0.parse(req.text, known, world.waypoints)
        if cmd is None:
            return self._finish(gh, res, t0, "not_found", detail=f"didn't understand '{req.text}'. {HELP}")
        r = resolve(cmd, world, pick_id=req.pick_id)        # single mode: req.mode is ignored
        res.interpretation = r.interpretation
        res.candidates = [self._cand(c) for c in r.candidates]
        res.items = [self._item(s) for s in r.steps]
        self.publish_markers(r)
        if r.status == "stop":
            stopped = self.cancel_active()
            return self._finish(gh, res, t0, "stopped", success=True,
                                detail="stopped" if stopped else "nothing of mine was running")
        if r.status != "plan":
            return self._finish(gh, res, t0, r.status, detail=r.detail)
        if req.dry_run:
            return self._finish(gh, res, t0, "plan", success=True, detail="dry run: not driving")
        res.latency_ms = 1e3 * (time.time() - t0)
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
        status = "done" if ok else ("canceled" if wrapped.status == 5 else "error")   # 5 = CANCELED
        return self._finish(gh, res, None, status, success=ok, detail=mres.detail)    # ('stop' / a newer command)

    def cancel_active(self) -> bool:
        mgh, self.active = self.active, None
        if mgh is None:
            return False
        mgh.cancel_goal_async()
        return True

    def _finish(self, gh, res, t0, status, success=False, detail=""):
        res.status, res.success, res.detail = status, success, detail
        if t0 is not None:
            res.latency_ms = 1e3 * (time.time() - t0)
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
        m.object_id, m.cls, m.score, m.why = int(c.obj.id), c.obj.cls, float(c.obj.score), c.why
        m.position = Point(x=float(c.obj.xyz[0]), y=float(c.obj.xyz[1]), z=float(c.obj.xyz[2]))
        return m

    def heartbeat(self):
        """Liveness for docker/healthcheck.py (HEALTH_NODE=m20_nlnav); WARN while blind."""
        h = HealthReport()
        h.header.stamp = self.get_clock().now().to_msg()
        h.node_name = self.get_name()
        with self.lock:
            blind = [n for n, v in (("object map", self.objects_json), ("pose", self.pose)) if v is None]
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
