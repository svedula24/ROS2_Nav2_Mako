"""ROS-level test of nlnav_node, offline phase (real rclpy; run by `colcon test` in the CI image).

Fixture world: an object map FILE (format v1, label embeddings), a scene yaml with named places, odometry
and an all-free costmap on real topics, and a FAKE RunMission server standing in for m20_missions. The LLM
is the JSON passthrough backend (the instruction IS the structured query) and the text encoder is the
torch-free label encoder, so no model or network is needed. Checks the NavigateNL contract end to end:
plan / not_found / not_understood / dry run, driving through RunMission, stop, and cancel forwarding.
"""
import json
import os
import tempfile
import threading
import time

import pytest
import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import OccupancyGrid, Odometry
from m20_msgs.action import NavigateNL, RunMission

from m20_nlnav.nlnav_node import NLNavNode
from m20_nlnav.object_map import MapObject, ObjectMap
from m20_nlnav.semantic import label_embedding


def mo(i, label, x, y):
    return MapObject(i, label, (x, y, 0.5), (0.6, 0.5, 1.0), label_embedding(label), -0.5, 10.0, 0.8, 5, 0.0, 0.8)


OBJECTS = [mo(1, "printer", 4.0, 0.0), mo(2, "office chair", 2.0, 3.0), mo(3, "office chair", -3.0, 1.0),
           mo(4, "whiteboard", 2.0, 5.0)]


def Q(*targets, stop=False, **kw):
    """A structured query as the JSON backend expects it."""
    steps = [dict({"target": t, "pick": "best"}, **kw) for t in targets]
    return json.dumps({"steps": steps, "stop": stop})


class FakeMission(Node):
    """RunMission server: 'drives' each item for `secs` seconds, honours cancel."""
    def __init__(self, secs=0.6):
        super().__init__("fake_mission")
        self.secs, self.received, self.canceled = secs, [], 0
        ActionServer(self, RunMission, "/m20/mission/run", execute_callback=self.execute,
                     cancel_callback=lambda gh: CancelResponse.ACCEPT,
                     callback_group=ReentrantCallbackGroup())

    def execute(self, gh):
        self.received.append([(it.label, it.goal.pose.position.x, it.goal.pose.position.y)
                              for it in gh.request.items])
        res = RunMission.Result()
        for i, it in enumerate(gh.request.items):
            t = time.time()
            while time.time() - t < self.secs:
                if gh.is_cancel_requested:
                    self.canceled += 1
                    gh.canceled()
                    res.detail = "canceled"
                    return res
                fb = RunMission.Feedback()
                fb.current_index, fb.current_label, fb.status = i, it.label, "navigating"
                gh.publish_feedback(fb)
                time.sleep(0.05)
            res.completed = i + 1
        res.success, res.detail = True, "ok"
        gh.succeed()
        return res


@pytest.fixture(scope="module")
def stack():
    tmp = tempfile.mkdtemp()
    scene = os.path.join(tmp, "scene.yaml")
    open(scene, "w").write("waypoints:\n  start: [0.0, 0.0, 0.0]\n  dock: [0.8, -8.1]\n")
    omap = os.path.join(tmp, "object_map.json")
    ObjectMap(OBJECTS, site="test").save(omap)
    rclpy.init(args=["--ros-args", "-p", f"scene_yaml:={scene}", "-p", f"object_map:={omap}",
                     "-p", "llm:=json", "-p", "text_encoder:=label", "-p", "reach_check:=false"])
    nl = NLNavNode()
    fake = FakeMission()
    io = Node("fixture_io")
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
    pub_odom = io.create_publisher(Odometry, "/odom_true", 10)
    pub_map = io.create_publisher(OccupancyGrid, "/global_costmap/costmap", latched)
    grid = OccupancyGrid()
    grid.header.frame_id = "map"
    grid.info.resolution, grid.info.width, grid.info.height = 0.1, 200, 200
    grid.info.origin.position.x = grid.info.origin.position.y = -10.0
    grid.data = [0] * (200 * 200)
    ex = MultiThreadedExecutor(num_threads=6)
    for n in (nl, fake, io):
        ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    pub_map.publish(grid)

    def feed():
        while rclpy.ok():
            o = Odometry()
            o.header.frame_id = "map"
            o.pose.pose.orientation.w = 1.0
            pub_odom.publish(o)
            time.sleep(0.1)
    threading.Thread(target=feed, daemon=True).start()
    client = ActionClient(io, NavigateNL, "/m20/nlnav/navigate")
    assert client.wait_for_server(timeout_sec=10.0)
    deadline = time.time() + 10.0                    # wait until the node has seen the world
    while time.time() < deadline and (nl.pose is None or nl.grid is None):
        time.sleep(0.1)
    yield {"client": client, "fake": fake, "nl": nl}
    rclpy.shutdown()


def call(client, text, dry_run=False, pick=-1, cancel_after=None):
    g = NavigateNL.Goal()
    g.text, g.dry_run, g.pick_id = text, dry_run, pick
    f = client.send_goal_async(g)
    while not f.done():
        time.sleep(0.02)
    gh = f.result()
    assert gh.accepted
    if cancel_after is not None:
        time.sleep(cancel_after)
        gh.cancel_goal_async()
    r = gh.get_result_async()
    t = time.time()
    while not r.done() and time.time() - t < 15.0:
        time.sleep(0.02)
    assert r.done()
    return r.result().result


def test_object_map_loaded_from_file(stack):
    assert len(stack["nl"].omap.objects) == 4


def test_dry_run_plan_does_not_drive(stack):
    n0 = len(stack["fake"].received)
    r = call(stack["client"], Q("printer"), dry_run=True)
    assert r.status == "plan" and r.success and len(r.items) == 1 and r.tier == 1
    assert r.items[0].label == "printer #1"
    assert len(stack["fake"].received) == n0


def test_drives_best_match_via_runmission(stack):
    r = call(stack["client"], Q("printer"))
    assert r.status == "done" and r.success
    assert stack["fake"].received[-1][0][0] == "printer #1"


def test_relation_and_pick(stack):
    r = call(stack["client"], json.dumps({"steps": [{"target": "chair", "anchor": "whiteboard",
                                                     "relation": "near", "pick": "best"}], "stop": False}),
             dry_run=True)
    assert r.status == "plan" and r.items[0].label == "office chair #2"
    r = call(stack["client"], Q("chair"), dry_run=True, pick=3)
    assert r.items[0].label == "office chair #3"


def test_named_place_and_multi_step(stack):
    r = call(stack["client"], Q("dock", "printer", "back"))
    assert r.status == "done"
    assert [lab for lab, _, _ in stack["fake"].received[-1]] == ["dock", "printer #1", "back"]


def test_not_understood_and_not_found(stack):
    r = call(stack["client"], "go to the printer")              # not a query: the JSON backend refuses
    assert r.status == "not_understood" and not r.success and "rephrase" in r.detail
    r = call(stack["client"], Q("forklift"))
    assert r.status == "not_found" and "closest" in r.detail and len(r.candidates) == 3


def test_stop_and_cancel_forwarding(stack):
    stack["fake"].secs = 3.0
    c0 = stack["fake"].canceled
    r = call(stack["client"], Q("printer"), cancel_after=0.8)
    stack["fake"].secs = 0.6
    assert r.status == "canceled"
    t = time.time()
    while stack["fake"].canceled == c0 and time.time() - t < 3.0:
        time.sleep(0.05)
    assert stack["fake"].canceled == c0 + 1
    r = call(stack["client"], Q(stop=True))
    assert r.status == "stopped" and r.success
    r = call(stack["client"], "stop")              # plain text: the JSON backend would refuse it, so this
    assert r.status == "stopped" and r.success     # proves "stop" bypasses the parser entirely
