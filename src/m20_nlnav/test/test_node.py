"""ROS-level test of nlnav_node (real rclpy; run by `colcon test` in the CI image).

Fixture world on real topics (object map JSON, odometry, an all-free costmap), a scene yaml with
waypoints, and a FAKE RunMission server standing in for m20_missions. Checks the NavigateNL
contract end to end: plan / not_found / dry run / explicit pick, driving through RunMission, and cancel
forwarding.
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
from std_msgs.msg import String
from m20_msgs.action import NavigateNL, RunMission

from m20_nlnav.nlnav_node import NLNavNode

OBJECTS = {"frame": [], "objects": [
    {"id": 1, "cls": "printer", "xyz": [4.0, 0.0, 0.5], "n": 9, "score": 0.82, "size": [0.6, 0.5, 1.0]},
    {"id": 2, "cls": "office chair", "xyz": [2.0, 3.0, 0.5], "n": 6, "score": 0.71, "size": [0.6, 0.6, 1.0]},
    {"id": 3, "cls": "office chair", "xyz": [-3.0, 1.0, 0.5], "n": 8, "score": 0.90, "size": [0.6, 0.6, 1.0]},
]}


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
    scene = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    scene.write("waypoints:\n  start: [0.0, 0.0, 0.0]\n  stop 3: [0.8, -8.1]\n")
    scene.close()
    rclpy.init(args=["--ros-args", "-p", f"scene_yaml:={scene.name}"])
    nl = NLNavNode()
    fake = FakeMission()
    io = Node("fixture_io")
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
    pub_obj = io.create_publisher(String, "/detections/objects", 10)
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
            pub_obj.publish(String(data=json.dumps(OBJECTS)))
            o = Odometry()
            o.header.frame_id = "map"
            o.pose.pose.orientation.w = 1.0
            pub_odom.publish(o)
            time.sleep(0.1)
    threading.Thread(target=feed, daemon=True).start()
    client = ActionClient(io, NavigateNL, "/m20/nlnav/navigate")
    assert client.wait_for_server(timeout_sec=10.0)
    deadline = time.time() + 10.0                    # wait until the node has seen the world
    while time.time() < deadline and (nl.objects_json is None or nl.pose is None or nl.grid is None):
        time.sleep(0.1)
    yield {"client": client, "fake": fake, "nl": nl}
    rclpy.shutdown()
    os.unlink(scene.name)


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


def test_dry_run_plan_does_not_drive(stack):
    n0 = len(stack["fake"].received)
    r = call(stack["client"], "go to the printer", dry_run=True)
    assert r.status == "plan" and r.success and len(r.items) == 1
    assert r.items[0].label == "printer #1"
    assert len(stack["fake"].received) == n0


def test_drives_highest_confidence_via_runmission(stack):
    r = call(stack["client"], "go to the chair")
    assert r.status == "done" and r.success
    assert stack["fake"].received[-1][0][0] == "office chair #3"


def test_several_matches_are_listed_and_pick_overrides(stack):
    r = call(stack["client"], "go to the chair", dry_run=True)
    assert r.status == "plan" and r.items[0].label == "office chair #3"      # highest confidence
    assert {c.object_id for c in r.candidates} == {2, 3}
    r = call(stack["client"], "go to the chair", dry_run=True, pick=2)
    assert r.status == "plan" and r.items[0].label == "office chair #2"


def test_waypoint_and_multi_step(stack):
    r = call(stack["client"], "go to stop 3 then go to the printer then come back")
    assert r.status == "done"
    labels = [lab for lab, _, _ in stack["fake"].received[-1]]
    assert labels == ["stop 3", "printer #1", "back"]


def test_not_understood_and_not_seen(stack):
    r = call(stack["client"], "go to the red chair")
    assert r.status == "not_found" and not r.success and "didn't understand" in r.detail
    r = call(stack["client"], "go to the ladder")
    assert r.status == "not_found" and "haven't seen" in r.detail


def test_cancel_is_forwarded_to_the_mission(stack):
    stack["fake"].secs = 3.0
    c0 = stack["fake"].canceled
    r = call(stack["client"], "go to the printer", cancel_after=0.8)
    stack["fake"].secs = 0.6
    assert r.status == "canceled"
    t = time.time()
    while stack["fake"].canceled == c0 and time.time() - t < 3.0:
        time.sleep(0.05)
    assert stack["fake"].canceled == c0 + 1
