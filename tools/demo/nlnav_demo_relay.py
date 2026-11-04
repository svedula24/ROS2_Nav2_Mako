"""Demo relay: runs INSIDE docker-commander-1 (the viewer pipes this file to `python3 -u -`), speaks JSON lines.

stdout, one JSON object per line:
  {"type":"hello"}
  {"type":"pose","x":..,"y":..,"yaw":..}           10 Hz, the robot where the STACK believes it is (TF map->base_link)
  {"type":"plan","pts":[[x,y],...]}                 Nav2's global plan whenever it is republished (<= 200 points)
  {"type":"sent","text":"go to the ladder"}         the NL goal went to /m20/nlnav/navigate
  {"type":"feedback","stage":..,"index":..,"label":..,"status":..}
  {"type":"result","status":..,"interpretation":..,"detail":..,"latency_ms":..,"items":[{label,x,y,yaw}],
                    "candidates":[{id,cls,x,y,score,why}]}
stdin, one command per line:
  send <instruction>      send the instruction (the viewer does this once it is recording)
  cancel                  cancel the running NL goal
  quit
"""
import json, math, sys, threading

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray
from tf2_ros import Buffer, TransformListener
from m20_msgs.action import NavigateNL


def out(d):
    sys.stdout.write(json.dumps(d) + "\n")
    sys.stdout.flush()


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Relay(Node):
    def __init__(self):
        super().__init__("m20_demo_relay")
        self.buf = Buffer()
        self.tfl = TransformListener(self.buf, self)
        self.cli = ActionClient(self, NavigateNL, "/m20/nlnav/navigate")
        self.create_subscription(Path, "/plan", self.on_plan, 10)
        # the navigator publishes its resolution (target cube, goal arrow, candidate spheres) BEFORE it drives
        self.create_subscription(MarkerArray, "/m20/nlnav/markers", self.on_markers, 10)
        self.create_timer(0.1, self.tick)
        self.goal_handle = None
        self.last_fb = None
        out({"type": "hello"})

    def tick(self):
        try:
            t = self.buf.lookup_transform("map", "base_link", rclpy.time.Time())
        except Exception:
            return
        tr, q = t.transform.translation, t.transform.rotation
        out({"type": "pose", "x": round(tr.x, 3), "y": round(tr.y, 3), "yaw": round(yaw_of(q), 4)})

    def on_plan(self, msg):
        pts = [(round(p.pose.position.x, 2), round(p.pose.position.y, 2)) for p in msg.poses]
        step = max(1, len(pts) // 200)
        out({"type": "plan", "pts": pts[::step] + ([pts[-1]] if pts and step > 1 else [])})

    def on_markers(self, msg):
        d = {"type": "markers", "targets": [], "goals": [], "candidates": []}
        for m in msg.markers:
            p = m.pose.position
            if m.ns == "targets":
                d["targets"].append({"x": round(p.x, 3), "y": round(p.y, 3), "z": round(p.z, 3),
                                     "sx": round(m.scale.x, 3), "sy": round(m.scale.y, 3), "sz": round(m.scale.z, 3)})
            elif m.ns == "goals":
                d["goals"].append({"x": round(p.x, 3), "y": round(p.y, 3), "yaw": round(yaw_of(m.pose.orientation), 4)})
            elif m.ns == "candidates":
                d["candidates"].append({"x": round(p.x, 3), "y": round(p.y, 3), "z": round(p.z, 3)})
        out(d)

    # ---------- the NL goal ----------
    def send(self, text):
        if not self.cli.wait_for_server(timeout_sec=5.0):
            out({"type": "result", "status": "error", "interpretation": "", "detail": "NL navigation not running",
                 "latency_ms": 0, "items": [], "candidates": []})
            return
        g = NavigateNL.Goal()
        g.text, g.dry_run, g.pick_id = text, False, -1
        fut = self.cli.send_goal_async(g, feedback_callback=self.on_fb)
        fut.add_done_callback(self.on_accepted)
        out({"type": "sent", "text": text})

    def on_accepted(self, fut):
        self.goal_handle = fut.result()
        if not self.goal_handle.accepted:
            out({"type": "result", "status": "error", "interpretation": "", "detail": "goal rejected",
                 "latency_ms": 0, "items": [], "candidates": []})
            return
        self.goal_handle.get_result_async().add_done_callback(self.on_result)

    def on_fb(self, msg):
        f = msg.feedback
        key = (f.stage, f.current_index, f.current_label, f.status)
        if key == self.last_fb:
            return
        self.last_fb = key
        out({"type": "feedback", "stage": f.stage, "index": int(f.current_index), "label": f.current_label,
             "status": f.status})

    def on_result(self, fut):
        r = fut.result().result
        items = []
        for it in r.items:
            p, q = it.goal.pose.position, it.goal.pose.orientation
            items.append({"label": it.label, "x": round(p.x, 3), "y": round(p.y, 3), "yaw": round(yaw_of(q), 4)})
        cands = [{"id": int(c.object_id), "cls": c.cls, "x": round(c.position.x, 3), "y": round(c.position.y, 3),
                  "score": round(float(c.score), 3), "why": c.why} for c in r.candidates]
        out({"type": "result", "status": r.status, "interpretation": r.interpretation, "detail": r.detail,
             "latency_ms": round(float(r.latency_ms)), "items": items, "candidates": cands})
        self.goal_handle = None

    def cancel(self):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()


def main():
    rclpy.init()
    node = Relay()
    stop = threading.Event()
    cmds = []                                   # stdin lines, run on the spinning thread (rclpy is not thread-safe)

    def reader():
        for line in sys.stdin:
            cmds.append(line.strip())
            if line.strip() == "quit":
                break
        stop.set()

    threading.Thread(target=reader, daemon=True).start()
    while rclpy.ok() and not stop.is_set():
        rclpy.spin_once(node, timeout_sec=0.05)
        while cmds:
            line = cmds.pop(0)
            if line.startswith("send "):
                node.send(line[5:].strip())
            elif line == "cancel":
                node.cancel()
    node.cancel()
    rclpy.spin_once(node, timeout_sec=0.2)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
