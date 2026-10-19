"""m20_vlmnav: Real-time B (NaVILA-style). A VLM picks the next short move from camera frames; Nav2 behaviors execute it.

Serves m20_msgs/action/NavigateNL on /m20/vlmnav/navigate (the same action type as the offline NL node, so no new
interface): text in -> a loop of
    current frame + 7 history frames + the moves executed so far --VLM (tools/realtime_b/cosmos_server.py)-->
    "move forward 75 cm" --moves.parse--> Nav2 behavior: DriveOnHeading (forward) / Spin (turns). No backward move.
    --> next decision
until the VLM says stop or an end condition fires (episode.py). No map is used for decisions.
dry_run=true asks the VLM once and returns the parsed move without moving. One episode at a time: a second goal
while one runs is rejected (two episodes would send conflicting moves).

Moves go straight to Nav2's behavior server, not through m20_missions: the behaviors check the local costmap and
abort before a collision, and the Godot bridge only applies /cmd_vel while the commander says armed, so the
commander's veto holds. This node also ends the episode (status "vetoed") when /m20/mode isn't ASSISTED/AUTONOMOUS.

Inputs:  /camera/image_raw (rgb8), /m20/mode, /odom_true (ground truth, logged for evals only; never shown to the VLM)
Params:  vlm_url (http://127.0.0.1:8090), drive_speed (0.3 m/s), log_dir (~/m20_sim/vlmnav),
         max_decisions (60), max_seconds (300), vlm_timeout_s (30)
Output:  per episode, <log_dir>/<stamp>/steps.jsonl + summary.json + one contact sheet per decision (the 8 frames)
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

from builtin_interfaces.msg import Duration
from m20_msgs.action import NavigateNL
from m20_msgs.msg import RobotMode
from nav2_msgs.action import DriveOnHeading, Spin
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

from . import history, moves
from .episode import Episode, Limits, terminal_action

SPIN_RATE = 0.5                 # rad/s assumed for the time allowance (the behavior's own max_rotational_vel is 0.7)
MOVE_MARGIN_S = 5.0             # allowance = ideal duration x 2 + this; the wait for the result = allowance + this
MAX_ALLOWANCE_S = 120.0         # keeps Duration.sec sane whatever the move (moves.py clamps first anyway)


def yaw_of(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def jpeg_b64(rgb: np.ndarray, quality: int = 85) -> str:
    buf = io.BytesIO()
    PILImage.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def duration(s: float) -> Duration:
    s = max(0.0, min(float(s), MAX_ALLOWANCE_S))
    return Duration(sec=int(s), nanosec=int((s - int(s)) * 1e9))


def allowance_s(move: moves.Move, speed: float) -> float:
    ideal = move.amount / speed if move.kind == moves.FORWARD else abs(move.yaw()) / SPIN_RATE
    return ideal * 2.0 + MOVE_MARGIN_S


class VLMNavNode(Node):
    def __init__(self):
        super().__init__("m20_vlmnav")
        for name, default in (("vlm_url", "http://127.0.0.1:8090"), ("drive_speed", 0.3),
                              ("log_dir", os.path.expanduser("~/m20_sim/vlmnav")),
                              ("max_decisions", 60), ("max_seconds", 300.0), ("vlm_timeout_s", 30.0)):
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value
        self.vlm_url, self.speed, self.log_dir = p("vlm_url"), float(p("drive_speed")), p("log_dir")
        self.limits = Limits(max_decisions=int(p("max_decisions")), max_seconds=float(p("max_seconds")))
        self.vlm_timeout = float(p("vlm_timeout_s"))

        self.cb = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self.frame, self.frame_t = None, 0.0          # newest camera frame (H x W x 3 uint8), monotonic time
        self.true_pose = None
        self.mode_ok = False
        self.active = None                            # behavior goal handle of the move in progress
        self.episode_running = False                  # one episode at a time; claimed in on_goal under this lock
        self.goal_lock = threading.Lock()

        self.create_subscription(Image, "/camera/image_raw", self.on_image, qos_profile_sensor_data,
                                 callback_group=self.cb)
        self.create_subscription(Odometry, "/odom_true", self.on_true, qos_profile_sensor_data, callback_group=self.cb)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(RobotMode, "/m20/mode", self.on_mode, latched, callback_group=self.cb)
        self.behaviors = {
            moves.FORWARD: ActionClient(self, DriveOnHeading, "/drive_on_heading", callback_group=self.cb),
            "turn": ActionClient(self, Spin, "/spin", callback_group=self.cb),
        }
        self.server = ActionServer(self, NavigateNL, "/m20/vlmnav/navigate", execute_callback=self.execute,
                                   goal_callback=self.on_goal, cancel_callback=lambda _gh: CancelResponse.ACCEPT,
                                   callback_group=self.cb)
        self.get_logger().info(f"Real-time B up: /m20/vlmnav/navigate, VLM at {self.vlm_url}, logs in {self.log_dir}")

    # ---------- inputs ----------
    def on_goal(self, g):
        if not g.text.strip():
            return GoalResponse.REJECT
        with self.goal_lock:                          # check-and-set: two goals a few ms apart can't both pass
            if self.episode_running:
                self.get_logger().warn("goal rejected: an episode is already running (send stop first)")
                return GoalResponse.REJECT
            self.episode_running = True
        return GoalResponse.ACCEPT

    def on_image(self, m: Image):
        if m.encoding not in ("rgb8", "bgr8"):
            return
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step // 3, 3)[:, :m.width]
        with self.lock:
            self.frame, self.frame_t = (a[:, :, ::-1] if m.encoding == "bgr8" else a).copy(), time.monotonic()

    def on_true(self, m: Odometry):
        p = m.pose.pose
        with self.lock:
            self.true_pose = (round(p.position.x, 3), round(p.position.y, 3), round(yaw_of(p.orientation), 4))

    def on_mode(self, m: RobotMode):
        self.mode_ok = m.armed and m.mode in (RobotMode.MODE_ASSISTED, RobotMode.MODE_AUTONOMOUS)

    # ---------- VLM ----------
    def ask_vlm(self, instruction: str, frames, executed):
        body = json.dumps({"instruction": instruction, "frames": [jpeg_b64(f) for f in frames],
                           "history": executed}).encode()
        req = urllib.request.Request(self.vlm_url + "/decide", data=body, headers={"Content-Type": "application/json"})
        t = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.vlm_timeout) as r:
                out = json.loads(r.read())
            return out.get("text"), (time.monotonic() - t) * 1000.0, out
        except Exception as e:
            self.get_logger().error(f"VLM request failed: {type(e).__name__}: {e}")
            return None, (time.monotonic() - t) * 1000.0, {"error": f"{type(e).__name__}: {e}"}

    # ---------- moves ----------
    def run_move(self, gh, move: moves.Move):
        """Execute one move with a Nav2 behavior. -> (ok, detail). detail: ok | behavior status N (aborted) |
        canceled | mode | timeout | behavior server ... unavailable | not acknowledged | rejected.
        Bounded: waits at most the time allowance + margin, then cancels and reports "timeout"."""
        allow = allowance_s(move, self.speed)
        if move.kind == moves.FORWARD:
            client = self.behaviors[moves.FORWARD]
            goal = DriveOnHeading.Goal()
            goal.target.x, goal.speed = float(move.amount), float(self.speed)
        else:
            client = self.behaviors["turn"]
            goal = Spin.Goal()
            goal.target_yaw = float(move.yaw())
        goal.time_allowance = duration(allow)
        if not client.wait_for_server(timeout_sec=5.0):
            return False, f"behavior server for {move.kind} unavailable"
        send = client.send_goal_async(goal)
        if not self._wait(send, 5.0):
            return False, "behavior goal not acknowledged"
        bgh = send.result()
        if bgh is None or not bgh.accepted:
            return False, "behavior rejected the goal"
        self.active = bgh
        try:
            res = bgh.get_result_async()
            t0 = time.monotonic()
            while not res.done():
                if gh.is_cancel_requested or not self.mode_ok:
                    why = "canceled" if gh.is_cancel_requested else "mode"
                    self._cancel_behavior(bgh, res, why)
                    return False, why
                if time.monotonic() - t0 > allow + MOVE_MARGIN_S:
                    self._cancel_behavior(bgh, res, "timeout")
                    return False, "timeout: no result from the behavior server"
                time.sleep(0.05)
            status = res.result().status
            return status == 4, "ok" if status == 4 else f"behavior status {status} (aborted: blocked or timed out)"
        finally:
            self.active = None

    def _cancel_behavior(self, bgh, res, why):
        """Cancel the running behavior and wait (briefly) for it to acknowledge; the behavior server itself sends
        the zero velocity when it stops. Logs whether it really stopped."""
        cf = bgh.cancel_goal_async()
        acked = self._wait(cf, 3.0) and bool(cf.result().goals_canceling)
        ended = self._wait(res, 3.0)
        self.get_logger().info(f"move {why}: cancel {'acknowledged' if acked else 'NOT acknowledged'}, "
                               f"behavior {'ended' if ended else 'still running after 3 s'}")

    @staticmethod
    def _wait(fut, timeout):
        t = time.monotonic()
        while not fut.done() and time.monotonic() - t < timeout:
            time.sleep(0.02)
        return fut.done()

    # ---------- the episode ----------
    def execute(self, gh):
        req, res = gh.request, NavigateNL.Result()
        ep = Episode(req.text.strip(), limits=self.limits)
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        out_dir = os.path.join(self.log_dir, stamp)
        os.makedirs(out_dir, exist_ok=True)           # (episode_running was claimed in on_goal)
        self.get_logger().info(f"episode: {ep.instruction!r}{' (dry run)' if req.dry_run else ''} -> {out_dir}")
        try:
            self._run(gh, ep, out_dir)
        except Exception as e:                            # never leave a behavior running or the log unwritten
            self.get_logger().error(f"episode crashed: {type(e).__name__}: {e}")
            ep.cancel("crash")
        finally:
            if self.active is not None:
                try:
                    self.active.cancel_goal_async()
                except Exception:
                    pass
                self.active = None
            with self.goal_lock:
                self.episode_running = False
        summary = ep.summary()
        with open(os.path.join(out_dir, "steps.jsonl"), "w") as f:
            for s in ep.steps:
                f.write(json.dumps(s) + "\n")
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        last = ep.steps[-1]["move"] if ep.steps else None
        res.status = ep.status
        res.success = res.status in ("done", "plan")
        res.interpretation = last or ""
        res.tier = 2                                   # 2 = Real-time B (VLM picks moves)
        res.latency_ms = float(ep.steps[0]["latency_ms"]) if ep.steps else 0.0
        res.detail = json.dumps({**summary, "log": stamp})    # the episode folder's name under log_dir
        self.get_logger().info(f"episode end: {summary}")
        action = terminal_action(res.status, gh.is_cancel_requested)
        if action == "canceled":
            res.status = "canceled"
            gh.canceled()
        elif action == "succeed":
            gh.succeed()
        else:
            gh.abort()
        return res

    def _run(self, gh, ep, out_dir):
        req = gh.request
        observations = []                                   # one frame per decision, since the instruction started
        while ep.check_limits():
            if gh.is_cancel_requested:
                ep.cancel("canceled"); return
            if not req.dry_run and not self.mode_ok:
                ep.cancel("mode"); return
            with self.lock:
                frame, age, pose = self.frame, time.monotonic() - self.frame_t, self.true_pose
            if frame is None or age > 2.0:
                ep.on_decision(None, 0.0, {"error": "no fresh camera frame"}); return
            observations.append(frame)
            frames = [observations[i] for i in history.pick(len(observations))]
            self._feedback(gh, len(ep.steps), "deciding", "")
            reply, latency, raw = self.ask_vlm(ep.instruction, frames, ep.executed_moves())
            self._contact_sheet(frames, os.path.join(out_dir, f"step_{len(ep.steps):03d}.jpg"))
            move = ep.on_decision(reply, latency, {"true_pose": pose, "server_latency_ms": raw.get("latency_ms"),
                                                   "prompt_tokens": raw.get("prompt_tokens"), "error": raw.get("error")})
            self.get_logger().info(f"  step {len(ep.steps) - 1}: {reply!r} -> {move.text() if move else ep.end_reason} "
                                   f"({latency:.0f} ms)")
            if move is None:
                return
            if req.dry_run:
                ep.cancel("dry_run"); return
            if gh.is_cancel_requested or not self.mode_ok:   # a stop that arrived during the VLM call
                ep.on_move_result(False, 0.0, "canceled" if gh.is_cancel_requested else "mode")
                ep.cancel("canceled" if gh.is_cancel_requested else "mode"); return
            self._feedback(gh, len(ep.steps) - 1, "moving", move.text())
            t = time.monotonic()
            ok, detail = self.run_move(gh, move)
            with self.lock:
                after = self.true_pose
            ep.on_move_result(ok, time.monotonic() - t, detail, {"true_pose_after": after})
            if detail in ("canceled", "mode"):
                ep.cancel(detail); return

    def _feedback(self, gh, idx, stage, label):
        fb = NavigateNL.Feedback()
        fb.stage, fb.current_index, fb.current_label, fb.status = stage, idx, label, stage
        gh.publish_feedback(fb)

    def _contact_sheet(self, frames, path):
        try:
            small = [PILImage.fromarray(f).resize((320, 240)) for f in frames]
            sheet = PILImage.new("RGB", (320 * 4, 240 * 2))
            for i, im in enumerate(small):
                sheet.paste(im, ((i % 4) * 320, (i // 4) * 240))
            sheet.save(path, quality=80)
        except Exception as e:                            # logging must never break the episode, but say so
            self.get_logger().warn(f"contact sheet not written ({type(e).__name__}: {e})")


def main():
    rclpy.init()
    node = VLMNavNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
