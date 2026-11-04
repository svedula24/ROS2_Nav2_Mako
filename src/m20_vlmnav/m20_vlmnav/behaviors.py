"""The one move executor of Real-time B: a moves.Move -> Nav2's DriveOnHeading (forward) or Spin (turn), bounded.
Used by the VLM node (stage 1: one move per decision) and by the search node's dispatcher (a typed move sequence).

    allowance   the behavior's own time budget = ideal duration x 2 + MOVE_MARGIN_S (a skid-steer turns at about
                half the commanded rate; a short budget made legitimate moves abort); the wait for the result is the
                allowance + MOVE_MARGIN_S more, then the move is cancelled and reported "timeout"
    result      (ok, detail): ok | behavior status N (aborted: blocked or its own allowance) | canceled | mode |
                timeout: no result from the behavior server | behavior server ... unavailable | not acknowledged |
                rejected. "canceled" / "mode" come from the caller's should_stop() (operator stop, commander veto).
"""
import time
from typing import Callable, Optional, Tuple

from . import moves

# ROS imports live inside the functions that need them: the timing rules above are pure and unit-tested without ROS.

SPIN_RATE = 0.5                 # rad/s assumed for the time allowance (the behavior's own max_rotational_vel is 0.7)
MOVE_MARGIN_S = 5.0             # allowance = ideal duration x 2 + this; the wait for the result = allowance + this
MAX_ALLOWANCE_S = 120.0         # keeps Duration.sec sane whatever the move (moves.py clamps first anyway)


def duration_fields(s: float) -> Tuple[int, int]:
    """(sec, nanosec) of a clamped, non-negative duration (pure; duration() wraps it in the message)."""
    s = max(0.0, min(float(s), MAX_ALLOWANCE_S))
    return int(s), int((s - int(s)) * 1e9)


def duration(s: float):
    from builtin_interfaces.msg import Duration
    sec, nanosec = duration_fields(s)
    return Duration(sec=sec, nanosec=nanosec)


def allowance_s(move: moves.Move, speed: float) -> float:
    ideal = move.amount / speed if move.kind == moves.FORWARD else abs(move.yaw()) / SPIN_RATE
    return ideal * 2.0 + MOVE_MARGIN_S


def make_clients(node, callback_group) -> dict:
    from nav2_msgs.action import DriveOnHeading, Spin
    from rclpy.action import ActionClient
    return {moves.FORWARD: ActionClient(node, DriveOnHeading, "/drive_on_heading", callback_group=callback_group),
            "turn": ActionClient(node, Spin, "/spin", callback_group=callback_group)}


def wait(fut, timeout: float) -> bool:
    t = time.monotonic()
    while not fut.done() and time.monotonic() - t < timeout:
        time.sleep(0.02)
    return fut.done()


def cancel_behavior(node, bgh, res, why: str):
    """Cancel the running behavior and wait (briefly) for it to acknowledge; the behavior server itself sends the
    zero velocity when it stops. Logs whether it really stopped."""
    cf = bgh.cancel_goal_async()
    acked = wait(cf, 3.0) and bool(cf.result().goals_canceling)
    ended = wait(res, 3.0)
    node.get_logger().info(f"move {why}: cancel {'acknowledged' if acked else 'NOT acknowledged'}, "
                           f"behavior {'ended' if ended else 'still running after 3 s'}")


def execute_move(node, clients: dict, move: moves.Move, speed: float,
                 should_stop: Callable[[], Optional[str]],
                 on_active: Callable[[object], None] = lambda _h: None) -> Tuple[bool, str]:
    """Execute one move with a Nav2 behavior and wait for it. should_stop() is polled while the move runs and
    returns None to keep going or the reason to cut the move short ("canceled", "mode"); on_active(handle | None)
    lets the caller remember the goal handle in flight (to cancel it from a crash path)."""
    from nav2_msgs.action import DriveOnHeading, Spin
    allow = allowance_s(move, speed)
    if move.kind == moves.FORWARD:
        client = clients[moves.FORWARD]
        goal = DriveOnHeading.Goal()
        goal.target.x, goal.speed = float(move.amount), float(speed)
    else:
        client = clients["turn"]
        goal = Spin.Goal()
        goal.target_yaw = float(move.yaw())
    goal.time_allowance = duration(allow)
    if not client.wait_for_server(timeout_sec=5.0):
        return False, f"behavior server for {move.kind} unavailable"
    send = client.send_goal_async(goal)
    if not wait(send, 5.0):
        return False, "behavior goal not acknowledged"
    bgh = send.result()
    if bgh is None or not bgh.accepted:
        return False, "behavior rejected the goal"
    on_active(bgh)
    try:
        res = bgh.get_result_async()
        t0 = time.monotonic()
        while not res.done():
            why = should_stop()
            if why:
                cancel_behavior(node, bgh, res, why)
                return False, why
            if time.monotonic() - t0 > allow + MOVE_MARGIN_S:
                cancel_behavior(node, bgh, res, "timeout")
                return False, "timeout: no result from the behavior server"
            time.sleep(0.05)
        status = res.result().status
        return status == 4, "ok" if status == 4 else f"behavior status {status} (aborted: blocked or timed out)"
    finally:
        on_active(None)
