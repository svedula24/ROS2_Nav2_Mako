"""Real-time B episode state machine: end conditions and the per-step log."""
from m20_vlmnav.episode import Episode, Limits, move_result_kind, terminal_action


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def ep(**limits):
    c = Clock()
    return Episode("find the ladder", limits=Limits(**limits) if limits else Limits(), clock=c), c


def test_stop_ends_done_and_moves_are_logged():
    e, c = ep()
    m = e.on_decision("move forward 75 cm", 800.0)
    assert m.kind == "forward" and not e.done
    c.t = 3.0
    e.on_move_result(True, 2.1)
    assert e.on_decision("stop", 700.0) is None
    assert e.done and e.status == "done" and e.end_reason == "stop"
    s = e.summary()
    assert s["decisions"] == 2 and s["moves"] == {"forward": 1, "stop": 1} and s["median_latency_ms"] == 750.0
    assert e.steps[0]["result"] == "ok" and e.steps[0]["move"] == "move forward 75 cm"


def test_unparseable_reply_ends_the_episode_without_guessing():
    e, _ = ep()
    assert e.on_decision("I think the ladder is somewhere ahead", 900.0) is None
    assert e.status == "error" and e.end_reason == "unparseable" and e.steps[0]["move"] is None


def test_vlm_failure_is_an_error_not_a_move():
    e, _ = ep()
    assert e.on_decision(None, 30000.0) is None and e.end_reason == "vlm_error"


def test_one_blocked_move_continues_three_in_a_row_end():
    e, _ = ep()
    for i in range(2):
        e.on_decision("move forward 1 m", 500.0)
        e.on_move_result(False, 0.4, "collision ahead")
        assert not e.done
    e.on_decision("turn left 30 degrees", 500.0)
    e.on_move_result(True, 1.0)                     # an ok move resets the run
    for i in range(3):
        e.on_decision("move forward 1 m", 500.0)
        e.on_move_result(False, 0.4)
    assert e.end_reason == "blocked" and e.summary()["blocked_moves"] == 5


def test_limits_on_decisions_and_time():
    e, _ = ep(max_decisions=2)
    for _ in range(2):
        assert e.check_limits()
        e.on_decision("turn right 15 degrees", 500.0)
        e.on_move_result(True, 0.5)
    assert not e.check_limits() and e.end_reason == "max_decisions"
    e, c = ep(max_seconds=10)
    c.t = 10.0
    assert not e.check_limits() and e.end_reason == "timeout"


def test_cancel_and_mode_veto_report_canceled_and_first_reason_wins():
    e, _ = ep()
    e.cancel("mode")
    e.cancel("canceled")
    assert e.end_reason == "mode" and e.status == "vetoed"      # a veto is reported as such, not as an operator stop


def test_operator_stop_mid_move_is_canceled_not_blocked():
    e, _ = ep()
    e.on_decision("move forward 5 m", 500.0)
    e.on_move_result(False, 3.1, "canceled")
    e.cancel("canceled")
    assert e.status == "canceled" and e.steps[0]["result"] == "canceled"
    assert e.summary()["blocked_moves"] == 0


def test_dry_run_reports_plan():
    e, _ = ep()
    assert e.on_decision("turn left 90 degrees", 500.0).kind == "turn_left"
    e.cancel("dry_run")
    assert e.status == "plan" and e.summary()["cap_would_trigger"] == 1


def test_cap_counter():
    e, _ = ep()
    e.on_decision("move forward 2 m", 500.0)
    e.on_move_result(True, 4.0)
    e.on_decision("turn left 90 degrees", 500.0)
    assert e.summary()["cap_would_trigger"] == 2


def test_terminal_action_never_cancels_without_a_cancel_request():
    """rcl_action rejects EXECUTING -> CANCELED unless a cancel was requested: a veto (status 'vetoed') or an
    operator stop that raced the end must abort, not cancel, or the client gets an empty result."""
    assert terminal_action("vetoed", cancel_requested=False) == "abort"
    assert terminal_action("canceled", cancel_requested=True) == "canceled"
    assert terminal_action("done", cancel_requested=False) == "succeed"
    assert terminal_action("plan", cancel_requested=False) == "succeed"
    assert terminal_action("error", cancel_requested=False) == "abort"
    assert terminal_action("done", cancel_requested=True) == "canceled"      # a stop that arrived as it finished


def test_mode_veto_reports_vetoed_and_executed_moves_skip_canceled_ones():
    e, _ = ep()
    e.on_decision("move forward 60 cm", 500.0)
    e.on_move_result(True, 2.0)
    e.on_decision("move forward 60 cm", 500.0)
    e.on_move_result(False, 0.4, "behavior status 6 (aborted: blocked or timed out)")
    e.on_decision("turn left 30 degrees", 500.0)
    e.on_move_result(False, 0.1, "mode")
    e.cancel("mode")
    assert e.status == "vetoed" and e.end_reason == "mode"
    assert e.executed_moves() == ["move forward 60 cm", "move forward 60 cm (blocked by an obstacle)"]


def test_infrastructure_failures_are_errors_not_obstacles():
    assert move_result_kind(False, "behavior server for forward unavailable") == "error"
    assert move_result_kind(False, "behavior goal not acknowledged") == "error"
    assert move_result_kind(False, "behavior rejected the goal") == "error"
    assert move_result_kind(False, "timeout: no result from the behavior server") == "error"
    assert move_result_kind(False, "behavior status 6 (aborted: blocked or timed out)") == "blocked"
    assert move_result_kind(False, "canceled") == "canceled" and move_result_kind(True, "ok") == "ok"
    e, _ = ep()
    e.on_decision("move forward 60 cm", 500.0)
    e.on_move_result(False, 5.0, "behavior server for forward unavailable")
    assert e.end_reason == "move_error" and e.status == "error" and e.summary()["blocked_moves"] == 0
