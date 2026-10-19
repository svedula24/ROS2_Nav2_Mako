"""One Real-time B episode as a state machine, without ROS: decide -> move -> decide ... until an end condition.

The node feeds it what happened (a VLM reply, a move's result, a cancel, a mode change); it answers what to do next
and keeps the per-step log. End reasons and what the action reports:
    stop            the VLM said stop                                  -> status "done"
    unparseable     the reply named no single valid move (never guessed) -> "error"
    max_decisions   60 decisions without a stop                       -> "error"
    timeout         5 minutes without a stop                          -> "error"
    blocked         3 moves in a row aborted by Nav2 (an obstacle)    -> "error"
    move_error      the behavior server failed (unavailable, rejected, lost, timed out) -> "error"
    vlm_error       the VLM server failed or timed out                -> "error"
    canceled        operator stop                                     -> "canceled"
    mode            the commander left ASSISTED (veto) / never armed  -> "vetoed"
    dry_run         one decision, nothing executed (dry_run=true)     -> "plan"
A blocked move alone doesn't end the episode: the next decision sees a fresh frame and may choose differently.
A move's result is one of ok | blocked (the behavior aborted: obstacle or its own time allowance) | canceled (cut
short on purpose: operator stop or veto) | error (infrastructure: the server, not the world).
"""
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from . import moves

STATUS = {"stop": "done", "canceled": "canceled", "mode": "vetoed", "dry_run": "plan"}
INTERRUPTED = ("canceled", "mode")
INFRA = ("behavior server", "not acknowledged", "rejected", "timeout", "lost")


def move_result_kind(ok: bool, detail: str) -> str:
    if ok:
        return "ok"
    if detail in INTERRUPTED:
        return "canceled"
    if any(k in detail for k in INFRA):
        return "error"
    return "blocked"


def terminal_action(status: str, cancel_requested: bool) -> str:
    """Which action-server call ends the goal: the goal may only be reported canceled if a cancel was requested
    (rcl_action rejects EXECUTING -> CANCELED otherwise, and the client would get an empty result)."""
    if cancel_requested:
        return "canceled"
    return "succeed" if status in ("done", "plan") else "abort"


@dataclass
class Limits:
    max_decisions: int = 60
    max_seconds: float = 300.0
    max_blocked_in_row: int = 3


@dataclass
class Episode:
    instruction: str
    limits: Limits = field(default_factory=Limits)
    clock: Callable[[], float] = time.monotonic
    steps: List[dict] = field(default_factory=list)
    end_reason: Optional[str] = None
    _t0: float = 0.0
    _blocked_run: int = 0

    def __post_init__(self):
        self._t0 = self.clock()

    @property
    def done(self) -> bool:
        return self.end_reason is not None

    @property
    def status(self) -> str:
        return STATUS.get(self.end_reason, "error") if self.end_reason else "running"

    def elapsed(self) -> float:
        return self.clock() - self._t0

    def _end(self, reason: str) -> None:
        if self.end_reason is None:
            self.end_reason = reason

    def check_limits(self) -> bool:
        """Call before each decision. -> True if the episode may continue."""
        if self.done:
            return False
        if len(self.steps) >= self.limits.max_decisions:
            self._end("max_decisions")
        elif self.elapsed() >= self.limits.max_seconds:
            self._end("timeout")
        return not self.done

    def executed_moves(self) -> List[str]:
        """What the robot did so far, as text for the VLM's prompt (its memory of actions; the frames are its memory
        of what it saw). Blocked moves are marked; canceled/error ones are left out (they say nothing about the world)."""
        out = []
        for s in self.steps:
            if s.get("move") and s.get("result") in ("ok", "blocked"):
                out.append(s["move"] + (" (blocked by an obstacle)" if s["result"] == "blocked" else ""))
        return out

    def on_decision(self, reply: Optional[str], latency_ms: float, extra: Optional[dict] = None) -> Optional[moves.Move]:
        """Record one VLM answer. -> the move to execute, or None if the episode just ended (stop/unparseable/error)."""
        step = {"i": len(self.steps), "t": round(self.elapsed(), 2), "reply": reply, "latency_ms": round(latency_ms, 1)}
        step.update(extra or {})
        self.steps.append(step)
        if reply is None:
            step["move"] = None
            self._end("vlm_error")
            return None
        move = moves.parse(reply)
        step["move"] = move.text() if move else None
        if move is None:
            self._end("unparseable")
            return None
        step["kind"], step["amount"], step["clamped"] = move.kind, round(move.amount, 4), move.clamped
        step["cap_would_trigger"] = moves.cap_would_trigger(move)
        if move.kind == moves.STOP:
            self._end("stop")
            return None
        return move

    def on_move_result(self, ok: bool, duration_s: float, detail: str = "", extra: Optional[dict] = None) -> None:
        """Record how the last move went. ok = Nav2 finished it; otherwise `detail` says why (see move_result_kind)."""
        step = self.steps[-1]
        kind = move_result_kind(ok, detail)
        step.update({"result": kind, "move_s": round(duration_s, 2), "detail": detail})
        step.update(extra or {})
        if kind == "error":
            self._end("move_error")
        elif kind in ("ok", "blocked"):
            self._blocked_run = 0 if ok else self._blocked_run + 1
            if self._blocked_run >= self.limits.max_blocked_in_row:
                self._end("blocked")

    def cancel(self, reason: str = "canceled") -> None:
        self._end(reason)

    def summary(self) -> dict:
        kinds = [s.get("kind") for s in self.steps if s.get("kind")]
        return {
            "instruction": self.instruction, "status": self.status, "end_reason": self.end_reason,
            "decisions": len(self.steps), "seconds": round(self.elapsed(), 1),
            "moves": {k: kinds.count(k) for k in sorted(set(kinds))},
            "blocked_moves": sum(1 for s in self.steps if s.get("result") == "blocked"),
            "clamped_moves": sum(1 for s in self.steps if s.get("clamped")),
            "cap_would_trigger": sum(1 for s in self.steps if s.get("cap_would_trigger")),
            "median_latency_ms": _median([s["latency_ms"] for s in self.steps if s.get("reply") is not None]),
        }


def _median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else round((xs[m - 1] + xs[m]) / 2, 1)
