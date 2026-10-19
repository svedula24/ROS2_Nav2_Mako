"""The move vocabulary of Real-time B (NaVILA-style): what the VLM may answer, and how its text becomes a move.

The VLM answers in words, one move per decision:
    "move forward 75 cm" | "turn left 30 degrees" | "turn right 15 degrees" | "stop"
NaVILA's set: forward / turn / stop. No backward move (removed 2026-10-15): the camera can't see behind the robot, so
the VLM would be reversing blind; to get out of a dead end it turns around instead. A reply asking to move backward
matches no move and parses to None like any other unknown action. parse() is deliberately forgiving about phrasing
("The next action is to move forward 0.75 m.")
and strict about meaning: no number, a zero amount, no recognisable action, or MORE THAN ONE move in the reply
-> None (the episode then ends: an unparseable or ambiguous reply is never guessed at). A reasoning block
(<think>...</think>) is ignored; an <answer>...</answer> block, if present, is the only part read.

Two different bounds, deliberately apart:
  - CAP_*: the 1 m / 45 deg per-move cap discussed in the design. NOT applied (2026-10-15: no move limits yet); only
    recorded per move so the evals show how often it would have bitten.
  - MAX_*: a sanity clamp (3 m / 180 deg). Applied: it keeps a nonsense reply ("move forward 50 m", "turn 3600
    degrees") from producing a 5-minute drive, ten spins, or an int32 overflow in the behavior's time allowance.
    A clamped move is flagged so the log shows it.
"""
import math
import re
from dataclasses import dataclass
from typing import Optional

FORWARD, TURN_LEFT, TURN_RIGHT, STOP = "forward", "turn_left", "turn_right", "stop"

CAP_DIST_M = 1.0                       # recorded only
CAP_TURN_RAD = math.radians(45.0)
MAX_DIST_M = 3.0                       # applied (sanity clamp)
MAX_TURN_RAD = math.pi

_NUM = r"(\d+(?:\.\d+)?)"
_DIST_UNIT = r"(cm|centimet(?:er|re)s?|m|met(?:er|re)s?)\b"
_PATTERNS = [
    (re.compile(r"\b(forward|forwards|ahead)\s*(?:by\s*)?" + _NUM + r"\s*" + _DIST_UNIT), "drive"),
    (re.compile(r"\bturn\s*(left|right)\s*(?:by\s*)?" + _NUM + r"\s*(?:°|deg(?:ree)?s?\b)"), "turn"),
    (re.compile(r"\bstop\b"), "stop"),
]
_THINK = re.compile(r"<think>.*?(</think>|$)", re.S)
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)


@dataclass(frozen=True)
class Move:
    kind: str            # forward | turn_left | turn_right | stop
    amount: float = 0.0  # metres for forward, radians (> 0) for turns, 0 for stop
    clamped: bool = False  # the reply asked for more than MAX_*; the amount was reduced to it

    def text(self) -> str:
        if self.kind == STOP:
            return "stop"
        if self.kind == FORWARD:
            return f"move forward {round(self.amount * 100):d} cm"
        return f"turn {'left' if self.kind == TURN_LEFT else 'right'} {round(math.degrees(self.amount)):d} degrees"

    def yaw(self) -> float:
        """Signed turn for Nav2's Spin: left = +yaw (counter-clockwise, ROS convention)."""
        return self.amount if self.kind == TURN_LEFT else -self.amount if self.kind == TURN_RIGHT else 0.0


def _clean(text: str) -> str:
    t = text.lower().replace("º", "°")                 # º (ordinal) typed for °
    t = _THINK.sub(" ", t)                                   # reasoning, finished or truncated, is not an answer
    m = _ANSWER.search(t)
    return m.group(1) if m else t


def parse(text: str) -> Optional[Move]:
    """The one move named in the reply, or None: no move, a zero amount, or several moves (ambiguous)."""
    if not text:
        return None
    t = _clean(text)
    found = [(m, kind) for rx, kind in _PATTERNS for m in rx.finditer(t)]
    if len(found) != 1:
        return None
    m, kind = found[0]
    if kind == "stop":
        return Move(STOP)
    value = float(m.group(2))
    if value <= 0:
        return None
    if kind == "drive":
        metres = value / 100.0 if m.group(3).startswith("c") else value
        return Move(FORWARD, min(metres, MAX_DIST_M), metres > MAX_DIST_M)
    rad = math.radians(value)
    return Move(TURN_LEFT if m.group(1) == "left" else TURN_RIGHT, min(rad, MAX_TURN_RAD), rad > MAX_TURN_RAD)


def cap_would_trigger(move: Move) -> bool:
    """Would a 1 m / 45 deg per-move cap have shortened this move? (logged, never applied)"""
    if move.kind == FORWARD:
        return move.amount > CAP_DIST_M + 1e-9
    if move.kind in (TURN_LEFT, TURN_RIGHT):
        return move.amount > CAP_TURN_RAD + 1e-9
    return False
