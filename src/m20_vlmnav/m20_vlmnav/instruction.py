"""What kind of instruction is this, and what does the search do with it (pure: no ROS).

    moves     a move or a sequence of moves the robot can execute by dead reckoning, no camera needed:
              "turn left 90 degrees", "move forward 2 meters, then stop", "turn around and walk 2 meters"
              -> Nav2 Spin / DriveOnHeading, one after the other, through behaviors.execute_move (the one executor,
              shared with stage 1's VLM node)
    object    something to look for: "go to the ladder", "find a trash can", "refrigerator" -> the VLFM-style search
    unknown   neither: a route ("go down the corridor and stop at the first desk on your left"), a question, noise
              -> not_understood, nothing moves (a search for a sentence that names no object wandered 104 m)
The move grammar is moves.py's plus the operator's spellings: "turn around" (180 degrees), "walk/go/drive N metres"
without "forward", chunks joined by "then" / "and" / ",", a trailing "stop" that only closes the sentence. A typed
distance above moves.MAX_DIST_M (3 m) is split into legs of at most 3 m (7 m -> 3, 3, 1), up to MAX_TYPED_DIST_M.
Known refusals of this first cut (unknown, nothing moves): an object phrase that contains a motion word or a digit
("stop sign", "the chair on the right", "monitor 2"), questions (a wh-word, or is/can/do + a pronoun, or a "?"),
a "stop" with a condition ("stop at the door": not judgeable by dead reckoning), and "stop" on its own: stopping a
RUNNING search is the action's cancel (tools/dev/m20 search stop), not an instruction.
"""
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

from . import moves
from .targets import phrase_from

MOVES, OBJECT, UNKNOWN = "moves", "object", "unknown"
_SPLIT = re.compile(r"\s*(?:,|\bthen\b|\band\b|;)\s*")
_AROUND = re.compile(r"^\s*turn\s+(?:a)?round\s*$")
_WALK = re.compile(r"^\s*(?:walk|go|drive|move|advance)\s+(?:forward\s+|ahead\s+|straight\s+)?(?:by\s+)?"
                   r"(\d+(?:\.\d+)?)\s*(cm|centimet(?:er|re)s?|m|met(?:er|re)s?)\b\s*$")
_QUESTION = re.compile(r"^\s*(?:what|where|when|who|which|how|why)\b|"
                       r"^\s*(?:is|are|do|does|did|can|could|would|should)\s+(?:you|i|we|it|there|this|that|the|a|an|he|she|they)\b|"
                       r"\?\s*$")
_STOP = re.compile(r"^\s*stop(?:\s+(?:here|now))?\s*\.?\s*$")
MAX_TYPED_DIST_M = 30.0           # a typed distance above this is refused (nothing moves), not silently shortened
_MOTION_WORDS = re.compile(r"\b(turn|move|forward|backward|back|ahead|straight|left|right|degrees?|met(?:er|re)s?|cm|"
                           r"corridor|hallway|walk|drive|go|stop|then)\b|\d")


@dataclass
class Instruction:
    kind: str
    seq: List[moves.Move] = field(default_factory=list)     # the moves, in order (named seq: `moves` is the module)
    phrase: Optional[str] = None


def _legs(metres: float) -> List[moves.Move]:
    """A typed distance as forward moves of at most MAX_DIST_M each: "walk 7 meters" -> 3 m, 3 m, 1 m. (The clamp
    in moves.py guards against VLM nonsense; a typed 7 m means 7 m, so it is split, not shortened.)"""
    out = []
    while metres > 1e-9:
        leg = min(metres, moves.MAX_DIST_M)
        out.append(moves.Move(moves.FORWARD, leg))
        metres -= leg
    return out


def _move(chunk: str) -> Optional[List[moves.Move]]:
    """One chunk -> its moves (one, or several legs for a long distance), or None. A bare "stop" -> [STOP] (dropped
    by the caller: it only ends the text). "stop at the door" / "stop next to it" is a CONDITION the robot can't
    judge by dead reckoning -> None, so the whole instruction is refused rather than the earlier moves run blind."""
    c = chunk.strip().lower()
    if not c:
        return None
    if _STOP.match(c):
        return [moves.Move(moves.STOP)]
    if _AROUND.match(c):
        return [moves.Move(moves.TURN_LEFT, math.pi)]
    m = _WALK.match(c)
    if m:
        v = float(m.group(1)); metres = v / 100.0 if m.group(2).startswith("c") else v
        if metres <= 0 or metres > MAX_TYPED_DIST_M:
            return None
        return _legs(metres)
    mv = moves.parse(c)
    if mv is None or mv.kind == moves.STOP:                 # a "stop" with other words around it
        return None
    if mv.kind == moves.FORWARD and mv.clamped:               # "move forward 5 m": the typed number, in legs
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*(cm|centimet(?:er|re)s?|m|met(?:er|re)s?)\b", c)
        v = float(m2.group(1)); metres = v / 100.0 if m2.group(2).startswith("c") else v
        return None if metres > MAX_TYPED_DIST_M else _legs(metres)
    return [mv]


def classify(text: str) -> Instruction:
    chunks = [c for c in _SPLIT.split(text.strip().lower()) if c.strip()]
    if chunks:
        parsed = [_move(c) for c in chunks]
        if all(p is not None for p in parsed):
            seq = [m for p in parsed for m in p if m.kind != moves.STOP]
            if seq:                                                # "stop" alone is not a journey
                return Instruction(MOVES, seq=seq)
    phrase = phrase_from(text)
    if phrase and not _MOTION_WORDS.search(phrase) and not _QUESTION.search(text.strip().lower()):
        return Instruction(OBJECT, phrase=phrase)
    return Instruction(UNKNOWN)
