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
amount above the sanity clamp (3 m / 180 degrees) is split into legs (7 m -> 3, 3, 1; 270 degrees -> 180, 90), and
refused above MAX_TYPED_DIST_M / MAX_TYPED_TURN_RAD: a typed number means what it says, never silently less.
Known refusals of this first cut (unknown, nothing moves): an object phrase that contains a motion word or a digit
("the chair on the right", "monitor 2"), questions (a wh-word or an auxiliary verb first, "can" only before a
pronoun, or a "?"), a "stop" with a condition ("stop at the door": not judgeable by dead reckoning), and "stop" on
its own: stopping a RUNNING search is the action's cancel (tools/dev/m20 search stop), not an instruction.
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
# a question: a wh-word first, an auxiliary verb first, or a "?". "can" alone is the exception: it is also a noun
# ("can of soda"), so it opens a question only before a pronoun / determiner ("can you", "can someone", "can the")
_QUESTION = re.compile(r"^\s*(?:what|where|when|who|which|how|why|is|are|do|does|did|could|would|should)\b|"
                       r"^\s*can\s+(?:you|i|we|it|there|this|that|these|those|the|a|an|he|she|they|my|your|our|their|"
                       r"some(?:one|body)|any(?:one|body)|one)\b|"
                       r"\?\s*$")
_STOP = re.compile(r"^\s*stop(?:\s+(?:here|now))?\s*\.?\s*$")
MAX_TYPED_DIST_M = 30.0           # a typed distance above this is refused (nothing moves), not silently shortened
MAX_TYPED_TURN_RAD = 4 * math.pi  # a typed turn above two full turns likewise
_LEG_EPS = {moves.FORWARD: 0.005, "turn": math.radians(0.5)}     # a remainder below this is not a leg (cm / half a degree)
# "stop" is not here: "bus stop" and "stop sign" name objects; a stop that is a command is caught by the move
# grammar first (bare "stop" -> no journey; "stop at the door" -> refused as a condition)
_MOTION_WORDS = re.compile(r"\b(turn|move|forward|backward|back|ahead|straight|left|right|degrees?|met(?:er|re)s?|cm|"
                           r"corridor|hallway|walk|drive|go|then)\b|\d")


@dataclass
class Instruction:
    kind: str
    seq: List[moves.Move] = field(default_factory=list)     # the moves, in order (named seq: `moves` is the module)
    phrase: Optional[str] = None


def _legs(kind: str, amount: float) -> Optional[List[moves.Move]]:
    """A typed amount as moves of at most MAX_DIST_M / MAX_TURN_RAD each: "walk 7 meters" -> 3 m, 3 m, 1 m;
    "turn right 270 degrees" -> 180, 90. (The clamp in moves.py guards against VLM nonsense; a typed number means
    what it says, so it is split, or refused (None) above MAX_TYPED_*.) A remainder under _LEG_EPS is dropped."""
    if kind == moves.FORWARD:
        cap, limit, eps = moves.MAX_DIST_M, MAX_TYPED_DIST_M, _LEG_EPS[moves.FORWARD]
    else:
        cap, limit, eps = moves.MAX_TURN_RAD, MAX_TYPED_TURN_RAD, _LEG_EPS["turn"]
    if amount <= 0 or amount > limit:
        return None
    out = []
    while amount > eps:
        leg = min(amount, cap)
        out.append(moves.Move(kind, leg))
        amount -= leg
    return out or None


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
        v = float(m.group(1))
        return _legs(moves.FORWARD, v / 100.0 if m.group(2).startswith("c") else v)
    r = moves.requested(c)                                    # the parser's number, unclamped (one regex for it)
    if r is None or r[0] == moves.STOP:                       # a "stop" with other words around it
        return None
    return _legs(*r)


def classify(text: str) -> Instruction:
    chunks = [c for c in _SPLIT.split(text.strip().lower()) if c.strip()]
    if chunks:
        parsed = [_move(c) for c in chunks]
        if all(p is not None for p in parsed):
            seq = [m for p in parsed for m in p if m.kind != moves.STOP]
            # "stop" alone is not a journey, and not an object either (the cancel path is for a running search)
            return Instruction(MOVES, seq=seq) if seq else Instruction(UNKNOWN)
    phrase = phrase_from(text)
    if phrase and not _MOTION_WORDS.search(phrase) and not _QUESTION.search(text.strip().lower()):
        return Instruction(OBJECT, phrase=phrase)
    return Instruction(UNKNOWN)
