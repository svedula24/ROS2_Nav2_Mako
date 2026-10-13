"""Tier 0: exact-match command parsing (no models).

Handles the fixed phrasings below and returns None for anything else, so a later tier (LLM) can
take over. It never guesses: a phrase with words it doesn't understand ("the RED chair", "the
chair NEXT TO the whiteboard") is rejected, not trimmed.

  stop | cancel | halt | abort
  [go|drive|navigate|head|move|walk] to <target>     take me to <target>
  go back | come back | return | go home              (back to the start; see below)
  go back to <target> | return to <target>
  <step>, then <step> [and then <step> ...]           multi-step
<target>: [the|a|an|my] [nearest|closest] <class or synonym> | <waypoint name> | start / home

"go back"/"come back"/"return" with no target means: as the ONLY step, the start (spawn); as a
LATER step ("... then come back"), where the robot was when the command was given.
"go home" / "return home" always means the start.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .world import classes_for, norm_name

STOP_WORDS = {"stop", "cancel", "halt", "abort", "stop moving", "stop now"}
GO_PREFIXES = ["take me to", "bring me to", "navigate to", "drive to", "head to", "move to",
               "walk to", "go over to", "go to", "goto", "visit", "go"]
BACK_PREFIXES = ["go back to", "come back to", "return to", "head back to", "get back to"]
BACK_ALONE = {"go back", "come back", "return", "head back", "get back", "come back here"}
HOME_ALONE = {"go home", "head home", "come home", "return home", "go back home"}   # always the start
START_NAMES = {"start", "the start", "home", "base", "spawn", "the beginning", "where you started",
               "the starting point", "starting point", "the starting position"}
ARTICLES = ("the ", "a ", "an ", "my ", "some ", "any ")
NEAREST = ("nearest ", "closest ")
NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
                "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15"}
THEN = re.compile(r"\s*(?:,\s*)?(?:\band then\b|\bthen\b|,|;)\s*")


@dataclass
class Target:
    kind: str                         # object | waypoint | start | back
    noun: str = ""                    # what the user said
    classes: List[str] = field(default_factory=list)   # object: candidate detector classes
    nearest: bool = False             # "nearest/closest X"
    name: str = ""                    # waypoint: display name


@dataclass
class Command:
    action: str                       # goto | stop
    steps: List[Target] = field(default_factory=list)


def _clean(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[!?.]+$", "", t)
    t = re.sub(r"^(please|ok|okay|hey robot|robot|can you|could you|now)[, ]+", "", t)
    t = re.sub(r"[, ]+(please|now|thanks|thank you)$", "", t)
    return " ".join(t.split())


def _numbers(s: str) -> str:
    return " ".join(NUMBER_WORDS.get(w, w) for w in s.split())


def _target(phrase: str, known_classes: List[str], waypoints: dict) -> Optional[Target]:
    p = phrase.strip()
    if p in START_NAMES:
        return Target("start", noun=p)
    for a in ARTICLES:
        if p.startswith(a):
            p = p[len(a):]
            break
    if p in START_NAMES:
        return Target("start", noun=p)
    wp = waypoints.get(norm_name(_numbers(p)))
    if wp is not None:
        return Target("waypoint", noun=phrase.strip(), name=wp.name)
    nearest = False
    for n in NEAREST:
        if p.startswith(n):
            p, nearest = p[len(n):], True
            break
    cls = classes_for(p, known_classes)
    if cls:
        return Target("object", noun=p, classes=cls, nearest=nearest)
    return None


def _step(s: str, known_classes, waypoints, first: bool) -> Optional[Target]:
    if s in HOME_ALONE:
        return Target("start", noun=s)
    if s in BACK_ALONE:
        return Target("start" if first else "back", noun=s)
    for pre in BACK_PREFIXES:
        if s.startswith(pre + " "):
            return _target(s[len(pre) + 1:], known_classes, waypoints)
    for pre in GO_PREFIXES:
        if s.startswith(pre + " "):
            return _target(s[len(pre) + 1:], known_classes, waypoints)
    if not first:                    # "go to the printer, then the ladder" - bare later steps
        return _target(s, known_classes, waypoints)
    return None


def parse(text: str, known_classes: List[str], waypoints: dict) -> Optional[Command]:
    """Command, or None if tier 0 can't parse ALL of the text."""
    t = _clean(text)
    if not t:
        return None
    if t in STOP_WORDS:
        return Command("stop")
    parts = [p for p in THEN.split(t) if p]
    steps = []
    for i, part in enumerate(parts):
        tgt = _step(part, known_classes, waypoints, first=(i == 0))
        if tgt is None:
            return None
        steps.append(tgt)
    return Command("goto", steps)
