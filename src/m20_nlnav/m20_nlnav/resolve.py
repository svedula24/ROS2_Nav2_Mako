"""Resolve a parsed command against the world: which object / waypoint, and the goal pose.

Ambiguity policy (one mode): when several objects match, pick the highest-confidence one
(ties -> nearer) and say which; "nearest X" picks by distance instead, and an explicit pick_id
(e.g. `m20 say --pick ID`) overrides both.
"nearest X" is measured from where the robot will be at that step (the previous step's goal),
not from where it is now, so multi-step routes chain sensibly.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .approach import approach_pose
from .tier0 import Command, Target
from .world import Obj, World

MAX_CANDIDATES = 10


@dataclass
class Candidate:
    obj: Obj
    why: str


@dataclass
class Step:
    label: str                                  # MissionItem label + display
    goal: Tuple[float, float, float]            # x, y, yaw (map)
    obj: Optional[Obj] = None
    note: str = ""


@dataclass
class Resolution:
    status: str                                 # plan | not_found | stop
    interpretation: str = ""
    steps: List[Step] = field(default_factory=list)
    candidates: List[Candidate] = field(default_factory=list)
    detail: str = ""


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _heading(frm, to, default=0.0) -> float:
    if frm is None or _dist(frm, to) < 1e-3:
        return default
    return math.atan2(to[1] - frm[1], to[0] - frm[0])


def _obj_label(o: Obj) -> str:
    return f"{o.cls} #{o.id}"


def _choose(t: Target, cands: List[Obj], here, pick_id: int):
    """-> (ordered objects to try, [Candidate])."""
    if pick_id is not None and pick_id >= 0:
        picked = [o for o in cands if o.id == pick_id]
        if picked:
            return picked, [Candidate(picked[0], "picked")]
    if t.nearest and here is not None:
        order = sorted(cands, key=lambda o: _dist(o.xy, here))
        why = [f"{_dist(o.xy, here):.1f} m away" for o in order]
        return order, [Candidate(o, w) for o, w in zip(order, why)]
    order = sorted(cands, key=lambda o: (-o.score, _dist(o.xy, here) if here else 0.0))
    why = [f"confidence {o.score:.2f}" + (f", {_dist(o.xy, here):.1f} m away" if here else "")
           for o in order]
    return order, [Candidate(o, w) for o, w in zip(order, why)]


def resolve(cmd: Command, world: World, pick_id: int = -1) -> Resolution:
    if cmd.action == "stop":
        return Resolution("stop", "stop")
    here = world.pose[:2] if world.pose else None
    start_here = world.pose
    steps: List[Step] = []
    shown: List[Candidate] = []
    for t in cmd.steps:
        if t.kind == "start":
            # a scene's own `start` waypoint wins over the default spawn (the Godot sim's M20
            # does not spawn at the map origin)
            wp = world.waypoints.get("start")
            g = (wp.x, wp.y, wp.yaw if wp.yaw is not None else 0.0) if wp else world.start
            steps.append(Step("start", (g[0], g[1], g[2])))
        elif t.kind == "back":
            if start_here is None:
                return Resolution("not_found", detail="robot pose unknown, can't 'come back'")
            steps.append(Step("back", tuple(start_here)))
        elif t.kind == "waypoint":
            wp = next(w for w in world.waypoints.values() if w.name == t.name)
            yaw = wp.yaw if wp.yaw is not None else _heading(here, (wp.x, wp.y))
            steps.append(Step(wp.name, (wp.x, wp.y, yaw)))
        else:                                                  # object
            cands = [o for o in world.objects if o.cls in t.classes]
            if not cands:
                what = t.noun if t.noun else " / ".join(t.classes)
                weak = [o for o in world.low_conf if o.cls in t.classes]
                if weak:                       # seen, but not confidently enough to drive to
                    best = max(o.score for o in weak)
                    return Resolution("not_found", detail=(
                        f"I've only seen a {what} with low confidence ({best:.2f}, below "
                        f"{world.min_score:.2f}); lower Min confidence to allow it"))
                return Resolution("not_found", detail=f"I haven't seen a {what} yet")
            order, cand_list = _choose(t, cands, here, pick_id)
            shown = cand_list[:MAX_CANDIDATES]
            goal, note, chosen = None, "", None
            for o in order:
                goal, note = approach_pose(o.xy, o.radius, world.grid, from_xy=here)
                if goal is not None:
                    chosen = o
                    break
                if pick_id is not None and pick_id >= 0:
                    break                        # an explicit pick is never swapped
            if goal is None:
                o = order[0]
                return Resolution("not_found", candidates=shown,
                                  detail=f"can't get close to {_obj_label(o)} safely ({note})")
            if chosen is not order[0]:
                note = (note + "; " if note else "") + f"{_obj_label(order[0])} unreachable"
            steps.append(Step(_obj_label(chosen), goal, obj=chosen, note=note))
        here = steps[-1].goal[:2]
    parts = []
    for s in steps:
        if s.obj is not None:
            parts.append(f"{s.label} ({s.obj.score:.2f})" + (f" [{s.note}]" if s.note else ""))
        else:
            parts.append(s.label)
    return Resolution("plan", " → ".join(parts), steps=steps, candidates=shown)
