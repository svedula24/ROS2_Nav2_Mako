"""Structured query -> goal poses, against the offline object map (no LLM here: plain, deterministic code).

Per step:
  1. Waypoint names first (the scene yaml's named places; "start"; "back" = where the command began).
  2. Score every object for the target phrase with OWLv2's own head on the stored embeddings
     (object_map.owl_prob), so any phrase works, not just the mapping vocabulary.
  3. Relation: the best-scoring anchor that satisfies near / left_of / right_of / in_front_of / behind in 3D,
     as seen from where the robot is at that step.
  4. Pick: "best" (highest score) or "nearest" (closest above threshold).
  5. Decide: clear winner -> go; close scores -> go to the best and say so; nothing above threshold ->
     not_found with the closest matches (the robot doesn't move).
  6. Approach pose on the costmap (approach.py), then an optional reachability check (reach_fn).
"""
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from .approach import approach_pose
from .llm import Query
from .object_map import MapObject, ObjectMap, normalize, owl_prob
from .world import DETECTOR_CLASSES, classes_for, norm_name

NEAR_M, REL_MAX_M, LATERAL_M = 1.5, 3.0, 1.5
# "best": if the top match can't be reached, only fall back to a candidate scoring at least this fraction of it.
# Without a floor the robot silently went to a 0.55 ladder when stale costmap marks hid the 0.88 and 0.86 ones
# (2026-10-14). Uncalibrated: chosen so near-ties (0.86 vs 0.88) still fall back and clear downgrades don't.
FALLBACK_MIN_RATIO = 0.8


@dataclass
class Candidate:
    obj: MapObject
    score: float
    why: str = ""


@dataclass
class GoalStep:
    label: str
    goal: Tuple[float, float, float]
    obj: Optional[MapObject] = None
    note: str = ""


@dataclass
class Resolution:
    status: str                                  # plan | not_found | stop
    interpretation: str = ""
    steps: List[GoalStep] = field(default_factory=list)
    candidates: List[Candidate] = field(default_factory=list)
    detail: str = ""


# ------------------------------------------------------------------ text encoders
class OwlTextEncoder:
    """OWLv2's text tower (same space as the stored class embeddings). CPU is plenty: one phrase per query."""

    def __init__(self, model_name="google/owlv2-base-patch16-ensemble", template="a photo of a {}"):
        import torch
        from transformers import Owlv2Model, Owlv2Processor
        self.torch, self.template = torch, template
        self.proc = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2Model.from_pretrained(model_name).eval()
        self.cache = {}

    def __call__(self, phrase: str) -> np.ndarray:
        if phrase not in self.cache:
            with self.torch.no_grad():
                t = self.proc(text=[self.template.format(phrase)], return_tensors="pt")
                self.cache[phrase] = normalize(self.model.get_text_features(**t)[0].numpy())
        return self.cache[phrase]


class LabelEncoder:
    """Torch-free stand-in: a phrase that names a detector class (or synonym) -> that class's one-hot vector;
    anything else -> a zero-ish vector. For CI, and for object maps built with one-hot label embeddings."""
    dim = 512

    def __call__(self, phrase: str) -> np.ndarray:
        v = np.full(self.dim, 1e-6, np.float32)
        for c in classes_for(phrase, DETECTOR_CLASSES):
            v[DETECTOR_CLASSES.index(c)] = 1.0
        return normalize(v)


def label_embedding(label: str) -> np.ndarray:
    return LabelEncoder()(label)


# ------------------------------------------------------------------ geometry
def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def relation_ok(rel: str, cand: MapObject, anchor: MapObject, viewer) -> bool:
    """Is `cand` <rel> `anchor`, as seen from `viewer` (x, y)? 2D, in the map frame."""
    c, a = np.array(cand.xy), np.array(anchor.xy)
    gap = _dist(c, a) - cand.radius - anchor.radius
    if rel == "near":
        return gap <= NEAR_M
    if _dist(c, a) > REL_MAX_M + cand.radius + anchor.radius:
        return False
    v = np.array(viewer, float)
    va = a - v                                    # viewer -> anchor
    n = np.linalg.norm(va)
    if n < 1e-6:
        return rel == "near"
    u = va / n
    d = c - a
    along, lateral = float(d @ u), float(u[0] * d[1] - u[1] * d[0])     # +lateral = to the viewer's left
    if rel == "left_of":
        return lateral > 0.2 and abs(along) < LATERAL_M + anchor.radius
    if rel == "right_of":
        return lateral < -0.2 and abs(along) < LATERAL_M + anchor.radius
    if rel == "in_front_of":
        return along < -0.2 and abs(lateral) < LATERAL_M
    if rel == "behind":
        return along > 0.2 and abs(lateral) < LATERAL_M
    return False


# ------------------------------------------------------------------ resolve
def resolve(q: Query, omap: ObjectMap, encoder: Callable[[str], np.ndarray], waypoints: dict, pose=None,
            grid=None, start=(0.0, 0.0, 0.0), threshold=0.10, margin=0.08, pick_id: int = -1,
            reach_fn: Optional[Callable[[Tuple[float, float, float]], bool]] = None) -> Resolution:
    if q.stop:
        return Resolution("stop", "stop")
    here = pose[:2] if pose else None
    E, shift, scale = omap.matrix()
    steps: List[GoalStep] = []
    shown: List[Candidate] = []
    for s in q.steps:
        key = norm_name(s.target)
        wp = waypoints.get(key)
        if s.target in ("back", "where you were", "here") and pose is not None:
            steps.append(GoalStep("back", tuple(pose)))
        elif wp is not None:
            yaw = wp.yaw if wp.yaw is not None else (math.atan2(wp.y - here[1], wp.x - here[0]) if here else 0.0)
            steps.append(GoalStep(wp.name, (wp.x, wp.y, yaw)))
        elif key in ("start", "home", "dock", "spawn"):
            g = waypoints.get("start")
            steps.append(GoalStep("start", (g.x, g.y, g.yaw or 0.0) if g else tuple(start)))
        else:
            if not omap.objects:
                return Resolution("not_found", detail="no object map loaded for this site")
            p = owl_prob(E, encoder(s.target), shift, scale)
            pa = owl_prob(E, encoder(s.anchor), shift, scale) if s.anchor else None
            order = np.argsort(-p)
            closest = [Candidate(omap.objects[i], float(p[i]), f"{p[i]:.2f}") for i in order[:3]]
            cands = []
            forced = [o for o in omap.objects if pick_id is not None and pick_id >= 0 and o.id == pick_id]
            if forced:                                   # the operator chose (m20 say --pick ID)
                cands = [Candidate(forced[0], 1.0, "picked")]
            for i in ([] if forced else np.nonzero(p >= threshold)[0]):
                o, sc, why = omap.objects[i], float(p[i]), f"{s.target} {p[i]:.2f}"
                if s.anchor:
                    viewer = here if here is not None else o.xy
                    best_a = max(((float(pa[j]), j) for j in np.nonzero(pa >= threshold)[0]
                                  if j != i and relation_ok(s.relation or "near", o, omap.objects[j], viewer)),
                                 default=None)
                    if best_a is None:
                        continue
                    sc *= best_a[0]
                    why += f", {s.relation or 'near'} {omap.objects[best_a[1]].label} #{omap.objects[best_a[1]].id}"
                cands.append(Candidate(o, sc, why))
            if not cands:
                names = ", ".join(f"{c.obj.label} #{c.obj.id} ({c.score:.2f})" for c in closest)
                what = s.target + (f" {s.relation or 'near'} the {s.anchor}" if s.anchor else "")
                return Resolution("not_found", candidates=closest,
                                  detail=f"nothing in the map matches '{what}' well enough; closest: {names}")
            if s.pick == "nearest" and here is not None:
                # "nearest X" = the nearest of the GOOD matches, not the nearest thing that clears the
                # floor: with 260 objects a mislabeled cabinet 2 m away scored 0.13 for "trash can" and
                # beat the real ones (0.29+) further on. Keep candidates within half of the best score.
                bar = max(threshold, 0.5 * max(c.score for c in cands))
                cands = sorted((c for c in cands if c.score >= bar), key=lambda c: _dist(c.obj.xy, here))
                note = f"nearest of {len(cands)} match{'es' if len(cands) != 1 else ''}"
            else:
                cands.sort(key=lambda c: -c.score)
                close = len(cands) > 1 and cands[0].score - cands[1].score < margin
                note = (f"close call: {cands[1].obj.label} #{cands[1].obj.id} scored {cands[1].score:.2f}"
                        if close else "")
            shown = cands[:5]
            # "nearest" already kept only good matches (>= half the best), so any of them may stand in; for "best",
            # a stand-in must be nearly as good as the top match, else say we can't reach it rather than downgrade
            floor = 0.0 if s.pick == "nearest" else FALLBACK_MIN_RATIO * cands[0].score
            goal, why_not, chosen, skipped = None, "", None, []
            for c in cands[:5]:
                if c is not cands[0] and c.score < floor:
                    skipped.append(c)
                    continue
                goal, why_not = approach_pose(c.obj.xy, c.obj.radius, grid, from_xy=here)
                if goal is not None and (reach_fn is None or reach_fn(goal)):
                    chosen = c
                    break
                goal = None
            if goal is None:
                weaker = (f"; weaker matches not used: " + ", ".join(f"{c.obj.label} #{c.obj.id} ({c.score:.2f})"
                                                                     for c in skipped[:3])) if skipped else ""
                return Resolution("not_found", candidates=shown,
                                  detail=f"found {cands[0].obj.label} #{cands[0].obj.id} but can't reach it "
                                         f"({why_not or 'no path'}){weaker}")
            if chosen is not cands[0]:
                # replaces, not appends: the close-call note was about the runner-up, which may be the one chosen
                # now ("close call: #223 scored 0.49" printed for #223 itself, live test 2026-10-14)
                note = f"{cands[0].obj.label} #{cands[0].obj.id} ({cands[0].score:.2f}) unreachable"
            steps.append(GoalStep(f"{chosen.obj.label} #{chosen.obj.id}", goal, chosen.obj, note))
        here = steps[-1].goal[:2]
    parts = [f"{st.label} ({next((c.score for c in shown if c.obj is st.obj), 0):.2f})" if st.obj else st.label
             for st in steps]
    notes = [st.note for st in steps if st.note]
    return Resolution("plan", " → ".join(parts), steps=steps, candidates=shown, detail="; ".join(notes))
