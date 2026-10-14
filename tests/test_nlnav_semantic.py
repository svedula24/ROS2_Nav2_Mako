"""semantic.resolve: structured query -> goals on an object map (label embeddings, no torch)."""
import math

import numpy as np

from m20_nlnav.llm import Query, QueryStep
from m20_nlnav.object_map import MapObject, ObjectMap
from m20_nlnav.semantic import LabelEncoder, label_embedding, relation_ok, resolve
from m20_nlnav.world import Waypoint
from nlnav_fixtures import open_grid

ENC = LabelEncoder()


def mo(i, label, xy, z=0.5, size=(0.5, 0.5, 1.0), shift=-0.5, scale=10.0):
    return MapObject(i, label, (xy[0], xy[1], z), size, label_embedding(label), shift, scale, 0.8, 4, 0.0, 0.8)


def office_map():
    return ObjectMap([
        mo(1, "ladder", (5.0, 0.0)),
        mo(2, "trash can", (2.0, 3.0)),
        mo(3, "trash can", (-3.0, 0.5)),
        mo(4, "whiteboard", (2.0, 5.0)),
        mo(5, "desk", (-4.0, -4.0)),
    ])


WPS = {"start": Waypoint("start", -2.0, 0.5, 0.0), "dock": Waypoint("dock", 6.0, -6.0, 1.57)}


def q(*steps, stop=False):
    return Query([QueryStep(*s) if isinstance(s, tuple) else QueryStep(s) for s in steps], stop)


def run(query, pose=(0.0, 0.0, 0.0), **kw):
    return resolve(query, office_map(), ENC, WPS, pose=pose, grid=open_grid(), **kw)


def test_target_scores_with_owl_head_and_goal_faces_object():
    r = run(q("ladder"))
    assert r.status == "plan" and r.steps[0].obj.id == 1
    x, y, yaw = r.steps[0].goal
    assert 0.9 < math.hypot(x - 5.0, y) < 3.0
    to_obj = math.atan2(-y, 5.0 - x)
    assert abs(math.atan2(math.sin(yaw - to_obj), math.cos(yaw - to_obj))) < 0.2


def test_synonym_phrase_and_nearest():
    r = run(q(("bin", None, None, "nearest")))
    assert r.status == "plan" and r.steps[0].obj.label == "trash can"
    d = {o.id: math.hypot(*o.xy) for o in office_map().objects if o.label == "trash can"}
    assert r.steps[0].obj.id == min(d, key=d.get)


def test_relation_near_picks_the_one_by_the_anchor():
    r = run(q(("trash can", "whiteboard", "near", "best")))
    assert r.status == "plan" and r.steps[0].obj.id == 2


def test_not_found_lists_closest_and_does_not_move():
    r = run(q("forklift"))
    assert r.status == "not_found" and not r.steps
    assert len(r.candidates) == 3 and "closest" in r.detail


def test_waypoints_first_then_back_and_start():
    r = run(q("dock"))
    assert r.steps[0].label == "dock" and r.steps[0].goal == (6.0, -6.0, 1.57)
    r = run(q("ladder", "back"), pose=(1.0, 1.0, 0.3))
    assert r.steps[-1].label == "back" and r.steps[-1].goal == (1.0, 1.0, 0.3)
    r = run(q("home"))
    assert r.steps[0].goal == (-2.0, 0.5, 0.0)


def test_multi_step_chains_from_previous_goal():
    r = run(q("ladder", ("trash can", None, None, "nearest")))
    p_goal = r.steps[0].goal[:2]
    cans = [o for o in office_map().objects if o.label == "trash can"]
    want = min(cans, key=lambda o: math.hypot(o.xy[0] - p_goal[0], o.xy[1] - p_goal[1]))
    assert r.steps[1].obj.id == want.id


def test_pick_id_forces_the_object():
    r = run(q("trash can"), pick_id=3)
    assert r.steps[0].obj.id == 3


def test_close_scores_go_to_best_and_say_so():
    om = ObjectMap([mo(1, "desk", (3, 0)), mo(2, "desk", (-3, 0))])
    r = resolve(q("desk"), om, ENC, WPS, pose=(0, 0, 0), grid=open_grid())
    assert r.status == "plan" and "close call" in r.detail


def test_unreachable_falls_back_to_next_candidate():
    r = run(q("trash can"), reach_fn=lambda goal: goal[1] < 1.5)      # only goals south of y=1.5 reachable
    assert r.status == "plan" and r.steps[0].obj.id == 3 and "unreachable" in r.detail
    assert "close call" not in r.detail          # the tie note named #3, the object now chosen: misleading after a fallback


def test_relation_geometry_left_right_front_behind():
    anchor = mo(9, "desk", (4.0, 0.0))
    viewer = (0.0, 0.0)                                          # looking along +x at the anchor
    assert relation_ok("left_of", mo(8, "x", (4.0, 1.0)), anchor, viewer)
    assert relation_ok("right_of", mo(8, "x", (4.0, -1.0)), anchor, viewer)
    assert relation_ok("in_front_of", mo(8, "x", (3.0, 0.0)), anchor, viewer)
    assert relation_ok("behind", mo(8, "x", (5.2, 0.0)), anchor, viewer)
    assert not relation_ok("left_of", mo(8, "x", (4.0, -1.0)), anchor, viewer)


def test_stop():
    assert run(Query([], True)).status == "stop"


def test_nearest_means_nearest_good_match_not_nearest_weak_one():
    """A mislabeled object 1 m away that barely clears the threshold must not beat a real match further on."""
    e_tc, e_desk = label_embedding("trash can"), label_embedding("desk")
    weak = 0.3 * e_tc + math.sqrt(1 - 0.3 ** 2) * e_desk          # cos("trash can") = 0.3 -> prob ~0.12
    cabinet = MapObject(9, "refrigerator", (1.0, 0.0, 0.5), (0.5, 0.5, 1.0), weak / np.linalg.norm(weak),
                        -0.5, 10.0, 0.8, 4, 0.0, 0.8)
    omap = ObjectMap(office_map().objects + [cabinet])
    r = resolve(q(("trash can", None, None, "nearest")), omap, ENC, WPS, pose=(0.0, 0.0, 0.0), grid=open_grid())
    assert r.status == "plan" and r.steps[0].obj.label == "trash can" and r.steps[0].obj.id == 3
    assert "nearest of 2" in r.detail


def test_unreachable_best_does_not_downgrade_to_a_weak_match():
    """Top match unreachable + only a much weaker alternative -> not_found naming both, never the weak one.
    (2026-10-14: stale costmap marks hid ladders 0.88/0.86 and the robot chose a 0.55 ladder.)"""
    e_tc, e_desk = label_embedding("trash can"), label_embedding("desk")
    weak = 0.3 * e_tc + math.sqrt(1 - 0.3 ** 2) * e_desk
    other = MapObject(9, "refrigerator", (-3.0, -3.0, 0.5), (0.5, 0.5, 1.0), weak / np.linalg.norm(weak),
                      -0.5, 10.0, 0.8, 4, 0.0, 0.8)
    omap = ObjectMap([mo(2, "trash can", (2.0, 3.0)), other])
    r = resolve(q("trash can"), omap, ENC, WPS, pose=(0.0, 0.0, 0.0), grid=open_grid(),
                reach_fn=lambda goal: goal[1] < 1.5)          # #2 (north) unreachable, #9 (south) reachable
    assert r.status == "not_found" and "can't reach" in r.detail and "#9" in r.detail
