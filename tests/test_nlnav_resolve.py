import math

from m20_nlnav import tier0
from m20_nlnav.resolve import resolve
from m20_nlnav.world import DETECTOR_CLASSES
from nlnav_fixtures import block, office, open_grid


def run(text, world=None, pick=-1):
    w = world or office()
    cmd = tier0.parse(text, DETECTOR_CLASSES, w.waypoints)
    assert cmd is not None, text
    return resolve(cmd, w, pick_id=pick)


def test_single_object_goal_is_in_front_of_it_and_faces_it():
    r = run("go to the printer")
    assert r.status == "plan" and r.steps[0].obj.id == 1
    x, y, yaw = r.steps[0].goal
    d = math.hypot(x - 4.0, y - 0.0)
    assert 0.9 < d < 3.0                                  # never on top of the object
    assert x < 4.0                                        # on the side the robot comes from
    to_obj = math.atan2(0.0 - y, 4.0 - x)
    assert abs(math.atan2(math.sin(yaw - to_obj), math.cos(yaw - to_obj))) < 0.2


def test_several_matches_picks_highest_confidence():
    r = run("go to the chair")
    assert r.status == "plan" and r.steps[0].obj.id == 3      # 0.90 of 0.90 / 0.71 / 0.55
    assert len(r.candidates) == 3 and r.candidates[0].obj.id == 3


def test_explicit_pick_overrides_confidence():
    r = run("go to the chair", pick=4)
    assert r.status == "plan" and r.steps[0].obj.id == 4


def test_nearest_picks_by_distance():
    r = run("go to the nearest chair")
    assert r.status == "plan"
    w = office()
    dists = {o.id: math.hypot(*o.xy) for o in w.objects if o.cls == "office chair"}
    assert r.steps[0].obj.id == min(dists, key=dists.get)


def test_not_seen():
    r = run("go to the ladder")
    assert r.status == "not_found" and "haven't seen" in r.detail


def test_low_confidence_sighting_is_reported_not_used():
    from m20_nlnav.world import Obj
    w = office()
    w.min_score = 0.5
    w.low_conf = [Obj(9, "ladder", (1.0, 1.0, 0.5), 0.41, 3)]
    r = run("go to the ladder", world=w)
    assert r.status == "not_found" and "low confidence (0.41" in r.detail


def test_waypoints_start_and_back():
    r = run("go to stop 3")
    assert r.status == "plan" and r.steps[0].goal[:2] == (0.8, -8.1)
    w = office()
    w.pose = (2.0, 2.0, 1.0)
    r = run("go to the printer then come back", world=w)
    assert r.steps[-1].label == "back" and r.steps[-1].goal == (2.0, 2.0, 1.0)
    r = run("go home", world=w)
    assert r.steps[0].goal == (0.0, 0.0, 0.0)


def test_scene_start_waypoint_overrides_the_default_spawn():
    # the Godot sim's M20 spawns away from the map origin; its scene yaml names `start`
    from m20_nlnav.world import Waypoint
    w = office()
    w.waypoints["start"] = Waypoint("start", -2.13, 0.35, -1.58)
    r = run("go back to the start", world=w)
    assert r.steps[0].goal == (-2.13, 0.35, -1.58)
    w.waypoints.pop("start")
    assert run("go home", world=w).steps[0].goal == (0.0, 0.0, 0.0)


def test_multi_step_nearest_is_from_previous_goal():
    # "nearest" is measured from the printer goal, not from the robot's current pose
    w = office()
    r = run("go to the printer then the nearest chair", world=w)
    assert r.steps[0].obj.id == 1
    p_goal = r.steps[0].goal[:2]
    chairs = [o for o in w.objects if o.cls == "office chair"]
    want = min(chairs, key=lambda o: math.hypot(o.xy[0] - p_goal[0], o.xy[1] - p_goal[1]))
    assert r.steps[1].obj.id == want.id


def test_falls_back_to_next_candidate_when_unreachable():
    g = open_grid()
    block(g, -6.0, -2.0, 0.0, 4.0)            # bury chair #3 (best score) in an obstacle
    r = run("go to the chair", world=office(g))
    assert r.status == "plan" and r.steps[0].obj.id != 3 and "unreachable" in r.steps[0].note


def test_explicit_pick_unreachable_is_reported_not_swapped():
    g = open_grid()
    block(g, -6.0, -2.0, 0.0, 4.0)
    r = run("go to the chair", world=office(g), pick=3)
    assert r.status == "not_found" and "can't get close" in r.detail


def test_stop():
    assert run("stop").status == "stop"
