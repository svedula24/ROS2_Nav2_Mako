"""tools/nlnav_eval: the case file parses and the judge handles each expectation kind (no ROS, no stack)."""
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools", "nlnav_eval"))
from judge import judge  # noqa: E402

CASES = yaml.safe_load(open(os.path.join(HERE, "..", "tools", "nlnav_eval", "cases.yaml")))


def test_cases_file_is_well_formed():
    names = [c["name"] for c in CASES["cases"]]
    assert len(names) == len(set(names)) and len(names) >= 11
    for c in CASES["cases"]:
        assert c["text"] and c["expect"]["status"] in ("plan", "not_found", "not_understood", "stopped", "error", "done", "canceled")
    assert sum(1 for c in CASES["cases"] if c.get("drive")) >= 3


def plan(*labels, ids=None, xy=(0.0, 0.0)):
    return {"status": "plan", "items": [{"label": f"{l} #{(ids or [1] * len(labels))[k]}", "id": (ids or [1] * len(labels))[k],
                                         "x": xy[0], "y": xy[1]} for k, l in enumerate(labels)]}


def test_judge_plan_level():
    e = {"status": "plan", "labels": ["ladder"], "ids": [26], "goal_xy": [-2.13, -8.52], "tol": 0.6}
    assert judge(e, plan("ladder", ids=[26], xy=(-2.0, -8.3)))[0]
    assert not judge(e, plan("ladder", ids=[42], xy=(-2.0, -8.3)))[0]              # wrong object
    assert not judge(e, plan("ladder", ids=[26], xy=(0.0, 0.0)))[0]                # goal too far
    assert judge({"status": "plan", "labels": ["ladder", "start"]}, plan("ladder", "start"))[0]
    assert not judge({"status": "plan", "labels": ["ladder", "start"]}, plan("ladder"))[0]   # dropped step
    assert judge({"status": "not_found"}, {"status": "not_found", "items": []})[0]
    assert not judge({"status": "not_understood"}, {"status": "plan", "items": []})[0]
    assert judge({"status": "plan", "ids": [20]}, plan("object", ids=[20]))[0]      # label free, id fixed


def test_judge_drive_level():
    e = {"status": "done", "labels": ["ladder"], "near_id": 26, "near_m": 2.5}
    objs = {26: (-2.12, -10.83)}
    ok = {"status": "done", "items": plan("ladder", ids=[26], xy=(-2.13, -8.52))["items"], "final_xy": [-2.2, -8.6],
          "max_tilt_rad": 0.1, "max_footprint_age_s": 0.3}
    assert judge(e, ok, objs)[0]
    assert not judge(e, dict(ok, final_xy=[0.0, 0.0]), objs)[0]                    # didn't arrive
    assert not judge(e, dict(ok, final_xy=[-3.0, -8.6]), objs)[0]                  # near the object, not at the pose
    assert not judge(e, dict(ok, max_tilt_rad=0.6), objs)[0]                       # tipped
    assert not judge(e, dict(ok, max_footprint_age_s=9.0), objs)[0]                # frozen pose
    assert not judge(e, dict(ok, failsafe="tip-over"), objs)[0]
    stop = {"status": "canceled", "stop_after_s": 4}
    assert judge(stop, {"status": "canceled", "items": [], "stop_drift_m": 0.27, "settle_m": 0.01})[0]
    assert not judge(stop, {"status": "canceled", "items": [], "stop_drift_m": 0.8, "settle_m": 0.01})[0]   # rolled too far
    assert not judge(stop, {"status": "canceled", "items": [], "stop_drift_m": 0.2, "settle_m": 0.3})[0]    # never stopped
