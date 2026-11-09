"""tools/missions/route.py: waypoint parsing, auto headings, refusals (pure)."""
import math
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools", "missions"))
from route import Waypoint, fill_auto_yaw, load_route, parse_wp_arg, quaternion_z  # noqa: E402


def test_parse_wp_arg_two_or_three_fields():
    w = parse_wp_arg("1.5,-2", 3)
    assert (w.x, w.y, w.yaw, w.label) == (1.5, -2.0, None, "wp_3")
    assert parse_wp_arg("0,0,1.57").yaw == 1.57
    for bad in ("1", "1,2,3,4", "a,b", "1,,2", ""):
        with pytest.raises(ValueError):
            parse_wp_arg(bad)


def test_load_route_accepts_lists_and_mappings(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("frame: map\nsite: circuit4\nwaypoints:\n  - {x: -2.13, y: -3.0, yaw: -1.57, label: a}\n"
                 "  - [0.5, -8.5]\n  - {x: -2.13, y: 0.35, yaw: 1.57, wait_s: 2}\n")
    wps = load_route(str(p))
    assert [w.label for w in wps] == ["a", "wp_1", "wp_2"]
    assert wps[1].yaw is None and wps[2].wait_s == 2.0


def test_load_route_refuses_bad_files(tmp_path):
    for text in ("waypoints: []\n", "frame: odom\nwaypoints: [[0, 0]]\n", "waypoints:\n  - [1]\n", "waypoints:\n  - {x: 1}\n", "- 1\n"):
        p = tmp_path / "bad.yaml"
        p.write_text(text)
        with pytest.raises(ValueError):
            load_route(str(p))


def test_auto_yaw_faces_the_next_waypoint_and_the_last_keeps_its_heading():
    wps = fill_auto_yaw([Waypoint(0, 0), Waypoint(2, 0), Waypoint(2, 3)])
    assert wps[0].yaw == pytest.approx(0.0)                   # toward (2, 0)
    assert wps[1].yaw == pytest.approx(math.pi / 2)           # toward (2, 3)
    assert wps[2].yaw == pytest.approx(math.pi / 2)           # arrives heading north, keeps it
    given = fill_auto_yaw([Waypoint(0, 0, yaw=1.0), Waypoint(1, 1)])
    assert given[0].yaw == 1.0                                # explicit yaw untouched
    assert fill_auto_yaw([Waypoint(0, 0)], start_yaw=-1.6)[0].yaw == -1.6
    assert fill_auto_yaw([Waypoint(0, 0)])[0].yaw == 0.0


def test_quaternion_sign():
    z, w = quaternion_z(math.pi / 2)
    assert z == pytest.approx(math.sqrt(0.5)) and w == pytest.approx(math.sqrt(0.5))
    assert quaternion_z(-math.pi / 2)[0] < 0
