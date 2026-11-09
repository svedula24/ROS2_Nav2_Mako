"""tools/nav/map_tf_adapter.py: the map->odom math (pure; tf2_ros is stubbed out for the import)."""
import math
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "nav"))
sys.modules.setdefault("tf2_ros", types.SimpleNamespace(Buffer=object, TransformBroadcaster=object, TransformListener=object))
sys.modules.setdefault("rclpy.time", types.SimpleNamespace(Time=object))
sys.modules.setdefault("rclpy.duration", types.SimpleNamespace(Duration=object))
import map_tf_adapter as a   # noqa: E402


def test_map_odom_reproduces_the_report_and_moves_with_odometry():
    odom_base = (2.0, 0.5, 0.3)
    map_base = (10.0, -4.0, 1.8)
    map_odom = a.compose(map_base, a.invert(odom_base))
    assert a.compose(map_odom, odom_base) == pytest.approx(map_base)
    # 1 m forward in odom = 1 m along the robot's map heading, map->odom unchanged (no jump between reports)
    moved = a.compose(odom_base, (1.0, 0.0, 0.0))
    mx, my, myaw = a.compose(map_odom, moved)
    assert math.hypot(mx - map_base[0], my - map_base[1]) == pytest.approx(1.0)
    assert myaw == pytest.approx(map_base[2])


def test_invert_is_an_inverse_and_yaw_wraps():
    p = (1.0, 2.0, 3.0)
    assert a.compose(p, a.invert(p)) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert a.compose((0, 0, 3.0), (0, 0, 1.0))[2] == pytest.approx(4.0 - 2 * math.pi)
