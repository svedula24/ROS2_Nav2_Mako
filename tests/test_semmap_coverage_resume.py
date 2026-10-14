"""tools/semmap/coverage_tour.py resume helpers: skip stops an earlier run saw, order the rest from the robot."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools", "semmap"))
from coverage_tour import nearest_first, seen_positions, unseen  # noqa: E402


def test_seen_positions_reads_camera_xy(tmp_path):
    T = np.eye(4); T[:3, 3] = (1.5, -2.0, 0.4)
    (tmp_path / "index.jsonl").write_text(json.dumps({"id": "kf_00001", "stamp": 0, "T_map_cam": T.tolist()}) + "\n")
    assert seen_positions([str(tmp_path), str(tmp_path / "missing")]).tolist() == [[1.5, -2.0]]


def test_unseen_drops_covered_stops_and_keeps_order():
    stops = [(0, 0), (5, 0), (10, 0), (0, 10)]
    seen = np.array([[0.5, 0.0], [9.0, 0.5]])
    assert unseen(stops, seen, 2.0) == [(5, 0), (0, 10)]
    assert unseen(stops, np.zeros((0, 2)), 2.0) == stops


def test_nearest_first_starts_at_the_robot():
    assert nearest_first([(10, 0), (1, 0), (5, 0)], (0, 0)) == [(1, 0), (5, 0), (10, 0)]
