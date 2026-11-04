"""VLFM value map, ported: cone geometry, occlusion by depth, the confidence falloff, both fusion rules, reading."""
import math

import numpy as np

from m20_vlmnav.value_map import DECISION_THRESHOLD, MIN_CONFIDENCE, ValueMap, depth_row_from_image, remap


def vmap(fusion="max_confidence"):
    # 20 x 20 m at 0.1 m, origin (-10, -10); camera 90 deg, 5 m
    return ValueMap(200, 200, 0.1, (-10.0, -10.0), math.radians(90), fusion=fusion)


def open_row(n=64):
    return np.full(n, 5.0)


def test_cone_covers_the_fov_ahead_and_nothing_behind():
    m = vmap()
    m.update(0.5, 0.0, 0.0, 0.0, open_row())                  # at the origin, facing +x
    assert m.conf[m.cell(3.0, 0.0)] > 0.99                      # straight ahead: full confidence
    assert m.conf[m.cell(3.0, 2.9)] > 0                         # 44 deg left: inside the 90 deg cone
    assert m.conf[m.cell(3.0, 3.3)] == 0                        # 48 deg: outside
    assert m.conf[m.cell(-1.0, 0.0)] == 0                       # behind
    assert m.conf[m.cell(5.5, 0.0)] == 0                        # beyond max_depth


def test_confidence_falls_as_cos_squared_to_a_floor_of_025():
    m = vmap()
    m.update(0.5, 0.0, 0.0, 0.0, open_row())
    # at 30 deg off-axis: angle = remap(30, 0, 45, 0, 90) = 60 deg; cos^2 = 0.25; remapped to [0.25, 1] -> 0.4375
    c = m.conf[m.cell(3.0, 3.0 * math.tan(math.radians(30)))]
    assert abs(c - remap(math.cos(math.radians(60)) ** 2, 0, 1, MIN_CONFIDENCE, 1)) < 0.02
    edge = m.conf[m.cell(3.0, 3.0 * math.tan(math.radians(44.5)))]
    assert MIN_CONFIDENCE <= edge < 0.3                         # the edge of view is worth 0.25, not 0


def test_occlusion_stops_the_cone_at_the_measured_depth():
    m = vmap()
    row = open_row(64)
    row[24:40] = 2.0                                            # the middle columns see a wall 2 m ahead
    m.update(0.5, 0.0, 0.0, 0.0, row)
    assert m.conf[m.cell(1.5, 0.0)] > 0                         # before the wall
    assert m.conf[m.cell(2.5, 0.0)] == 0                        # behind it: nothing painted
    assert m.conf[m.cell(3.0, 2.5)] > 0                         # a side column still open to 5 m


def test_occlusion_uses_forward_depth_not_ray_length():
    """A depth image stores z (forward distance). A wall 2 m ahead across the whole view is 2.83 m along the
    45 deg ray; cells at forward distance 1.9 m must be painted all the way to the cone's edge."""
    m = vmap()
    m.update(0.5, 0.0, 0.0, 0.0, np.full(64, 2.0))
    assert m.conf[m.cell(1.9, 1.8)] > 0                         # forward 1.9 m, 43 deg off-axis: ray 2.6 m
    assert m.conf[m.cell(2.1, 0.0)] == 0                        # forward 2.1 m: behind the wall


def test_columns_map_to_bearings_through_the_pinhole():
    """With only the LEFTMOST eighth of the image open to 5 m, the open bearing is tan-mapped: u in [0.75, 1] ->
    theta in [36.9, 45] deg, so a cell at 30 deg is occluded (linear mapping would call 33.75+ deg open)."""
    m = vmap()
    row = np.full(64, 1.0); row[:8] = 5.0
    m.update(0.5, 0.0, 0.0, 0.0, row)
    assert m.conf[m.cell(3.0 * math.cos(math.radians(40)), 3.0 * math.sin(math.radians(40)))] > 0    # 40 deg: open
    assert m.conf[m.cell(3.0 * math.cos(math.radians(30)), 3.0 * math.sin(math.radians(30)))] == 0   # 30 deg: wall


def test_depth_row_takes_the_column_max_and_treats_holes_as_open():
    depth = np.zeros((4, 3), np.float32)
    depth[:, 0] = [0.0, 1.2, 3.4, 2.0]
    depth[:, 1] = 0.0                                           # no return at all
    depth[:, 2] = [7.0, 6.5, 0.0, 0.0]
    assert np.allclose(depth_row_from_image(depth, 0.5, 5.0), [3.4, 5.0, 5.0])


def test_yaw_rotates_the_cone():
    m = vmap()
    m.update(0.5, 0.0, 0.0, math.pi / 2, open_row())            # facing +y
    assert m.conf[m.cell(0.0, 3.0)] > 0.99 and m.conf[m.cell(3.0, 0.0)] == 0


def test_max_confidence_fusion_replaces_only_where_the_new_look_is_better():
    m = vmap()
    m.update(0.21, 0.0, 0.0, 0.0, open_row())                   # corridor view: low score, facing +x
    m.update(0.62, 0.0, 0.0, math.pi / 2, open_row())           # shelving view: high score, facing +y
    assert m.value[m.cell(3.0, 0.0)] == np.float32(0.21)        # only the first look covered this
    assert m.value[m.cell(0.0, 3.0)] == np.float32(0.62)
    # a cell at 40 deg from +x (conf ~0.28) and 50 deg from +y (outside the second cone): keeps 0.21
    assert m.value[m.cell(3.0, 2.5)] == np.float32(0.21)
    # the second look at a cell the first saw only at its edge replaces it (new confidence higher)
    m2 = vmap()
    m2.update(0.21, 0.0, 0.0, math.radians(40), open_row())      # cell (3, 0) is 40 deg off-axis: conf low
    low = m2.conf[m2.cell(3.0, 0.0)]
    m2.update(0.62, 0.0, 0.0, 0.0, open_row())                  # now straight ahead: conf 1
    assert 0 < low < DECISION_THRESHOLD + 0.1 and m2.value[m2.cell(3.0, 0.0)] == np.float32(0.62)


def test_weak_edge_looks_over_a_stored_better_one_are_dropped_in_both_modes():
    """VLFM's decision threshold: a new confidence under 0.35 that is also below the stored one is zeroed BEFORE
    fusion. In max_confidence mode replacement already needs a higher confidence; in weighted mode the rule is what
    stops a glance from the edge of view dragging a good value down."""
    for fusion in ("max_confidence", "weighted"):
        m = vmap(fusion)
        m.update(0.62, 0.0, 0.0, 0.0, open_row())               # straight ahead: conf 1 at (3, 0)
        m.update(0.10, 0.0, 0.0, math.radians(42), open_row())  # same cell now at the cone's edge: conf ~0.26 < 0.35
        assert abs(m.value[m.cell(3.0, 0.0)] - 0.62) < 1e-6 and m.conf[m.cell(3.0, 0.0)] > 0.99, fusion
    # ...but an edge look over NOTHING stored still paints (c_old = 0 is not "better")
    m = vmap("weighted")
    m.update(0.10, 0.0, 0.0, math.radians(42), open_row())
    assert 0 < m.conf[m.cell(3.0, 0.0)] < DECISION_THRESHOLD
    # ...and a mid-confidence look (>= 0.35) over a better one is averaged in weighted mode, as the paper says
    m = vmap("weighted")
    m.update(0.62, 0.0, 0.0, 0.0, open_row())
    m.update(0.10, 0.0, 0.0, math.radians(20), open_row())      # 20 deg off: conf ~0.69
    assert 0.3 < m.value[m.cell(3.0, 0.0)] < 0.62


def test_weighted_fusion_matches_the_papers_equations():
    m = vmap(fusion="weighted")
    m.update(0.21, 0.0, 0.0, 0.0, open_row())
    r, c = m.cell(3.0, 0.0)
    c1 = m.conf[r, c]                                           # 1.0 on axis
    m.update(0.62, 0.0, 0.0, 0.0, open_row())                   # same look again, conf 1.0: plain average
    assert abs(m.value[r, c] - 0.415) < 1e-6 and abs(m.conf[r, c] - c1) < 1e-6   # same confidence twice: unchanged
    # the artifact's worked example: c_old 0.1 / v 0.21, then c_new 1.0 / v 0.62 -> 0.58
    m2 = vmap(fusion="weighted")
    m2.conf[r, c], m2.value[r, c] = 0.1, 0.21
    m2.update(0.62, 0.0, 0.0, 0.0, open_row())
    assert abs(m2.value[r, c] - (0.1 * 0.21 + 1.0 * 0.62) / 1.1) < 2e-3   # on-axis conf is 0.9992, not 1.0


def test_read_is_the_median_of_observed_cells_in_the_radius_or_minus_one():
    m = vmap()
    assert m.read(3.0, 0.0) == -1.0
    m.update(0.4, 0.0, 0.0, 0.0, open_row())
    assert abs(m.read(3.0, 0.0) - 0.4) < 1e-6
    assert m.read(-5.0, -5.0) == -1.0                           # never looked there
    assert m.read(3.0, 0.0, reduction="max") == np.float32(0.4)
