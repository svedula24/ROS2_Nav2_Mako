import math

from m20_nlnav.approach import RELAXED_COST, approach_pose, footprint_cost
from nlnav_fixtures import block, open_grid


def test_open_floor_takes_the_closest_ring_toward_the_robot():
    g = open_grid()
    pose, note = approach_pose((3.0, 0.0), 0.3, g, from_xy=(0.0, 0.0))
    x, y, yaw = pose
    assert note == ""
    assert abs(y) < 1e-6 and x < 3.0                         # straight back toward the robot
    assert abs(math.hypot(x - 3.0, y) - (0.3 + 0.9 + 0.43)) < 1e-6
    assert abs(yaw) < 1e-6                                   # facing +x, at the object


def test_wall_between_pushes_the_pose_around_it():
    g = open_grid()
    block(g, 0.5, -3.0, 1.8, 3.0)                            # wall on the robot's side of the object
    pose, _ = approach_pose((3.0, 0.0), 0.3, g, from_xy=(0.0, 0.0))
    assert pose is not None
    assert footprint_cost(g, *pose) < 50
    assert pose[0] > 1.8 or abs(pose[1]) > 3.0              # not inside / behind the wall


def test_relaxes_then_gives_up():
    g = open_grid()
    g.data[:] = 70                                           # everything inside inflation
    pose, note = approach_pose((0.0, 0.0), 0.3, g, from_xy=(-3.0, 0.0))
    assert pose is not None and note == "close to obstacles"
    g.data[:] = RELAXED_COST + 5
    pose, note = approach_pose((0.0, 0.0), 0.3, g)
    assert pose is None and "no free floor" in note


def test_unknown_space_is_not_free():
    g = open_grid()
    g.data[:] = -1
    pose, _ = approach_pose((0.0, 0.0), 0.3, g)
    assert pose is None


def test_nav2_raw_costs_convert_like_nav2s_topic():
    from m20_nlnav.world import grid_from_costmap
    raw = bytes([0, 1, 128, 252, 253, 254, 255, 0])
    g = grid_from_costmap(4, 2, 0.1, (0.0, 0.0), raw)
    assert g.data.tolist() == [[0, 1, 50, 98], [99, 100, -1, 0]]   # 1 + 97*(c-1)//251
    assert g.cost(0.05, 0.05) == 0 and g.cost(0.15, 0.15) == 100


def test_no_costmap_returns_unchecked_pose():
    pose, note = approach_pose((3.0, 0.0), 0.3, None, from_xy=(0.0, 0.0))
    assert pose is not None and "unchecked" in note
