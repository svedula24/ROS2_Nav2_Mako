"""Frontiers of the exploration costmap and VLFM's selection rule."""
import numpy as np

from m20_vlmnav import frontiers as fr
from m20_vlmnav.frontiers import Frontier, Selector


def corridor_map():
    """40 x 40 cells at 0.1 m: a known free corridor (cols 10..29, rows 0..19) in an unknown map, walls at its sides.
    Frontiers: the corridor's far end (row 19, 20 cells = 2 m) and a short 3-cell gap in the east wall (noise)."""
    cost = np.full((40, 40), 255, np.uint8)
    cost[0:20, 10:30] = 0                      # free corridor, open at row 19 toward unknown rows
    cost[0:20, 9] = 254; cost[0:20, 30] = 254  # walls
    cost[5:8, 30] = 0                          # a 3-cell gap in the east wall: a tiny frontier
    cost[0, 10:30] = 254                       # closed at the near end
    return cost


def test_frontier_cells_are_free_cells_touching_unknown():
    cost = corridor_map()
    m = fr.frontier_mask(cost)
    assert m[19, 15] and not m[10, 15]          # the open far end, not the middle of the corridor
    assert not m[19, 9]                         # a wall cell is never a frontier


def test_extract_keeps_long_segments_drops_noise_and_puts_the_waypoint_mid_segment():
    cost = corridor_map()
    fs = fr.extract(cost, 0.1, (0.0, 0.0), min_len_m=1.0)
    assert len(fs) == 1                         # the 0.3 m gap is dropped
    f = fs[0]
    assert abs(f.length_m - 2.0) < 0.15 and abs(f.xy[0] - 2.0) < 0.15 and abs(f.xy[1] - 1.95) < 0.1
    fs_all = fr.extract(cost, 0.1, (0.0, 0.0), min_len_m=0.1)
    assert len(fs_all) == 2


def test_waypoint_is_nudged_off_inflated_cells():
    cost = corridor_map()
    cost[15:20, 10:30] = 120                    # the far end sits inside the inflation (cost 120 >= 50)
    f = fr.extract(cost, 0.1, (0.0, 0.0))[0]
    assert cost[int(f.xy[1] / 0.1), int(f.xy[0] / 0.1)] < 50   # moved back onto a cell Nav2 accepts


def test_nudge_never_hops_a_wall_into_the_next_room():
    """The frontier's whole neighbourhood is inflated; the only cheap cells within 1 m are across a wall."""
    cost = np.full((40, 40), 255, np.uint8)
    cost[0:20, 10:30] = 120                     # the corridor, all inside the inflation
    cost[0:20, 30] = 254                        # wall
    cost[0:21, 31:37] = 0                       # the next room, cheap floor 2-6 cells past the wall...
    cost[0:21, 36] = 254; cost[20, 31:36] = 254; cost[0, 31:36] = 254   # ...fully walled: no frontier of its own
    cost[0, 10:30] = 254
    assert fr.extract(cost, 0.1, (0.0, 0.0)) == []     # the far-end frontier has no reachable goal cell: dropped
    cost[10:20, 12:14] = 0                      # cheap cells INSIDE the corridor, connected: now it has one
    f = fr.extract(cost, 0.1, (0.0, 0.0))[0]
    assert 1.2 <= f.xy[0] <= 1.4 and f.xy[1] < 2.0


def test_short_frontiers_a_door_wide_survive_the_default_filter():
    cost = corridor_map()
    cost[5:12, 30] = 0                          # a 0.7 m opening in the east wall: a door after inflation
    assert len(fr.extract(cost, 0.1, (0.0, 0.0))) == 2


def mk(xy, value):
    return Frontier(xy=xy, length_m=2.0, n_cells=20, cells=np.zeros((1, 2)), value=value)


def test_selection_takes_the_highest_value_then_sticks_while_it_holds_up():
    s = Selector()
    fs = [mk((5.0, 0.0), 0.21), mk((0.0, 5.0), 0.62)]
    assert s.choose(fs, (0.0, 0.0)) == 1                        # highest value
    fs2 = [mk((5.0, 0.0), 0.70), mk((0.1, 5.0), 0.615)]         # a new leader, but the last one dropped only 0.005
    assert s.choose(fs2, (0.0, 1.0)) == 1                       # hysteresis: stick
    fs3 = [mk((5.0, 0.0), 0.70), mk((0.1, 5.0), 0.50)]          # dropped by 0.12 > 0.01: let go
    assert s.choose(fs3, (0.0, 1.0)) == 0


def test_selection_skips_a_frontier_tried_twice_from_the_same_spot_then_falls_back_to_the_farthest():
    s = Selector()
    fs = [mk((5.0, 0.0), 0.9), mk((0.0, 8.0), 0.3)]
    assert s.choose(fs, (0.0, 0.0)) == 0
    s.last_xy = None                                            # (a fresh choice each time: no hysteresis)
    assert s.choose(fs, (0.0, 0.0)) == 0                        # second try of the same pair is allowed
    s.last_xy = None
    assert s.choose(fs, (0.0, 0.0)) == 1                        # third: skipped, the next best
    s.last_xy = None
    assert s.choose(fs, (0.0, 0.0)) == 1
    s.last_xy = None
    assert s.choose(fs, (0.0, 0.0)) == 1                        # both exhausted: the farthest (8 m > 5 m)


def test_no_frontiers_means_none():
    assert Selector().choose([], (0.0, 0.0)) is None


def test_a_failed_frontier_is_never_chosen_again_even_while_it_still_wins_on_value():
    s = Selector()
    fs = [mk((5.0, 0.0), 0.9), mk((0.0, 8.0), 0.2)]
    assert s.choose(fs, (0.0, 0.0)) == 0
    s.fail(fs[0].xy)                            # Nav2 could not reach it (glass, a gap)
    for _ in range(5):                          # every later tick: the other one, however good the first looks
        assert s.choose(fs, (0.0, 0.0)) == 1
    s.fail(fs[1].xy)                            # both failed: VLFM's fallback, the farthest (8 m > 5 m), never None
    assert s.choose(fs, (0.0, 0.0)) == 1


def test_same_class_accepts_the_detectors_synonyms_both_ways():
    from m20_vlmnav.targets import same_class
    assert same_class("refrigerator", "refrigerator")
    assert same_class("refrigerator", "fridge") and same_class("fridge", "refrigerator")
    assert same_class("trash can", "bin") and not same_class("trash can", "cone")


def test_tracking_a_drifting_waypoint_keeps_the_hysteresis_against_the_last_ticks_value():
    s = Selector()
    fs = [mk((5.0, 0.0), 0.30), mk((0.0, 8.0), 0.29)]
    assert s.choose(fs, (0.0, 0.0)) == 0
    moved = [mk((5.6, 0.3), 0.295), mk((0.0, 8.0), 0.302)]   # the goal's waypoint shifted 0.67 m; the other crept up
    s.track(moved[0].xy)                                      # the node found it again within 1 m
    assert s.choose(moved, (0.0, 0.0)) == 0                   # 0.295 + 0.01 > 0.30: still the one, no thrash
    # VLFM compares with the LAST TICK's value, not the value at choice: a slow decline (0.005 a tick, 0.295 -> 0.265
    # over six ticks, 0.035 in all) never trips the 0.01 margin, so the frontier is kept
    v = 0.295
    for _ in range(6):
        v -= 0.005
        slow = [mk((5.6, 0.3), v), mk((0.0, 8.0), 0.302)]
        s.track(slow[0].xy)
        assert s.choose(slow, (0.0, 0.0)) == 0
    dropped = [mk((5.6, 0.3), 0.25), mk((0.0, 8.0), 0.302)]   # 0.265 -> 0.25: more than 0.01 in one tick
    s.track(dropped[0].xy)
    assert s.choose(dropped, (0.0, 0.0)) == 1                 # VLFM lets go
