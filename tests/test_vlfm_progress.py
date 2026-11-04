"""The search's stuck test and goal commit rule (m20_vlmnav.progress), pure."""
from m20_vlmnav.progress import is_stuck, prune, switch_allowed


def walk(points, t0=0.0, dt=1.0):
    return [(t0 + i * dt, p) for i, p in enumerate(points)]


def test_stuck_needs_the_trail_to_span_the_window():
    trail = walk([(0.0, 0.0)] * 20)                            # 19 s of standing still, window 30 s
    assert not is_stuck(trail, 19.0, (0.0, 0.0), 30.0, 1.0)
    trail = walk([(0.0, 0.0)] * 29)                            # 28 s: under 0.95 x 30 = 28.5
    assert not is_stuck(trail, 28.0, (0.0, 0.0), 30.0, 1.0)
    trail = walk([(0.0, 0.0)] * 30)                            # 29 s: spans enough
    assert is_stuck(trail, 29.0, (0.0, 0.0), 30.0, 1.0)


def test_recovery_wiggles_count_as_stuck_but_real_motion_does_not():
    wiggle = [(0.0, 0.0), (0.4, 0.0), (0.0, 0.3), (-0.3, -0.2), (0.5, 0.1)] * 6      # back-ups and spins, 30 samples
    assert is_stuck(walk(wiggle), 29.0, (0.1, 0.0), 30.0, 1.0)
    drift = [(0.04 * i, 0.0) for i in range(31)]                                      # 1.2 m across the window
    assert not is_stuck(walk(drift), 30.0, (1.2, 0.0), 30.0, 1.0)


def test_prune_keeps_only_the_window():
    trail = walk([(0.0, 0.0)] * 50)
    assert len(prune(trail, 49.0, 30.0)) == 31


def test_switch_is_gated_by_the_commit_time():
    assert not switch_allowed(now=105.0, goal_t=100.0, commit_s=10.0)
    assert switch_allowed(now=110.0, goal_t=100.0, commit_s=10.0)
