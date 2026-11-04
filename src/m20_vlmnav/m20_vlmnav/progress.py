"""The search's two timing rules, pure (no ROS), so they can be tested like episode.terminal_action:

    stuck            the robot has made no progress: over the last `window_s` it never got farther than `radius_m`
                     from where it is now, whatever goals were sent meanwhile. Nav2 keeps a boxed-in robot "running"
                     through recoveries (back-ups and spins of 0.3-0.5 m) for as long as the mission lasts, so a
                     displacement test fails to see it, and a window that restarted on every goal would never fill
                     while better-valued frontiers keep being accepted. The node restarts the window only on a
                     REACHED goal: short goals reached one after the other inside radius_m are progress, not a box.
    switch allowed   a goal just sent is kept at least `commit_s` before a better-valued frontier may replace it (the
                     scorer is noisier than VLFM's setting; values near the goal repaint while Nav2 spins); a RETARGET
                     (the same frontier's waypoint moved) is never gated.
"""
import math
from typing import List, Tuple

Trail = List[Tuple[float, Tuple[float, float]]]      # (time, xy) samples, oldest first
WINDOW_FILL = 0.95                                   # the trail must span this much of window_s before it can judge


def prune(trail: Trail, now: float, window_s: float) -> Trail:
    """The samples still inside the window."""
    return [tp for tp in trail if now - tp[0] <= window_s]


def is_stuck(trail: Trail, now: float, xy: Tuple[float, float], window_s: float, radius_m: float) -> bool:
    """True when the (already pruned) trail spans the window and every sample lies within radius_m of xy."""
    if not trail or now - trail[0][0] < window_s * WINDOW_FILL:
        return False
    return max(math.dist(xy, p) for _t, p in trail) < radius_m


def switch_allowed(now: float, goal_t: float, commit_s: float) -> bool:
    """May the running goal be replaced by a better frontier now? Once it has had commit_s. (A retarget, the same
    frontier's waypoint moving, is sent by the node without asking.)"""
    return now - goal_t >= commit_s
