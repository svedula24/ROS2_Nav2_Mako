"""Approach pose: where the robot should stop to "go to" an object.

Never the object's own position (that's inside it). Sample a ring of poses around the object's
footprint, keep those where the WHOLE robot footprint sits on low-cost floor in the global
costmap, and pick the closest ring, preferring the side the robot is coming from. The robot
faces the object.
"""
import math
from typing import Optional, Tuple

from .world import Grid

FOOTPRINT = (0.86, 0.56)          # M20 length x width [m] (same as the demo's Nav2 footprint)
RADII = (0.9, 1.2, 1.5)           # gap between object edge and robot centre [m], tried in order
N_HEADINGS = 24
MAX_COST = 50                     # inside the keep-away inflation, so it matches how the lap drives
RELAXED_COST = 90                 # second pass, still below inscribed (99) / lethal (100)


def footprint_cost(grid: Grid, x: float, y: float, yaw: float, pad: float = 0.05) -> int:
    """Max cost under the (padded) robot rectangle at (x, y, yaw); unknown (-1) counts as 100."""
    hl, hw = FOOTPRINT[0] / 2 + pad, FOOTPRINT[1] / 2 + pad
    step = max(grid.res, 0.05)
    c, s = math.cos(yaw), math.sin(yaw)
    worst = 0
    nu, nv = int(2 * hl / step) + 1, int(2 * hw / step) + 1
    for i in range(nu + 1):
        u = -hl + 2 * hl * i / nu
        for j in range(nv + 1):
            v = -hw + 2 * hw * j / nv
            k = grid.cost(x + c * u - s * v, y + s * u + c * v)
            k = 100 if k < 0 else k
            if k > worst:
                worst = k
                if worst >= 100:
                    return worst
    return worst


def approach_pose(obj_xy: Tuple[float, float], obj_radius: float, grid: Optional[Grid],
                  from_xy: Optional[Tuple[float, float]] = None,
                  max_cost: int = MAX_COST) -> Tuple[Optional[Tuple[float, float, float]], str]:
    """((x, y, yaw) | None, note). With no costmap the nearest ring pose toward the robot is
    returned unchecked (Nav2 still refuses goals inside obstacles)."""
    ox, oy = obj_xy
    toward = math.atan2(from_xy[1] - oy, from_xy[0] - ox) if from_xy else 0.0
    if grid is None:
        d = obj_radius + RADII[0] + FOOTPRINT[0] / 2
        a = toward
        return (ox + d * math.cos(a), oy + d * math.sin(a), a + math.pi), "costmap unavailable (unchecked)"
    for limit, note in ((max_cost, ""), (RELAXED_COST, "close to obstacles")):
        best = None
        for r in RADII:
            d = obj_radius + r + FOOTPRINT[0] / 2
            for k in range(N_HEADINGS):
                a = toward + 2 * math.pi * k / N_HEADINGS
                x, y, yaw = ox + d * math.cos(a), oy + d * math.sin(a), a + math.pi
                if footprint_cost(grid, x, y, yaw) >= limit:
                    continue
                # prefer the closest ring, then the side facing the robot's approach
                score = r + 0.5 * (1 - math.cos(a - toward))
                if best is None or score < best[0]:
                    best = (score, (x, y, math.atan2(math.sin(yaw), math.cos(yaw))))
            if best is not None:          # a valid pose on this ring beats any farther ring
                return best[1], note
    return None, "no free floor within 1.5 m of it"
