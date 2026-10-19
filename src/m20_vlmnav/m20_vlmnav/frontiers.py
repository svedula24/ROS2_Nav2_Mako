"""Frontiers of the exploration costmap, and VLFM's rule for choosing one.

The exploration map is Nav2's global costmap run non-rolling with unknown tracking (bringup hook M20_EXPLORE_MAP):
raw costs 0..252 known (0 free, rising inside the inflation), 253/254 inscribed/lethal, 255 UNKNOWN.
    frontier cell   a known cell with cost < 128 (free enough) that touches an unknown cell (4-neighbourhood)
    segment         8-connected group of frontier cells; shorter than min_len (0.5 m: a door after inflation is
                    0.8-0.9 m) it is noise (a single lidar beam through a gap leaves a thin "ray" of free cells edged
                    with frontier) and is dropped
    waypoint        the segment's middle cell by path order (VLFM: the contour's midpoint by arc length), nudged to
                    the nearest known cell with cost < max_cost that is CONNECTED to it through known cells (never
                    across a thin wall) so Nav2 accepts it as a goal
Selection (VLFM ITMPolicyV2 _get_best_frontier): highest value wins, with hysteresis: keep the last chosen frontier
while it still exists (a waypoint within 0.5 m) and its value hasn't dropped by more than 0.01; skip a frontier that
was already chosen twice from the same robot cell (the acyclic enforcer, simplified) or that Nav2 FAILED to reach
(Selector.fail: behind glass, across an unreachable gap); if every frontier is cyclic, VLFM's code takes the farthest.
"""
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

UNKNOWN = 255
FREE_MAX = 128          # cost below this = free enough to count as explored floor
GOAL_MAX_COST = 50      # a goal must sit on a cell below this (outside the keep-away inflation, as approach.py uses)
MIN_LEN_M = 0.5
STICK_RADIUS_M = 0.5
TRACK_RADIUS_M = 1.0    # a goal frontier whose waypoint moved less than this as the map filled is the SAME frontier
STICK_MARGIN = 0.01
MAX_TRIES_PER_SPOT = 2


@dataclass
class Frontier:
    xy: Tuple[float, float]          # waypoint, metres
    length_m: float
    n_cells: int
    cells: np.ndarray = field(repr=False)   # (n, 2) rows/cols
    value: float = -1.0


def frontier_mask(cost: np.ndarray) -> np.ndarray:
    unknown = cost == UNKNOWN
    free = (cost < FREE_MAX)
    nb = np.zeros_like(unknown)
    nb[1:, :] |= unknown[:-1, :]; nb[:-1, :] |= unknown[1:, :]
    nb[:, 1:] |= unknown[:, :-1]; nb[:, :-1] |= unknown[:, 1:]
    return free & nb


def _segments(mask: np.ndarray) -> List[np.ndarray]:
    """8-connected components of a boolean mask, as (n, 2) arrays of (row, col)."""
    seen = np.zeros_like(mask, bool)
    out = []
    rows, cols = np.nonzero(mask)
    for r, c in zip(rows, cols):
        if seen[r, c]:
            continue
        comp, q = [], deque([(r, c)])
        seen[r, c] = True
        while q:
            y, x = q.popleft()
            comp.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        q.append((yy, xx))
        out.append(np.array(comp))
    return out


def _path_order(cells: np.ndarray) -> np.ndarray:
    """Order a segment's cells along it: BFS from the cell farthest from the centroid (an end), then by BFS distance."""
    centroid = cells.mean(axis=0)
    start = int(np.argmax(((cells - centroid) ** 2).sum(axis=1)))
    index = {tuple(c): i for i, c in enumerate(cells)}
    dist = {start: 0}
    q = deque([start])
    while q:
        i = q.popleft()
        y, x = cells[i]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                j = index.get((y + dy, x + dx))
                if j is not None and j not in dist:
                    dist[j] = dist[i] + 1
                    q.append(j)
    order = sorted(dist, key=dist.get)
    return cells[order]


def _nudge(cost: np.ndarray, r: int, c: int, max_cost: int, radius: int) -> Optional[Tuple[int, int]]:
    """The nearest cell with cost < max_cost reachable from (r, c) through known free-enough cells (cost < FREE_MAX)
    within `radius` steps, or None. A breadth-first walk, so it never hops a wall into the next room."""
    if cost[r, c] < max_cost:
        return r, c
    seen = {(r, c)}
    q = deque([(r, c, 0)])
    while q:
        y, x, d = q.popleft()
        if d >= radius:
            continue
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = y + dy, x + dx
            if (yy, xx) in seen or not (0 <= yy < cost.shape[0] and 0 <= xx < cost.shape[1]):
                continue
            seen.add((yy, xx))
            if cost[yy, xx] >= FREE_MAX:             # lethal, inflated or unknown: not a path
                continue
            if cost[yy, xx] < max_cost:
                return yy, xx
            q.append((yy, xx, d + 1))
    return None


def extract(cost: np.ndarray, resolution: float, origin: Tuple[float, float],
            min_len_m: float = MIN_LEN_M, max_cost: int = GOAL_MAX_COST) -> List[Frontier]:
    """All frontiers of the map worth going to, waypoints in metres."""
    out = []
    for cells in _segments(frontier_mask(cost)):
        ordered = _path_order(cells)
        length = len(ordered) * resolution
        if length < min_len_m:
            continue
        mid = ordered[len(ordered) // 2]
        nudged = _nudge(cost, int(mid[0]), int(mid[1]), max_cost, int(round(1.0 / resolution)))
        if nudged is None:
            continue
        r, c = nudged
        out.append(Frontier(xy=(origin[0] + (c + 0.5) * resolution, origin[1] + (r + 0.5) * resolution),
                            length_m=round(length, 2), n_cells=len(cells), cells=cells))
    return out


@dataclass
class Selector:
    """VLFM's _get_best_frontier state: the last choice and how often each (spot, frontier) pair was tried."""
    last_xy: Optional[Tuple[float, float]] = None
    last_value: float = float("-inf")
    tries: Dict[Tuple[int, int, int, int], int] = field(default_factory=dict)
    failed: List[Tuple[float, float]] = field(default_factory=list)   # waypoints Nav2 could not reach
    spot_m: float = 1.0              # robot positions are binned to this to count "the same spot"
    frontier_bin_m: float = 1.0

    def fail(self, xy: Tuple[float, float]):
        """Nav2 failed to reach this waypoint (or reached it and the frontier is still there): never pick a frontier
        within STICK_RADIUS_M of it again, and drop the hysteresis so the next choice is a fresh one."""
        self.failed.append((float(xy[0]), float(xy[1])))
        self.last_xy, self.last_value = None, float("-inf")

    def track(self, xy: Tuple[float, float]):
        """The chosen frontier's waypoint moved a little (segments shift as cells fill in): follow it without
        touching last_value (the value seen at the last tick, as VLFM's code keeps it), so the hysteresis still
        compares this tick's value with the last tick's."""
        self.last_xy = (float(xy[0]), float(xy[1]))

    def _ok(self, f: Frontier) -> bool:
        return all(math.dist(f.xy, xy) > STICK_RADIUS_M for xy in self.failed)

    def _key(self, robot_xy, f_xy):
        return (int(robot_xy[0] // self.spot_m), int(robot_xy[1] // self.spot_m),
                int(f_xy[0] // self.frontier_bin_m), int(f_xy[1] // self.frontier_bin_m))

    def choose(self, frontiers: Sequence[Frontier], robot_xy: Tuple[float, float]) -> Optional[int]:
        """Index of the frontier to go to, or None if there are none. Frontiers must carry their `value`."""
        if not frontiers:
            return None
        order = sorted(range(len(frontiers)), key=lambda i: frontiers[i].value, reverse=True)
        usable = [i for i in order if self._ok(frontiers[i])]
        if not usable:                                      # every frontier failed once: VLFM's fallback, the farthest
            chosen = max(range(len(frontiers)), key=lambda i: math.dist(frontiers[i].xy, robot_xy))
            self.last_xy, self.last_value = frontiers[chosen].xy, frontiers[chosen].value
            return chosen
        order = usable
        chosen = None
        if self.last_xy is not None:                        # hysteresis: stick to the last one while it holds up
            for i in order:
                f = frontiers[i]
                if math.dist(f.xy, self.last_xy) <= STICK_RADIUS_M and f.value + STICK_MARGIN > self.last_value:
                    chosen = i
                    break
        if chosen is None:                                  # best non-cyclic
            for i in order:
                if self.tries.get(self._key(robot_xy, frontiers[i].xy), 0) < MAX_TRIES_PER_SPOT:
                    chosen = i
                    break
        if chosen is None:                                  # all cyclic: VLFM's code takes the farthest
            chosen = max(order, key=lambda i: math.dist(frontiers[i].xy, robot_xy))
        f = frontiers[chosen]
        k = self._key(robot_xy, f.xy)
        if self.last_xy is None or math.dist(f.xy, self.last_xy) > STICK_RADIUS_M:
            self.tries[k] = self.tries.get(k, 0) + 1
        self.last_xy, self.last_value = f.xy, f.value
        return chosen
