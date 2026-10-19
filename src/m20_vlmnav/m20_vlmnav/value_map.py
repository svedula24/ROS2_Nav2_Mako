"""The value map of VLFM (Vision-Language Frontier Maps, Yokoyama et al. 2023), ported from its code.

A top-down grid, the same cells as the exploration costmap, holding for every cell a VALUE (how promising this spot
looked for the target, 0..1-ish: the image-text cosine of the frame that looked that way) and a CONFIDENCE (how
central that spot was in that frame's view). One observation = one camera frame, scored once by the image-text
matcher, painted onto the cells inside the camera's cone of view:

    cone        the camera FOV out to max_depth, OCCLUDED per image column: a cell whose FORWARD distance (the
                camera's z, what a depth image measures) is beyond the depth measured in its column gets nothing
                (vlfm value_map._process_local_data: depth_row = max over rows); columns map to bearings through
                the pinhole (tan), not linearly
    confidence  angle = remap(|theta|, 0, fov/2, 0, pi/2);  c = cos(angle)^2;  c = remap(c, 0, 1, 0.25, 1)
                (vlfm _get_confidence_mask, _min_confidence = 0.25): 1 on the optical axis, 0.25 at the edge
    fusion      first, in both modes (vlfm _fuse_new_data): a new confidence below 0.35 (_decision_threshold) AND
                below the stored one is dropped (a weak edge-of-view glance never dilutes a good look).
                "max_confidence" (vlfm's default, what ran on Spot): where the new confidence beats the stored one,
                value and confidence are REPLACED.
                "weighted" (the paper's equations 1-2): w1 = c_old/(c_old+c_new), w2 = c_new/(c_old+c_new);
                value = w1*v_old + w2*v_new; confidence = w1*c_old + w2*c_new.
    read        the value at a frontier = the median of the non-zero cells within 0.5 m (vlfm
                pixel_value_within_radius, reduction "median"); -1 if no cell there was ever observed.
The robot's pose comes from odometry (as VLFM's did); no map frame is assumed beyond the grid's own origin.
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

MIN_CONFIDENCE = 0.25
DECISION_THRESHOLD = 0.35
PROMPT = "Seems like there is a {target} ahead."      # VLFM text_prompt, target_object substituted


def remap(v, lo_in, hi_in, lo_out, hi_out):
    return (v - lo_in) / (hi_in - lo_in) * (hi_out - lo_out) + lo_out


def depth_row_from_image(depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    """Per-column observed depth: the largest valid depth in each column (VLFM takes the max over rows), clipped to
    [min_depth, max_depth]. A column with no valid pixel counts as open to max_depth. depth: HxW metres, 0/NaN = none."""
    d = np.where(np.isfinite(depth) & (depth > 0.0), depth, 0.0)
    row = d.max(axis=0)
    row = np.where(row > 0.0, row, max_depth)
    return np.clip(row, min_depth, max_depth)


@dataclass
class ValueMap:
    width: int                       # cells
    height: int
    resolution: float                # m per cell
    origin: tuple                    # (x, y) of cell (0, 0)'s corner, metres
    fov: float                       # camera horizontal FOV, radians
    max_depth: float = 5.0
    min_depth: float = 0.5
    fusion: str = "max_confidence"   # or "weighted"

    def __post_init__(self):
        self.value = np.zeros((self.height, self.width), np.float32)
        self.conf = np.zeros((self.height, self.width), np.float32)
        self.n_updates = 0

    # ---------- geometry ----------
    def cell(self, x: float, y: float):
        return (int(math.floor((y - self.origin[1]) / self.resolution)),
                int(math.floor((x - self.origin[0]) / self.resolution)))

    def _observation(self, x: float, y: float, yaw: float, depth_row: np.ndarray):
        """-> (rows, cols, confidence) of the cells inside this observation's occluded cone."""
        r = int(math.ceil(self.max_depth / self.resolution))
        r0, c0 = self.cell(x, y)
        rows = np.arange(max(r0 - r, 0), min(r0 + r + 1, self.height))
        cols = np.arange(max(c0 - r, 0), min(c0 + r + 1, self.width))
        cc, rr = np.meshgrid(cols, rows)
        # cell centres relative to the robot, in the robot frame (x forward, y left)
        wx = self.origin[0] + (cc + 0.5) * self.resolution - x
        wy = self.origin[1] + (rr + 0.5) * self.resolution - y
        fx = wx * math.cos(yaw) + wy * math.sin(yaw)
        fy = -wx * math.sin(yaw) + wy * math.cos(yaw)
        dist = np.hypot(fx, fy)
        theta = np.arctan2(fy, fx)
        inside = (dist <= self.max_depth) & (np.abs(theta) <= self.fov / 2) & (dist > 1e-6)
        # occlusion: the depth measured in the image column that looks at this bearing. A depth image holds the
        # FORWARD distance (z), so compare the cell's forward component, not its ray length; columns run left to
        # right (+fov/2 .. -fov/2 in the robot's left-positive convention) through the pinhole: u = tan(theta)/tan(fov/2)
        n = len(depth_row)
        u = np.clip(np.tan(np.clip(theta, -self.fov / 2, self.fov / 2)) / math.tan(self.fov / 2), -1.0, 1.0)
        col = np.clip(((1.0 - u) / 2.0 * (n - 1)).round().astype(int), 0, n - 1)
        inside &= fx <= depth_row[col]
        angle = remap(np.abs(theta), 0.0, self.fov / 2, 0.0, math.pi / 2)
        conf = remap(np.cos(angle) ** 2, 0.0, 1.0, MIN_CONFIDENCE, 1.0)
        conf = np.where(inside, conf, 0.0).astype(np.float32)
        return rr, cc, conf

    # ---------- the two operations ----------
    def update(self, score: float, x: float, y: float, yaw: float, depth_row: np.ndarray) -> int:
        """Paint one scored observation. -> number of cells touched."""
        rr, cc, c_new = self._observation(x, y, yaw, depth_row)
        c_old = self.conf[rr, cc]
        v_old = self.value[rr, cc]
        # vlfm _fuse_new_data, both modes: a weak look (edge of view) over a better stored one is dropped
        c_new = np.where((c_new < DECISION_THRESHOLD) & (c_new < c_old), 0.0, c_new).astype(np.float32)
        if self.fusion == "max_confidence":
            take = c_new > c_old
            self.value[rr, cc] = np.where(take, np.float32(score), v_old)
            self.conf[rr, cc] = np.where(take, c_new, c_old)
            touched = int(take.sum())
        elif self.fusion == "weighted":
            den = c_old + c_new
            w1 = np.where(den > 0, c_old / np.where(den > 0, den, 1.0), 0.0)
            w2 = np.where(den > 0, c_new / np.where(den > 0, den, 1.0), 0.0)
            self.value[rr, cc] = w1 * v_old + w2 * np.float32(score)
            self.conf[rr, cc] = w1 * c_old + w2 * c_new
            touched = int((c_new > 0).sum())
        else:
            raise ValueError(f"unknown fusion {self.fusion!r}")
        self.n_updates += 1
        return touched

    def read(self, x: float, y: float, radius_m: float = 0.5, reduction: str = "median") -> float:
        """The value around a point: median (VLFM's default) / max / mean of the cells within the radius that were
        ever observed (confidence > 0); -1.0 if none."""
        r0, c0 = self.cell(x, y)
        r = max(1, int(round(radius_m / self.resolution)))
        rs = slice(max(r0 - r, 0), min(r0 + r + 1, self.height)); cs = slice(max(c0 - r, 0), min(c0 + r + 1, self.width))
        rr, cc = np.mgrid[rs, cs]
        disk = (rr - r0) ** 2 + (cc - c0) ** 2 <= r * r
        seen = disk & (self.conf[rs, cs] > 0)
        if not seen.any():
            return -1.0
        vals = self.value[rs, cs][seen]
        return float({"median": np.median, "max": np.max, "mean": np.mean}[reduction](vals))
