#!/usr/bin/env python3
"""Coverage tour over a site map (Nav2 map_server pgm/yaml) for the semantic mapping run.

  python3 coverage_tour.py tools/sim/godot_circuit4_map.yaml --start -2.13 0.35 [--spacing 3.0]
  -> prints "x,y" waypoints (map frame), one per line, ending back at the start

Reuses plan_tour() from tools/splat/extract_geometry.py (one stop per spacing x spacing tile of reachable,
open floor; walking-distance order + 2-opt; via-points where a straight leg would thread a squeeze), so the
mapping drive sees every part of the site the robot can reach.
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "splat"))
from extract_geometry import plan_tour   # noqa: E402


def load_map(yaml_path):
    """-> (occ, free, res, origin_xy, height) with row 0 = SOUTH (y up), like the map frame."""
    meta = yaml.safe_load(open(yaml_path))
    pgm = os.path.join(os.path.dirname(yaml_path), meta["image"])
    with open(pgm, "rb") as f:
        assert f.readline().strip() == b"P5"
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = (int(v) for v in line.split())
        f.readline()
        img = np.frombuffer(f.read(w * h), np.uint8).reshape(h, w)[::-1]     # flip: row 0 = south
    return img < 50, img > 250, float(meta["resolution"]), meta["origin"][:2], h


def tour(yaml_path, start_xy, spacing=3.0, clearance=0.5):
    occ, free, res, (x0, y0), _ = load_map(yaml_path)
    cell = lambda x, y: (int((y - y0) / res), int((x - x0) / res))
    spawn = cell(*start_xy)
    stops = plan_tour(occ, free, spawn, res, clearance, spacing)
    return [(x0 + (c + 0.5) * res, y0 + (r + 0.5) * res) for r, c in stops]


def seen_positions(run_dirs):
    """Camera positions (x, y) of every keyframe in earlier mapping runs."""
    pts = []
    for d in run_dirs:
        idx = os.path.join(d, "index.jsonl")
        if os.path.exists(idx):
            for line in open(idx):
                T = json.loads(line)["T_map_cam"]
                pts.append((T[0][3], T[1][3]))
    return np.array(pts).reshape(-1, 2)


def unseen(stops, seen, radius):
    """Stops with no earlier keyframe within `radius` m, in tour order."""
    if not len(seen):
        return list(stops)
    return [s for s in stops if np.min(np.hypot(seen[:, 0] - s[0], seen[:, 1] - s[1])) > radius]


def nearest_first(stops, start):
    """Greedy nearest-neighbour order from `start`: the leftover stops of a resumed tour are scattered, and the
    original order would drive back across covered ground first (straight-line distance; good enough here)."""
    left, out, here = list(stops), [], tuple(start)
    while left:
        k = min(range(len(left)), key=lambda i: (left[i][0] - here[0]) ** 2 + (left[i][1] - here[1]) ** 2)
        here = left.pop(k)
        out.append(here)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_yaml")
    ap.add_argument("--start", type=float, nargs=2, required=True, help="robot x y (map frame)")
    ap.add_argument("--spacing", type=float, default=3.0)
    ap.add_argument("--clearance", type=float, default=0.5)   # 0.7 seals circuit4's aisles: 1 stop
    ap.add_argument("--skip-seen", nargs="*", default=[], metavar="RUN_DIR",
                    help="drop stops an earlier mapping run already passed within --seen-radius of")
    ap.add_argument("--seen-radius", type=float, default=2.0)
    a = ap.parse_args()
    stops = tour(a.map_yaml, a.start, a.spacing, a.clearance)
    if a.skip_seen:
        kept = nearest_first(unseen(stops, seen_positions(a.skip_seen), a.seen_radius), a.start)
        print(f"# {len(stops)} stops, {len(stops) - len(kept)} already covered by {len(a.skip_seen)} run(s), "
              f"{len(kept)} to drive", file=sys.stderr)
        stops = kept
    for x, y in stops:
        print(f"{x:.2f},{y:.2f}")


if __name__ == "__main__":
    main()
