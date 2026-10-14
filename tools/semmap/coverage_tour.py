#!/usr/bin/env python3
"""Coverage tour over a site map (Nav2 map_server pgm/yaml) for the semantic mapping run.

  python3 coverage_tour.py tools/sim/godot_circuit4_map.yaml --start -2.13 0.35 [--spacing 3.0]
  -> prints "x,y" waypoints (map frame), one per line, ending back at the start

Reuses plan_tour() from tools/splat/extract_geometry.py (one stop per spacing x spacing tile of reachable,
open floor; walking-distance order + 2-opt; via-points where a straight leg would thread a squeeze), so the
mapping drive sees every part of the site the robot can reach.
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_yaml")
    ap.add_argument("--start", type=float, nargs=2, required=True, help="robot x y (map frame)")
    ap.add_argument("--spacing", type=float, default=3.0)
    ap.add_argument("--clearance", type=float, default=0.5)   # 0.7 seals circuit4's aisles: 1 stop
    a = ap.parse_args()
    for x, y in tour(a.map_yaml, a.start, a.spacing, a.clearance):
        print(f"{x:.2f},{y:.2f}")


if __name__ == "__main__":
    main()
