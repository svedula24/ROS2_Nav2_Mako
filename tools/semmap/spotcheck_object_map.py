#!/usr/bin/env python3
"""Spot-check an object map by eye: label inventory + a contact sheet of each object's best crop.

  python3 tools/semmap/spotcheck_object_map.py ~/m20_sim/semmap/circuit4/object_map.json [--out sheet.jpg] [--top 48]

Prints one line per object (id, label, xyz, size, sightings, score) grouped by label, then writes a contact
sheet: the best crop of each object with "id label score n" and its map position, sorted by score. The
positions are what nlnav drives to, so a crop that doesn't show its label is a map bug, not a detector one.
"""
import argparse
import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "m20_nlnav"))
from m20_nlnav.object_map import ObjectMap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("object_map")
    ap.add_argument("--out", help="contact sheet path (default: next to the map)")
    ap.add_argument("--top", type=int, default=48)
    ap.add_argument("--tile", type=int, default=160)
    a = ap.parse_args()
    omap = ObjectMap.load(a.object_map)
    base = os.path.dirname(os.path.abspath(a.object_map))
    objs = sorted(omap.objects, key=lambda o: -o.score)
    print(f"{len(objs)} objects, site {omap.site!r}")
    for label, n in Counter(o.label for o in objs).most_common():
        print(f"  {n:3d}  {label}")
    print()
    for o in objs:
        print(f"#{o.id:<3} {o.label:18} at ({o.xyz[0]:6.2f},{o.xyz[1]:6.2f},{o.xyz[2]:5.2f}) size "
              f"({o.size[0]:.2f},{o.size[1]:.2f},{o.size[2]:.2f}) n={o.n:<3} score={o.score:.2f}")
    T, cols = a.tile, 8
    rows = (min(a.top, len(objs)) + cols - 1) // cols
    sheet = np.full((max(1, rows) * (T + 28), cols * T, 3), 30, np.uint8)
    for i, o in enumerate(objs[: a.top]):
        r, c = divmod(i, cols)
        y0, x0 = r * (T + 28), c * T
        crop = cv2.imread(os.path.join(base, o.crop)) if o.crop else None
        if crop is not None and crop.size:
            h, w = crop.shape[:2]
            s = T / max(h, w)
            crop = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))))
            sheet[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
        cv2.putText(sheet, f"#{o.id} {o.label[:14]}", (x0 + 2, y0 + T + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(sheet, f"{o.score:.2f} n{o.n} ({o.xyz[0]:.1f},{o.xyz[1]:.1f})", (x0 + 2, y0 + T + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 220, 255), 1)
    out = a.out or os.path.join(base, "object_map_sheet.jpg")
    cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"\ncontact sheet -> {out}")


if __name__ == "__main__":
    main()
