#!/usr/bin/env python3
"""Carry the object ids pinned in cases.yaml over to a rebuilt object map.
  python3 tools/nlnav_eval/remap_ids.py OLD_MAP.json NEW_MAP.json [--cases tools/nlnav_eval/cases.yaml] [--write]
Rebuilding the map (new runs, new thresholds) renumbers objects. Each pinned id (`ids: [...]`, `near_id: N`) is
matched to the new object closest in 3D among those with embedding cosine >= 0.75 (the builder's own merge
rule), within 1.0 m. Prints the mapping; --write rewrites only those numbers in cases.yaml (comments kept).
An id with no match is reported and left alone, so the case fails visibly instead of silently changing target.
"""
import argparse
import base64
import json
import re

import numpy as np


def load(path):
    objs = {}
    for o in json.load(open(path))["objects"]:
        e = np.frombuffer(base64.b64decode(o["emb"]), np.float16).astype(np.float32)
        objs[int(o["id"])] = (np.array(o["xyz"], float), e / (np.linalg.norm(e) + 1e-9), o["label"], o["score"])
    return objs


def match(old, new, max_dist=1.0, min_cos=0.75):
    p, e, label, _ = old
    best = None
    for nid, (q, f, nl, ns) in new.items():
        d, c = float(np.linalg.norm(p - q)), float(e @ f)
        if d <= max_dist and c >= min_cos and (best is None or d < best[1]):
            best = (nid, d, c, nl)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old_map")
    ap.add_argument("new_map")
    ap.add_argument("--cases", default="tools/nlnav_eval/cases.yaml")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    old, new = load(a.old_map), load(a.new_map)
    text = open(a.cases).read()
    pinned = sorted({int(x) for m in re.finditer(r"ids:\s*\[([^\]]*)\]", text) for x in re.findall(r"-?\d+", m.group(1))
                     if int(x) > 0} | {int(m.group(1)) for m in re.finditer(r"near_id:\s*(\d+)", text)})
    mapping = {}
    for oid in pinned:
        if oid not in old:
            print(f"  #{oid}: not in the old map (already remapped?)"); continue
        m = match(old[oid], new)
        if m is None:
            print(f"  #{oid} {old[oid][2]}: NO MATCH in the new map: left as is"); continue
        mapping[oid] = m[0]
        print(f"  #{oid} {old[oid][2]} -> #{m[0]} {m[3]}  ({m[1]:.2f} m, cos {m[2]:.2f})")
    if a.write and mapping:
        sub = lambda s: re.sub(r"-?\d+", lambda d: str(mapping.get(int(d.group()), int(d.group()))), s)
        text = re.sub(r"(ids:\s*\[)([^\]]*)(\])", lambda m: m.group(1) + sub(m.group(2)) + m.group(3), text)
        text = re.sub(r"(near_id:\s*)(\d+)", lambda m: m.group(1) + str(mapping.get(int(m.group(2)), int(m.group(2)))), text)
        open(a.cases, "w").write(text)
        print(f"wrote {a.cases}")


if __name__ == "__main__":
    main()
