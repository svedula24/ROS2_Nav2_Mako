#!/usr/bin/env python3
"""Offline phase, step 2: build the object map from a recorded mapping run (keyframe_recorder.py).

  python3 build_object_map.py <run dir> [--out <object_map.json>] [--min-sightings 2]

Per keyframe, ONE OWLv2 pass gives, for every candidate box: an objectness score, a 512-d class embedding in
the same space as OWLv2's text encoder, and the box's own logit shift/scale. We keep the boxes that are
object-like (objectness) or match the office vocabulary, place each one in 3D from depth (the largest depth
cluster at the box centre, so thin objects don't land on the wall behind them), and merge repeat sightings
ConceptGraphs-style (close in 3D AND similar embedding). Objects seen fewer than --min-sightings times are
dropped. Output: object map format v1 (src/m20_nlnav/m20_nlnav/object_map.py), plus best crops.
Runs in the m20_splat image (torch, transformers, OWLv2 weights baked in; GPU if available).
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "m20_nlnav"))
from m20_nlnav.object_map import MapObject, ObjectMap, merge_into, normalize, same_object   # noqa: E402
from m20_nlnav.world import DETECTOR_CLASSES                                                 # noqa: E402

OWL_MODEL = os.environ.get("OWL_MODEL", "google/owlv2-base-patch16-ensemble")


def owl_frame(model, proc, text, rgb, dev, top_k=40, obj_thr=0.10, cls_thr=0.20, nms_iou=0.5):
    """One OWLv2 pass -> list of dict(box px, emb (512), shift, scale, objectness, label idx, label score)."""
    import torch
    from torchvision.ops import nms
    h, w = rgb.shape[:2]
    inp = proc(images=rgb, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**inp, input_ids=text["input_ids"], attention_mask=text["attention_mask"])
        feats = out.image_embeds.reshape(1, -1, out.image_embeds.shape[-1])
        shift = model.class_head.logit_shift(feats)[0, :, 0]
        scale = (model.class_head.elu(model.class_head.logit_scale(feats)) + 1)[0, :, 0]
    obj = torch.sigmoid(out.objectness_logits[0])
    cls_p = torch.sigmoid(out.logits[0])                   # (patches, n_classes)
    cls_s, cls_i = cls_p.max(-1)
    keep = (obj > obj_thr) | (cls_s > cls_thr)
    if not keep.any():
        return []
    side = max(h, w)                                       # OWLv2 pads to a square: boxes live in it
    cx, cy, bw, bh = (out.pred_boxes[0] * side).unbind(-1)
    boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1)
    idx = keep.nonzero()[:, 0]
    rank = torch.maximum(obj[idx], cls_s[idx])
    idx = idx[nms(boxes[idx], rank, nms_iou)][:top_k]
    res = []
    for i in idx.tolist():
        x0, y0, x1, y1 = boxes[i].tolist()
        if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
            continue
        res.append({"box": (max(0, x0), max(0, y0), min(w, x1), min(h, y1)),
                    "emb": normalize(out.class_embeds[0, i].float().cpu().numpy()),
                    "shift": float(shift[i]), "scale": float(scale[i]), "objectness": float(obj[i]),
                    "label": int(cls_i[i]), "label_score": float(cls_s[i])})
    return res


def place(depth, box, K, T_map_cam, band=0.35, max_range=6.0):
    """3D AABB (map frame) of a detection: back-project the box's depth pixels within `band` of the depth at
    the box CENTRE (the largest cluster there), keep 10-90th percentile extents. -> (centre, size) or None."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    d = depth[y0:y1, x0:x1].astype(np.float32)
    if d.size < 25:
        return None
    hh, ww = d.shape
    core = d[hh // 3: max(hh // 3 + 1, 2 * hh // 3), ww // 3: max(ww // 3 + 1, 2 * ww // 3)]
    core = core[(core > 0.1) & (core < max_range)]
    if core.size < 5:
        return None
    zc = float(np.median(core))
    vv, uu = np.mgrid[y0:y1, x0:x1]
    m = (d > 0.1) & (np.abs(d - zc) < band)
    if m.sum() < 12:
        return None
    z = d[m]
    p_cam = np.stack([(uu[m] + 0.5 - K["cx"]) / K["fx"] * z, (vv[m] + 0.5 - K["cy"]) / K["fy"] * z, z], 1)
    T = np.asarray(T_map_cam)
    pw = p_cam @ T[:3, :3].T + T[:3, 3]
    lo, hi = np.percentile(pw, 10, axis=0), np.percentile(pw, 90, axis=0)
    size = np.clip(hi - lo, 0.08, 4.0)
    return ((lo + hi) / 2).tolist(), size.tolist()


def stale_frames(run_dir, frames):
    """Ids of keyframes whose colour image is byte-identical to ANOTHER keyframe's in the same run. A frozen camera
    picture saved at several poses places one detection at several map positions (circuit4 run 1: kf_00226-229, the
    tip-over, one view of a desk -> three "desks" 1.5 m apart). Every member of such a group is dropped: once the
    picture froze, nothing says which pose it really belongs to."""
    import hashlib
    groups = {}
    for fr in frames:
        with open(os.path.join(run_dir, fr["id"] + ".jpg"), "rb") as f:
            groups.setdefault(hashlib.md5(f.read()).hexdigest(), []).append(fr["id"])
    return {i for ids in groups.values() if len(ids) > 1 for i in ids}


def build(run_dirs, out_path, min_sightings=2, max_area=0.35, every=1, log=print):
    """run_dirs: one mapping run or several of the same site (e.g. a run that ended early + its --resume);
    their keyframes are pooled, so an object seen in both runs merges into one."""
    import cv2
    import torch                                           # heavy deps only here: place() stays numpy-only
    if isinstance(run_dirs, str):
        run_dirs = [run_dirs]
    frames, Ks = [], {}
    for rd in run_dirs:
        Ks[rd] = json.load(open(os.path.join(rd, "camera.json")))
        run_frames = [dict(json.loads(l), run=rd) for l in open(os.path.join(rd, "index.jsonl"))][::every]
        stale = stale_frames(rd, run_frames)
        if stale:
            log(f"[build] {os.path.basename(rd)}: dropping {len(stale)} keyframes with a frozen (repeated) image: "
                f"{sorted(stale)}")
        frames += [fr for fr in run_frames if fr["id"] not in stale]
    log(f"[build] {len(frames)} keyframes from {len(run_dirs)} run(s)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    proc = Owlv2Processor.from_pretrained(OWL_MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL).eval().to(dev)
    text = proc(text=[DETECTOR_CLASSES], return_tensors="pt").to(dev)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(os.path.join(out_dir, "crops"), exist_ok=True)
    objs, best_crop, t0 = [], {}, time.time()
    for k, fr in enumerate(frames):
        bgr = cv2.imread(os.path.join(fr["run"], fr["id"] + ".jpg"))
        depth = np.load(os.path.join(fr["run"], fr["id"] + "_depth.npy")).astype(np.float32)
        K = Ks[fr["run"]]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        for det in owl_frame(model, proc, text, rgb, dev):
            x0, y0, x1, y1 = det["box"]
            if (x1 - x0) * (y1 - y0) > max_area * W * H:
                continue                                   # frame-filling boxes = walls / splat blur
            placed = place(depth, det["box"], K, fr["T_map_cam"])
            if placed is None:
                continue
            xyz, size = placed
            label, lscore = DETECTOR_CLASSES[det["label"]], det["label_score"]
            score = max(det["objectness"], lscore)
            match = next((o for o in objs if same_object(o, xyz, size, det["emb"])), None)
            if match is None:
                match = MapObject(id=len(objs) + 1, label=label if lscore > 0.1 else "object", xyz=tuple(xyz),
                                  size=tuple(size), emb=det["emb"], shift=det["shift"], scale=det["scale"],
                                  score=score, n=1, last_seen=fr["stamp"], label_score=lscore)
                objs.append(match)
            else:
                merge_into(match, xyz, size, det["emb"], det["shift"], det["scale"], score, fr["stamp"], label, lscore)
            if score > best_crop.get(match.id, (0, None))[0]:
                pad = 0.15
                bx0, by0 = int(max(0, x0 - pad * (x1 - x0))), int(max(0, y0 - pad * (y1 - y0)))
                bx1, by1 = int(min(W, x1 + pad * (x1 - x0))), int(min(H, y1 + pad * (y1 - y0)))
                best_crop[match.id] = (score, bgr[by0:by1, bx0:bx1].copy())
        if (k + 1) % 25 == 0:
            log(f"[build] {k + 1}/{len(frames)} keyframes, {len(objs)} candidate objects ({time.time() - t0:.0f} s)")
    kept = [o for o in objs if o.n >= min_sightings]
    for i, o in enumerate(kept, 1):
        crop = best_crop.get(o.id, (0, None))[1]
        o.id = i
        if crop is not None and crop.size:
            o.crop = f"crops/obj_{i:04d}.jpg"
            cv2.imwrite(os.path.join(out_dir, o.crop), crop)
    om = ObjectMap(kept, site=os.path.basename(os.path.dirname(os.path.abspath(run_dirs[0]))),
                   built=datetime.datetime.now().isoformat(timespec="seconds"))
    om.save(out_path)
    labels = {}
    for o in kept:
        labels[o.label] = labels.get(o.label, 0) + 1
    log(f"[build] {len(kept)} objects (of {len(objs)} candidates, >= {min_sightings} sightings) -> {out_path}")
    log(f"[build] labels: {dict(sorted(labels.items(), key=lambda kv: -kv[1]))}")
    return om


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", help="one or more mapping runs of the same site")
    ap.add_argument("--out", help="default: <first run_dir>/../object_map.json")
    ap.add_argument("--min-sightings", type=int, default=2)
    ap.add_argument("--every", type=int, default=1, help="use every Nth keyframe (quick tests)")
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.run_dirs[0])), "object_map.json")
    build(a.run_dirs, out, a.min_sightings, every=a.every)


if __name__ == "__main__":
    main()
