"""Object map format v1: the contract between the semantic-map builder and the language side.

Built offline (tools/semmap/build_object_map.py) from a recorded mapping run; read at run time by
m20_nlnav. One JSON file per site:

{
  "version": 1,
  "site": "circuit4",
  "frame": "map",                       # the site's ENU / Nav2 map frame, metres
  "built": "2026-10-14T10:00:00",
  "embedding": "owlv2-base-patch16-ensemble/class_embed",   # 512-d, L2-normalised, base64 float16
  "objects": [
    {"id": 7, "label": "ladder", "label_score": 0.81,       # best office-vocabulary class (readability)
     "xyz": [x, y, z], "size": [sx, sy, sz],                # world AABB centre / extent
     "emb": "<base64 float16 x512>",                        # OWLv2 class embedding (mean over sightings)
     "shift": -0.43, "scale": 7.9,                          # OWLv2 per-box logit calibration (mean)
     "score": 0.81, "n": 6, "last_seen": 1791931534.3,      # max detection score, sightings, unix time
     "crop": "crops/obj_0007.jpg"}                          # best keyframe crop (relative to the map file)
  ]
}

Scoring a text phrase against an object reproduces OWLv2's own detection head:
    p = sigmoid((dot(emb, text_emb) + shift) * scale)          (both embeddings L2-normalised)
so any phrase, including ones nobody listed at mapping time, gets an OWLv2-calibrated probability.
"""
import base64
import json
import math
from dataclasses import dataclass, field, replace
from typing import List, Optional

import numpy as np

VERSION = 1


def encode_emb(v) -> str:
    return base64.b64encode(np.asarray(v, np.float16).tobytes()).decode("ascii")


def decode_emb(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), np.float16).astype(np.float32)


def normalize(v):
    v = np.asarray(v, np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


@dataclass
class MapObject:
    id: int
    label: str
    xyz: tuple
    size: tuple
    emb: np.ndarray                      # (512,) normalised
    shift: float
    scale: float
    score: float = 0.0
    n: int = 1
    last_seen: float = 0.0
    label_score: float = 0.0
    crop: str = ""

    @property
    def xy(self):
        return self.xyz[0], self.xyz[1]

    @property
    def radius(self) -> float:
        return 0.5 * max(self.size[0], self.size[1]) if self.size else 0.2

    def to_json(self):
        return {"id": int(self.id), "label": self.label, "label_score": round(float(self.label_score), 3),
                "xyz": [round(float(v), 3) for v in self.xyz], "size": [round(float(v), 3) for v in self.size],
                "emb": encode_emb(self.emb), "shift": round(float(self.shift), 4), "scale": round(float(self.scale), 4),
                "score": round(float(self.score), 3), "n": int(self.n), "last_seen": float(self.last_seen),
                "crop": self.crop}

    @staticmethod
    def from_json(d):
        return MapObject(id=int(d["id"]), label=d.get("label", "object"), xyz=tuple(d["xyz"]), size=tuple(d["size"]),
                         emb=normalize(decode_emb(d["emb"])), shift=float(d["shift"]), scale=float(d["scale"]),
                         score=float(d.get("score", 0.0)), n=int(d.get("n", 1)), last_seen=float(d.get("last_seen", 0.0)),
                         label_score=float(d.get("label_score", 0.0)), crop=d.get("crop", ""))


@dataclass
class ObjectMap:
    objects: List[MapObject] = field(default_factory=list)
    site: str = ""
    frame: str = "map"
    built: str = ""
    embedding: str = "owlv2-base-patch16-ensemble/class_embed"

    def save(self, path):
        doc = {"version": VERSION, "site": self.site, "frame": self.frame, "built": self.built,
               "embedding": self.embedding, "objects": [o.to_json() for o in self.objects]}
        with open(path, "w") as f:
            json.dump(doc, f)

    @staticmethod
    def load(path):
        d = json.load(open(path))
        if d.get("version") != VERSION:
            raise ValueError(f"object map version {d.get('version')} != {VERSION}")
        return ObjectMap([MapObject.from_json(o) for o in d["objects"]], d.get("site", ""), d.get("frame", "map"),
                         d.get("built", ""), d.get("embedding", ""))

    def confident(self, min_label: float = 0.4, min_keep: float = 0.0):
        """A copy keeping objects with score >= min_keep; those below min_label keep their embedding (still
        matchable by any phrase) but are relabeled "object": the detector's word is unreliable at low score
        (circuit4: 22 "refrigerators" that were white cabinets), the embedding much less so."""
        objs = []
        for o in self.objects:
            if o.score < min_keep:
                continue
            if o.score < min_label and o.label != "object":
                o = replace(o, label="object")
            objs.append(o)
        return ObjectMap(objs, self.site, self.frame, self.built, self.embedding)

    def matrix(self):
        """-> (E (n,512), shift (n,), scale (n,)) for vectorised scoring."""
        if not self.objects:
            return np.zeros((0, 512), np.float32), np.zeros(0), np.zeros(0)
        return (np.stack([o.emb for o in self.objects]), np.array([o.shift for o in self.objects]),
                np.array([o.scale for o in self.objects]))


def owl_prob(emb, text_emb, shift, scale):
    """OWLv2's detection head on stored embeddings: sigmoid((emb . text + shift) * scale)."""
    logits = (np.asarray(emb) @ normalize(text_emb) + shift) * scale
    return 1.0 / (1.0 + np.exp(-logits))


# ------------------------------------------------------------------ merging (used by the builder)
def aabb_iou(c1, s1, c2, s2) -> float:
    lo1, hi1 = np.subtract(c1, np.divide(s1, 2)), np.add(c1, np.divide(s1, 2))
    lo2, hi2 = np.subtract(c2, np.divide(s2, 2)), np.add(c2, np.divide(s2, 2))
    inter = np.prod(np.clip(np.minimum(hi1, hi2) - np.maximum(lo1, lo2), 0, None))
    union = np.prod(s1) + np.prod(s2) - inter
    return float(inter / union) if union > 0 else 0.0


def same_object(o: MapObject, xyz, size, emb, max_dist=0.6, min_iou=0.1, min_cos=0.75) -> bool:
    """ConceptGraphs-style association: close in 3D (centre distance or AABB overlap) AND similar in
    embedding space. Both, so two different objects side by side, or one object seen as two classes,
    are handled."""
    near = math.dist(o.xyz, xyz) < max_dist or aabb_iou(o.xyz, o.size, xyz, size) > min_iou
    return near and float(np.dot(o.emb, normalize(emb))) >= min_cos


def merge_into(o: MapObject, xyz, size, emb, shift, scale, score, stamp, label, label_score):
    """Running means for geometry, embedding and calibration; max score; the label of the best sighting."""
    n = o.n
    w = 1.0 / min(n + 1, 20)                  # settles after ~20 sightings, still tracks a moved object
    o.xyz = tuple((1 - w) * np.asarray(o.xyz) + w * np.asarray(xyz))
    o.size = tuple((1 - w) * np.asarray(o.size) + w * np.asarray(size))
    o.emb = normalize((1 - w) * o.emb + w * normalize(emb))
    o.shift = (1 - w) * o.shift + w * shift
    o.scale = (1 - w) * o.scale + w * scale
    if label_score > o.label_score:
        o.label, o.label_score = label, label_score
    o.score = max(o.score, score)
    o.n = n + 1
    o.last_seen = max(o.last_seen, stamp)
