"""Object map format v1 + the builder's geometry and merge rules (numpy only)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "semmap"))
from m20_nlnav.object_map import (MapObject, ObjectMap, aabb_iou, decode_emb, encode_emb, merge_into,  # noqa: E402
                                  normalize, owl_prob, same_object)
from build_object_map import place                                                              # noqa: E402

rng = np.random.default_rng(0)


def unit(i=None):
    v = rng.normal(size=512) if i is None else np.eye(512)[i]
    return normalize(v)


def obj(i, xyz=(0, 0, 0.5), size=(0.5, 0.5, 1.0), emb=None, label="ladder"):
    return MapObject(i, label, tuple(xyz), tuple(size), emb if emb is not None else unit(i), -0.5, 10.0, 0.8, 3, 1.0, 0.8)


def test_embedding_codec_roundtrip_float16():
    v = unit()
    back = decode_emb(encode_emb(v))
    assert back.shape == (512,) and np.allclose(back, v, atol=1e-3)


def test_object_map_save_load_roundtrip(tmp_path):
    om = ObjectMap([obj(1), obj(2, xyz=(3, 1, 0.4), label="trash can")], site="circuit4", built="now")
    p = tmp_path / "om.json"
    om.save(p)
    back = ObjectMap.load(p)
    assert [o.label for o in back.objects] == ["ladder", "trash can"]
    assert np.allclose(back.objects[1].xyz, (3, 1, 0.4)) and back.site == "circuit4"
    assert np.allclose(back.objects[0].emb, om.objects[0].emb, atol=1e-3)
    E, sh, sc = back.matrix()
    assert E.shape == (2, 512) and sh.shape == (2,)


def test_owl_prob_matches_owlv2_head():
    e = unit(3)
    assert owl_prob(e[None], e, np.array([-0.5]), np.array([10.0]))[0] > 0.99      # same direction
    assert owl_prob(e[None], unit(4), np.array([-0.5]), np.array([10.0]))[0] < 0.01  # orthogonal


def test_same_object_needs_both_space_and_embedding():
    o = obj(1, emb=unit(7))
    assert same_object(o, (0.2, 0.1, 0.5), (0.5, 0.5, 1.0), unit(7))                # close + same emb
    assert not same_object(o, (0.2, 0.1, 0.5), (0.5, 0.5, 1.0), unit(8))            # close, different thing
    assert not same_object(o, (3.0, 0.0, 0.5), (0.5, 0.5, 1.0), unit(7))            # same thing, far away
    assert aabb_iou((0, 0, 0), (1, 1, 1), (0, 0, 0), (1, 1, 1)) == 1.0


def test_merge_running_means_and_best_label():
    o = obj(1, xyz=(0, 0, 0.5), label="object")
    o.label_score = 0.05
    merge_into(o, (1.0, 0.0, 0.5), (0.5, 0.5, 1.0), o.emb, -0.5, 10.0, 0.9, 5.0, "ladder", 0.7)
    assert o.n == 4 and 0.2 < o.xyz[0] < 0.3                  # 1/4 weight towards the new sighting
    assert o.label == "ladder" and o.score == 0.9 and o.last_seen == 5.0


def test_place_picks_the_object_not_the_wall_behind_it():
    K = {"fx": 320.0, "fy": 320.0, "cx": 319.5, "cy": 239.5}
    depth = np.full((480, 640), 5.0, np.float32)              # wall 5 m away
    depth[200:280, 300:340] = 2.0                             # thin object 2 m away, centre of the box
    box = (280, 180, 360, 300)                                # detection box larger than the object
    T = np.eye(4)                                             # camera = map frame (optical axes)
    centre, size = place(depth, box, K, T)
    assert abs(centre[2] - 2.0) < 0.05                        # z (optical depth) is the object's
    assert size[2] < 0.3                                      # not smeared to the wall
    assert place(np.zeros((480, 640), np.float32), box, K, T) is None


def test_confident_relabels_low_scores_and_can_drop():
    from dataclasses import replace
    from m20_nlnav.object_map import MapObject, ObjectMap
    import numpy as np
    e = np.zeros(512, np.float32); e[0] = 1.0
    base = MapObject(1, "refrigerator", (0.0, 0.0, 0.5), (0.5, 0.5, 1.0), e, -0.5, 10.0, 0.13, 3, 0.0, 0.13)
    m = ObjectMap([base, replace(base, id=2, label="ladder", score=0.9), replace(base, id=3, label="object", score=0.2)])
    c = m.confident(min_label=0.4)
    assert [o.label for o in c.objects] == ["object", "ladder", "object"]
    assert np.array_equal(c.objects[0].emb, e) and c.objects[0].score == 0.13      # embedding/score untouched
    assert m.objects[0].label == "refrigerator"                                     # original map unchanged
    assert [o.id for o in m.confident(min_label=0.4, min_keep=0.15).objects] == [2, 3]
