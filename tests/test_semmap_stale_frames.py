"""build_object_map.stale_frames: keyframes with a byte-identical (frozen) image are all dropped."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools", "semmap"))
from build_object_map import stale_frames  # noqa: E402


def test_frozen_image_group_is_dropped_whole(tmp_path):
    imgs = {"kf_00001": b"view-a", "kf_00002": b"view-b", "kf_00003": b"frozen", "kf_00004": b"frozen",
            "kf_00005": b"frozen", "kf_00006": b"view-c"}
    for k, v in imgs.items():
        (tmp_path / f"{k}.jpg").write_bytes(v)
    frames = [{"id": k} for k in imgs]
    assert stale_frames(str(tmp_path), frames) == {"kf_00003", "kf_00004", "kf_00005"}


def test_no_duplicates_drops_nothing(tmp_path):
    for k in ("kf_00001", "kf_00002"):
        (tmp_path / f"{k}.jpg").write_bytes(k.encode())
    assert stale_frames(str(tmp_path), [{"id": "kf_00001"}, {"id": "kf_00002"}]) == set()


def test_recorder_frozen_frame_guard():
    """keyframe_recorder's check (frozen_frames.FrozenFrameGuard): an image identical to the last SAVED keyframe is
    frozen; a new image isn't; the very first image never is."""
    from frozen_frames import FrozenFrameGuard
    g = FrozenFrameGuard()
    assert not g.is_frozen(b"A")              # nothing saved yet
    g.saved(b"A")
    assert g.is_frozen(b"A")                  # same picture again while the robot moved
    assert g.is_frozen(bytearray(b"A"))       # message data may arrive as array/bytearray
    assert not g.is_frozen(b"B")
    g.saved(b"B")
    assert not g.is_frozen(b"A")              # compares with the LAST saved image only
