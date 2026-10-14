"""Frozen-camera detection for the keyframe recorder (no ROS, so it can be unit-tested).

A camera picture that is byte-identical to the last SAVED keyframe while the robot has moved means the sim's frame
froze (seen during a tip-over, 2026-10-13): saving it would place every object in it at a wrong position. The recorder
asks `is_frozen()` only after it has decided the robot moved enough for a new keyframe, and calls `saved()` when it
writes one.
"""
import hashlib


class FrozenFrameGuard:
    def __init__(self):
        self._last = None                     # md5 of the last saved colour image

    @staticmethod
    def _digest(data) -> bytes:
        return hashlib.md5(bytes(data)).digest()

    def is_frozen(self, data) -> bool:
        return self._last is not None and self._digest(data) == self._last

    def saved(self, data):
        self._last = self._digest(data)
