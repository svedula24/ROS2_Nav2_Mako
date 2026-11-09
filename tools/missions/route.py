"""Waypoint routes, one format for the sim and the robot (pure, no ROS).

A route is a yaml file or a list of "X,Y[,YAW]" arguments; each waypoint has a map-frame position, an optional
heading and label, and an optional pause. A missing heading means "face the next waypoint" (the last one keeps the
heading it arrived with), so the robot never turns to an arbitrary direction at every stop (the old CLI sent every
goal with yaw 0 = face east).

  frame: map
  site: circuit4
  waypoints:
    - {x: -2.13, y: -3.0, yaw: -1.57, label: wp_a}
    - [0.5, -8.5]                      # yaw auto
    - {x: -2.13, y: 0.35, yaw: 1.57, wait_s: 2}

Used by tools/dev/m20 (`mission --route file.yaml`, `mission "X,Y,YAW" ...`) to build RunMission items, and by the
waypoint eval to know the goals. Headings are radians; positions metres.
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import yaml


@dataclass
class Waypoint:
    x: float
    y: float
    yaw: Optional[float] = None       # None = auto (toward the next waypoint)
    label: str = ""
    wait_s: float = 0.0


def parse_wp_arg(text: str, index: int = 0) -> Waypoint:
    """ "X,Y" or "X,Y,YAW" -> Waypoint. Anything else is an error, never a guess."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) not in (2, 3) or any(p == "" for p in parts):
        raise ValueError(f"waypoint {text!r}: expected X,Y or X,Y,YAW")
    try:
        nums = [float(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"waypoint {text!r}: {e}") from None
    return Waypoint(nums[0], nums[1], nums[2] if len(nums) == 3 else None, label=f"wp_{index}")


def load_route(path: str) -> List[Waypoint]:
    doc = yaml.safe_load(open(path))
    if not isinstance(doc, dict) or not isinstance(doc.get("waypoints"), list) or not doc["waypoints"]:
        raise ValueError(f"{path}: a route needs a non-empty 'waypoints' list")
    frame = doc.get("frame", "map")
    if frame != "map":
        raise ValueError(f"{path}: frame {frame!r} not supported (goals are map-frame metres)")
    out = []
    for i, w in enumerate(doc["waypoints"]):
        if isinstance(w, (list, tuple)):
            if len(w) not in (2, 3):
                raise ValueError(f"{path}: waypoint {i} list must be [x, y] or [x, y, yaw]")
            out.append(Waypoint(float(w[0]), float(w[1]), float(w[2]) if len(w) == 3 else None, label=f"wp_{i}"))
        elif isinstance(w, dict):
            if "x" not in w or "y" not in w:
                raise ValueError(f"{path}: waypoint {i} needs x and y")
            out.append(Waypoint(float(w["x"]), float(w["y"]), None if w.get("yaw") is None else float(w["yaw"]),
                                label=str(w.get("label") or f"wp_{i}"), wait_s=float(w.get("wait_s", 0.0))))
        else:
            raise ValueError(f"{path}: waypoint {i} must be a list or a mapping")
    return out


def fill_auto_yaw(wps: Sequence[Waypoint], start_yaw: Optional[float] = None) -> List[Waypoint]:
    """Resolve yaw=None: face the next waypoint; the last one keeps the heading it arrives with (from the previous
    waypoint, or start_yaw, or 0 when nothing is known)."""
    out = [Waypoint(w.x, w.y, w.yaw, w.label, w.wait_s) for w in wps]
    for i, w in enumerate(out):
        if w.yaw is not None:
            continue
        if i + 1 < len(out):
            nxt = out[i + 1]
            w.yaw = math.atan2(nxt.y - w.y, nxt.x - w.x)
        elif i > 0:
            prv = out[i - 1]
            w.yaw = math.atan2(w.y - prv.y, w.x - prv.x)
        else:
            w.yaw = 0.0 if start_yaw is None else float(start_yaw)
    return out


def quaternion_z(yaw: float):
    """(z, w) of a yaw-only quaternion."""
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)
