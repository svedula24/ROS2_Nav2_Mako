"""World model for natural-language navigation: what the robot has seen and where it can go.

Plain data (no rclpy) so the matching/resolution logic is testable without ROS; nlnav_node.py
fills it from /detections/objects, /odom_true, the global costmap and the scene yaml.
"""
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# The detector's vocabulary (tools/splat/detector_node.py OFFICE; tests/test_nlnav_tier0.py keeps
# them in sync). Known even before they're seen, so "go to the printer" can answer "haven't seen
# one" instead of "didn't understand".
DETECTOR_CLASSES = ["office chair", "desk", "computer monitor", "laptop", "keyboard", "computer mouse",
                    "whiteboard", "filing cabinet", "bookshelf", "printer", "trash can", "cardboard box",
                    "coffee mug", "water bottle", "backpack", "potted plant", "door", "telephone",
                    "desk lamp", "sofa", "person", "table", "storage cabinet", "microwave",
                    "refrigerator", "traffic cone", "ladder", "projector screen", "clock", "book"]

# Words people use -> detector classes (tools/splat/detector_node.py OFFICE). A word can mean
# several classes ("cabinet"); exact class names always match themselves (see classes_for()).
SYNONYMS: Dict[str, List[str]] = {
    "chair": ["office chair"], "seat": ["office chair"],
    "monitor": ["computer monitor"], "screen": ["computer monitor"], "display": ["computer monitor"],
    "mouse": ["computer mouse"],
    "board": ["whiteboard"], "white board": ["whiteboard"],
    "cabinet": ["filing cabinet", "storage cabinet"], "file cabinet": ["filing cabinet"],
    "shelf": ["bookshelf"], "bookcase": ["bookshelf"], "book shelf": ["bookshelf"],
    "bin": ["trash can"], "trash": ["trash can"], "garbage can": ["trash can"],
    "rubbish bin": ["trash can"], "wastebasket": ["trash can"], "waste bin": ["trash can"],
    "trash bin": ["trash can"], "garbage bin": ["trash can"],
    "box": ["cardboard box"],
    "mug": ["coffee mug"], "cup": ["coffee mug"],
    "bottle": ["water bottle"],
    "plant": ["potted plant"],
    "phone": ["telephone"],
    "lamp": ["desk lamp"],
    "couch": ["sofa"],
    "fridge": ["refrigerator"],
    "cone": ["traffic cone"],
    "projector": ["projector screen"],
    "bag": ["backpack"],
}


@dataclass
class Obj:
    id: int
    cls: str
    xyz: Tuple[float, float, float]
    score: float
    n: int = 0
    size: Optional[Tuple[float, float, float]] = None

    @property
    def xy(self):
        return self.xyz[0], self.xyz[1]

    @property
    def radius(self) -> float:
        """Half the larger horizontal extent (0.2 m when the detector has no size yet)."""
        if not self.size:
            return 0.2
        return 0.5 * max(self.size[0], self.size[1])


@dataclass
class Waypoint:
    name: str
    x: float
    y: float
    yaw: Optional[float] = None       # None = face the direction of travel


@dataclass
class Grid:
    """2D cost grid (nav_msgs/OccupancyGrid semantics: 0-100, -1 unknown), row 0 at origin y."""
    data: np.ndarray                  # int16 H x W
    origin: Tuple[float, float]
    res: float

    def cost(self, x: float, y: float) -> int:
        c = int(math.floor((x - self.origin[0]) / self.res))
        r = int(math.floor((y - self.origin[1]) / self.res))
        if r < 0 or c < 0 or r >= self.data.shape[0] or c >= self.data.shape[1]:
            return -1
        return int(self.data[r, c])


@dataclass
class World:
    objects: List[Obj] = field(default_factory=list)
    waypoints: Dict[str, Waypoint] = field(default_factory=dict)
    pose: Optional[Tuple[float, float, float]] = None     # robot (x, y, yaw), map frame
    grid: Optional[Grid] = None
    start: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # spawn = sim origin
    low_conf: List[Obj] = field(default_factory=list)     # seen, but below min_score (never targets)
    min_score: float = 0.0

    def classes(self) -> List[str]:
        return sorted({o.cls for o in self.objects})


def norm_name(s: str) -> str:
    """Waypoint-name key: lowercase, no spaces/underscores/dashes ("Stop 3" == "stop3" == "STOP_3")."""
    return "".join(ch for ch in s.lower() if ch.isalnum())


def classes_for(noun: str, known: List[str]) -> List[str]:
    """Detector classes a noun refers to: exact class, synonym, or plural of either. The WHOLE
    phrase must match: "red chair" is not "chair" (dropping the adjective would pick a wrong
    target), so anything with extra words is left to a later tier."""
    noun = " ".join(noun.strip().lower().split())
    forms = [noun]
    if noun.endswith("es"):
        forms.append(noun[:-2])
    if noun.endswith("s"):
        forms.append(noun[:-1])
    for f in forms:
        if f in known:
            return [f]
        if f in SYNONYMS:
            return list(SYNONYMS[f])
    return []


def objects_from_json(text: str, min_score: float = 0.0) -> List[Obj]:
    """/detections/objects payload ({"frame": [...], "objects": [...confirmed...]}) -> objects."""
    data = json.loads(text)
    out = []
    for o in data.get("objects", []):
        if o.get("score", 0.0) < min_score:
            continue
        size = o.get("size")
        out.append(Obj(id=int(o["id"]), cls=o["cls"], xyz=tuple(o["xyz"]), score=float(o["score"]),
                       n=int(o.get("n", 0)), size=tuple(size) if size else None))
    return out


def waypoints_from_scene(scene: dict) -> Dict[str, Waypoint]:
    """Scene yaml `waypoints:` {name: [x, y] | [x, y, yaw]} -> {norm_name: Waypoint}."""
    out = {}
    for name, v in (scene.get("waypoints") or {}).items():
        name = str(name)
        out[norm_name(name)] = Waypoint(name, float(v[0]), float(v[1]),
                                        float(v[2]) if len(v) > 2 else None)
    return out


def grid_from_msg(width: int, height: int, res: float, origin_xy, data) -> Grid:
    return Grid(np.asarray(data, dtype=np.int16).reshape(height, width), tuple(origin_xy), float(res))


# Nav2 raw costs (0-255: 253 inscribed, 254 lethal, 255 unknown) -> OccupancyGrid scale (0-100,
# -1 unknown), the same table Nav2 uses to publish its costmap topic (Costmap2DPublisher).
_COST_TO_OCC = np.empty(256, np.int16)
_COST_TO_OCC[0] = 0
_COST_TO_OCC[1:253] = 1 + (97 * (np.arange(1, 253) - 1)) // 251
_COST_TO_OCC[253], _COST_TO_OCC[254], _COST_TO_OCC[255] = 99, 100, -1


def grid_from_costmap(size_x: int, size_y: int, res: float, origin_xy, data) -> Grid:
    """nav2_msgs/Costmap (raw 0-255 costs, e.g. from the get_costmap service) -> Grid."""
    raw = np.frombuffer(bytes(data), np.uint8).reshape(size_y, size_x)
    return Grid(_COST_TO_OCC[raw], tuple(origin_xy), float(res))
