"""What counts as the search target among the detector's labels (pure: no ROS)."""
# The detector labels a box with the best of ITS vocabulary; a search phrase may be a synonym of one of those labels
SYNONYMS = {"refrigerator": {"fridge"}, "trash can": {"bin", "garbage can", "wastebasket", "trash bin"},
            "traffic cone": {"cone"}, "computer monitor": {"monitor", "screen"}, "office chair": {"chair"},
            "potted plant": {"plant"}, "cardboard box": {"box"}, "coffee mug": {"mug", "cup"},
            "water bottle": {"bottle"}, "sofa": {"couch"}, "telephone": {"phone"}}

def same_class(cls: str, target: str) -> bool:
    """Does a detector label name the search target? Exact, or a known synonym either way."""
    if cls == target:
        return True
    return target in SYNONYMS.get(cls, ()) or cls in SYNONYMS.get(target, ())


import re

_LEAD = re.compile(r"^(?:please\s+)?(?:go|navigate|walk|drive|head|get)\s+(?:over\s+)?to\s+(?:the|a|an)\s+|"
                   r"^(?:please\s+)?(?:find|locate|look\s+for|search\s+for)\s+(?:the|a|an)\s+", re.I)
_TRAIL = re.compile(r"(?:\s*,?\s*(?:and\s+)?(?:then\s+)?stop(?:\s+(?:in\s+front\s+of|next\s+to|near|by)\s+it)?)\s*\.?$|"
                    r"(?:\s*,?\s*please)\s*\.?$", re.I)


def phrase_from(text: str) -> str:
    """The object phrase a search looks for, out of an instruction: "go to the ladder" -> "ladder", "find a trash
    can and stop in front of it" -> "trash can". An instruction with no such shape ("turn left 90 degrees", a route)
    comes back whole, lower-cased: the search then looks for a thing that is not one and fails honestly."""
    t = " ".join(text.strip().lower().split())
    t = _TRAIL.sub("", t)
    t = _LEAD.sub("", t)
    return t.strip(" .,")
