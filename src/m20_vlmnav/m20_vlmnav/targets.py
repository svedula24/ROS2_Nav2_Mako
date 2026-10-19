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
