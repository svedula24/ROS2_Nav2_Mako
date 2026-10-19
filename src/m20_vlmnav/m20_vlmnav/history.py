"""Which camera frames the VLM sees: the current one plus 7 evenly spaced earlier ones since the instruction started
(NaVILA's sampling; the history is the model's only memory of where it has been)."""
from typing import List

N_FRAMES = 8


def pick(n_available: int, k: int = N_FRAMES) -> List[int]:
    """Indices into the frames recorded since the instruction started (0 = first, n-1 = current), oldest first.
    Always k long and always ends with the current frame. Fewer than k frames: the oldest is repeated at the front,
    so the model always gets the same number of images."""
    if n_available <= 0:
        return []
    if n_available >= k:
        # k points evenly spread over [0, n-1], both ends included
        return [round(i * (n_available - 1) / (k - 1)) for i in range(k)]
    return [0] * (k - n_available) + list(range(n_available))
