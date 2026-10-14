"""Pass/fail judgement for one eval attempt against a case's `expect` (pure python; shared by run_eval.py inside the
ROS container and by the unit tests). An attempt is a plain dict:
  {"status": str, "items": [{"label": str, "id": int, "x": float, "y": float}], ...}
and for drive cases also {"final_xy": [x, y], "max_tilt_rad": float, "max_footprint_age_s": float,
  "stop_drift_m": float}. Returns (passed, why)."""
import math


def _norm(label):
    return (label or "").split(" #")[0].strip().lower()


def judge(expect: dict, attempt: dict, objects: dict = None):
    """objects: {id: (x, y)} map positions for drive-level `near_id` checks."""
    why = []
    st = attempt.get("status")
    if st != expect.get("status"):
        return False, f"status {st!r} != {expect.get('status')!r}"
    items = attempt.get("items") or []
    if "labels" in expect:
        got = [_norm(i.get("label")) for i in items]
        want = [l.lower() for l in expect["labels"]]
        if got != want:
            return False, f"labels {got} != {want}"
    if "ids" in expect:
        for k, want in enumerate(expect["ids"]):
            if want in (None, -1):
                continue
            if k >= len(items):
                return False, f"step {k} missing (wanted id {want})"
            if items[k].get("id") != want:
                return False, f"step {k} id {items[k].get('id')} != {want}"
    if "goal_xy" in expect:
        if not items:
            return False, "no goal to check"
        gx, gy = expect["goal_xy"]
        d = math.hypot(items[0]["x"] - gx, items[0]["y"] - gy)
        if d > expect.get("tol", 0.5):
            return False, f"goal {d:.2f} m from expected (tol {expect.get('tol', 0.5)})"
        why.append(f"goal within {d:.2f} m")
    # ---- drive-level ----
    if "stop_after_s" in expect:
        # "stops": a bounded roll-out after the command (pipeline latency + braking: measured 0.27 m at 0.5 m/s)
        # and no motion in the second after that. The roll-out is recorded; the limit is generous on purpose.
        drift, settle = attempt.get("stop_drift_m"), attempt.get("settle_m")
        if drift is None or drift > expect.get("max_rollout_m", 0.5):
            return False, f"rolled {drift} m after stop (limit {expect.get('max_rollout_m', 0.5)})"
        if settle is None or settle > 0.05:
            return False, f"still moving {settle} m/s after the roll-out"
        why.append(f"stopped: rolled {drift:.2f} m, then still")
    if "near_m" in expect and "stop_after_s" not in expect:
        fx, fy = attempt.get("final_xy", (None, None))
        if fx is None:
            return False, "no final pose"
        if items:                                               # arrival = at the approach pose the planner chose
            ax, ay = items[-1]["x"], items[-1]["y"]
            da = math.hypot(fx - ax, fy - ay)
            if da > expect.get("approach_tol_m", 0.6):
                return False, f"ended {da:.2f} m from the approach pose (tol {expect.get('approach_tol_m', 0.6)})"
            why.append(f"at approach pose ({da:.2f} m)")
        if "near_xy" in expect:
            tx, ty = expect["near_xy"]
        elif "near_id" in expect:
            if not objects or expect["near_id"] not in objects:
                return False, f"object {expect['near_id']} position unknown"
            tx, ty = objects[expect["near_id"]]
        elif items:
            tx, ty = items[-1]["x"], items[-1]["y"]
        else:
            return False, "nothing to measure arrival against"
        d = math.hypot(fx - tx, fy - ty)
        if d > expect["near_m"]:                                # sanity bound: a wrong approach pose can't pass
            return False, f"ended {d:.2f} m from the target (limit {expect['near_m']})"
        why.append(f"{d:.2f} m from target")
    if attempt.get("max_tilt_rad") is not None and attempt["max_tilt_rad"] > 0.3:
        return False, f"tilted {attempt['max_tilt_rad']:.2f} rad during the drive"
    if attempt.get("max_footprint_age_s") is not None and attempt["max_footprint_age_s"] > 3.0:
        return False, f"nav2 pose froze (footprint age {attempt['max_footprint_age_s']:.1f} s)"
    if attempt.get("failsafe"):
        return False, f"commander failsafe: {attempt['failsafe']}"
    return True, "; ".join(why) or "ok"
