"""Pass/fail judgement for one eval attempt against a case's `expect` (pure python; shared by run_eval.py inside the
ROS container and by the unit tests). An attempt is a plain dict:
  {"status": str, "items": [{"label": str, "id": int, "x": float, "y": float}], ...}
and for drive cases also {"final_xy": [x, y], "max_tilt_rad": float, "max_footprint_age_s": float,
  "stop_drift_m": float}. Returns (passed, why)."""
import math


def _norm(label):
    return (label or "").split(" #")[0].strip().lower()


def judge(expect: dict, attempt: dict, objects: dict = None, localization: str = None):
    """objects: {id: (x, y)} map positions for drive-level `near_id` checks.
    localization: the run's localization mode (eval.sh config): gt | amcl | odom. Under "odom" (drifting odometry, no
    correction: the baseline that shows what drift does) the localization limits are recorded, not judged."""
    ok, why = _judge(expect, attempt, objects, localization)
    spl = attempt.pop("_spl_if_success", None)
    if spl is not None:                                     # SPL = success x l / max(p, l): 0 for a failed attempt
        attempt["spl"] = spl if ok else 0.0
    return ok, why


def _judge(expect, attempt, objects, localization):
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
            ok_ids = set(want) if isinstance(want, (list, tuple)) else {want}   # a list = equally right answers
            if k >= len(items):
                return False, f"step {k} missing (wanted id {want})"
            if items[k].get("id") not in ok_ids:
                return False, f"step {k} id {items[k].get('id')} not in {sorted(ok_ids)}"
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
        elif "near_id" in expect:                               # one id, or a list = any of them (the nearest counts)
            ids = expect["near_id"] if isinstance(expect["near_id"], (list, tuple)) else [expect["near_id"]]
            known = [objects[i] for i in ids if objects and i in objects]
            if not known:
                return False, f"object {expect['near_id']} position unknown"
            tx, ty = min(known, key=lambda o: math.hypot(fx - o[0], fy - o[1]))
        elif items:
            tx, ty = items[-1]["x"], items[-1]["y"]
        else:
            return False, "nothing to measure arrival against"
        d = math.hypot(fx - tx, fy - ty)
        if d > expect["near_m"]:                                # sanity bound: a wrong approach pose can't pass
            return False, f"ended {d:.2f} m from the target (limit {expect['near_m']})"
        why.append(f"{d:.2f} m from target")
        # straight-line SPL (success x shortest / actual path, with the straight line standing in for the shortest
        # path, so it's pessimistic); recorded, never judged. Finalised in judge(): 0 unless every check passes.
        if attempt.get("path_len_m") and attempt.get("start_xy"):
            sx, sy = attempt["start_xy"]
            straight = math.hypot(tx - sx, ty - sy)
            attempt["_spl_if_success"] = round(straight / max(attempt["path_len_m"], straight), 3)
        if "facing_deg" in expect:                              # Real-time B: "stops near the object, facing it"
            yaw = attempt.get("final_yaw")
            if yaw is None:
                return False, "no final heading"
            off = math.degrees(abs(math.remainder(math.atan2(ty - fy, tx - fx) - yaw, 2 * math.pi)))
            if off > expect["facing_deg"]:
                return False, f"not facing the target: off by {off:.0f} deg (limit {expect['facing_deg']})"
            why.append(f"facing it ({off:.0f} deg off)")
    if "final_yaw_deg" in expect:                               # e.g. "turn left 90 degrees": heading, not position
        yaw = attempt.get("final_yaw")
        if yaw is None:
            return False, "no final heading"
        off = math.degrees(abs(math.remainder(math.radians(expect["final_yaw_deg"]) - yaw, 2 * math.pi)))
        if off > expect.get("yaw_tol_deg", 20):
            return False, f"heading {math.degrees(yaw):.0f} deg, expected {expect['final_yaw_deg']} (off by {off:.0f})"
        why.append(f"heading within {off:.0f} deg")
    if attempt.get("max_tilt_rad") is not None and attempt["max_tilt_rad"] > 0.3:
        return False, f"tilted {attempt['max_tilt_rad']:.2f} rad during the drive"
    if attempt.get("max_footprint_age_s") is not None and attempt["max_footprint_age_s"] > 3.0:
        return False, f"nav2 pose froze (footprint age {attempt['max_footprint_age_s']:.1f} s)"
    if attempt.get("failsafe"):
        return False, f"commander failsafe: {attempt['failsafe']}"
    if "max_loc_err_m" in expect or "max_loc_yaw_deg" in expect:   # localization: believed pose vs ground truth
        err, yaw = attempt.get("loc_err_max_m"), attempt.get("loc_yaw_err_max_deg")
        if err is None:
            return False, "no localization samples (TF map->base_link vs /odom_true)"
        if localization == "odom":
            why.append(f"uncorrected odometry: believed pose off by up to {err:.2f} m / {yaw or 0:.0f} deg (not judged)")
        else:
            if err > expect.get("max_loc_err_m", math.inf):
                return False, f"localization off by {err:.2f} m (limit {expect['max_loc_err_m']})"
            if yaw is not None and yaw > expect.get("max_loc_yaw_deg", math.inf):
                return False, f"localization heading off by {yaw:.1f} deg (limit {expect['max_loc_yaw_deg']})"
            why.append(f"localized within {err:.2f} m / {yaw or 0:.1f} deg")
    return True, "; ".join(why) or "ok"
