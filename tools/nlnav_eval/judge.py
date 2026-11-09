"""Pass/fail judgement for one eval attempt against a case's `expect` (pure python; shared by run_eval.py inside the
ROS container and by the unit tests). An attempt is a plain dict:
  {"status": str, "items": [{"label": str, "id": int, "x": float, "y": float}], ...}
and for drive cases also {"final_xy": [x, y], "max_tilt_rad": float, "max_footprint_age_s": float,
  "stop_drift_m": float}. Returns (passed, why).
near_id as a list (2026-10-19): the attempt passes if ANY listed id meets near_m (and facing_deg); the reference used
to be the NEAREST listed id, which failed a search that had approached another listed object. A single id is judged
exactly as before."""
import math


def _norm(label):
    return (label or "").split(" #")[0].strip().lower()


def facing_off_deg(fx, fy, yaw, tx, ty):
    """How far (degrees, >= 0) the heading `yaw` at (fx, fy) is off the direction to (tx, ty)."""
    return math.degrees(abs(math.remainder(math.atan2(ty - fy, tx - fx) - yaw, 2 * math.pi)))


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
    if "legs" in attempt:                                       # a waypoint walk: every leg ends near its goal
        tol = expect.get("leg_tol_m", 0.45)
        far = [l for l in attempt["legs"] if l["err_m"] > tol]
        if st == "done" and far:
            return False, "legs off: " + ", ".join(f"{l['label']} {l['err_m']:.2f} m" for l in far) + f" (tol {tol})"
        if attempt["legs"]:
            why.append("legs " + ", ".join(f"{l['err_m']:.2f} m/{l['t_s']:.0f} s" for l in attempt["legs"]))
        if "min_scan_m" in expect:                              # closest the lidar saw anything: keep-away held
            seen = attempt.get("min_scan_m")
            if seen is None:
                return False, "no scan samples"
            if seen < expect["min_scan_m"]:
                return False, f"came within {seen:.2f} m of an obstacle (limit {expect['min_scan_m']})"
            why.append(f"closest obstacle {seen:.2f} m")
        if "max_map_odom_step_m" in expect and localization != "gt":
            step = attempt.get("map_odom_step_max_m")
            if step is None:
                return False, "no map->odom samples"
            if step > expect["max_map_odom_step_m"]:
                return False, f"map->odom jumped {step:.3f} m (limit {expect['max_map_odom_step_m']})"
            why.append(f"map->odom steps <= {step:.3f} m")
    if isinstance(expect.get("max_loc_err_m"), dict):          # a limit per localization mode (vendor vs amcl)
        expect = dict(expect, max_loc_err_m=expect["max_loc_err_m"].get(localization, math.inf))
    if "cancel_after_s" in expect:                              # a mid-leg cancel: a bounded roll-out, then still
        expect = dict(expect, stop_after_s=expect["cancel_after_s"])
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
            # one id, or a list = any of them is right. Judge against the listed one the robot actually ended at:
            # the first that meets near_m (and facing_deg when asked); if none does, the nearest, so the report
            # says how far off it was. (Before 2026-10-19 the NEAREST listed id was always the reference, so a
            # search that approached listed fridge #223 was measured against #216 behind it and failed "facing".)
            ids = expect["near_id"] if isinstance(expect["near_id"], (list, tuple)) else [expect["near_id"]]
            known = [objects[i] for i in ids if objects and i in objects]
            if not known:
                return False, f"object {expect['near_id']} position unknown"
            yaw0 = attempt.get("final_yaw")
            def _ok(o):
                if math.hypot(fx - o[0], fy - o[1]) > expect["near_m"]:
                    return False
                if "facing_deg" in expect and yaw0 is not None:
                    return facing_off_deg(fx, fy, yaw0, o[0], o[1]) <= expect["facing_deg"]
                return True
            good = [o for o in known if _ok(o)]
            tx, ty = good[0] if good else min(known, key=lambda o: math.hypot(fx - o[0], fy - o[1]))
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
            off = facing_off_deg(fx, fy, yaw, tx, ty)
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
