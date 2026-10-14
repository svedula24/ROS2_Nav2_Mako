#!/usr/bin/env python3
"""Host-side finish of an eval run: join the NL node's `timing:` log lines (llm / costmap / resolve ms) to the
attempts by time window, then print the table.   python3 finish_run.py run.json nlnav_timing.log"""
import json
import re
import statistics
import sys

run = json.load(open(sys.argv[1]))
lines = []
if len(sys.argv) > 2:
    for l in open(sys.argv[2], errors="replace"):
        m = re.search(r"\[(\d+\.\d+)\].*timing: llm (\d+) ms, costmap (\d+) ms, resolve\+reach (\d+) ms", l)
        if m:
            lines.append((float(m.group(1)), {"llm_ms": int(m.group(2)), "costmap_ms": int(m.group(3)), "resolve_ms": int(m.group(4))}))
for c in run["cases"]:
    for a in c["attempts"]:
        if a.get("status") in ("not_understood", "stopped", "error", "rejected"):
            continue                                  # these paths return before the timing line is logged
        hits = [t for ts, t in lines if a["t_start"] - 0.2 <= ts <= a["t_end"] + 0.2]
        if hits:
            a["timing"] = hits[-1]
json.dump(run, open(sys.argv[1], "w"), indent=1)


def med(xs):
    xs = [x for x in xs if x is not None]
    return f"{statistics.median(xs):.0f}" if xs else "-"


def verdict(c):
    ok = all(a["pass"] for a in c["attempts"])
    if c.get("known_issue"):
        return "FIXED?" if ok else "KNOWN"          # a known issue that now passes deserves a look (and the key removed)
    return "PASS" if ok else "FAIL"


print(f"\n{run['tag']}  ({run['ts']}, {'drive' if run.get('drive') else 'plan'}-level, config {run.get('config')})")
print(f"{'case':<32} {'result':<6} {'status':<14} {'total ms':>9} {'llm ms':>7} {'resolve':>8}  note")
for c in run["cases"]:
    att = c["attempts"]
    print(f"{c['name']:<32} {verdict(c):<6} {att[0]['status']:<14} {med([a.get('latency_ms') for a in att]):>9} "
          f"{med([(a.get('timing') or {}).get('llm_ms') for a in att]):>7} {med([(a.get('timing') or {}).get('resolve_ms') for a in att]):>8}  {att[-1]['why'][:60]}")
    loc = [a.get("loc_err_max_m") for a in att if a.get("loc_err_max_m") is not None]
    if loc:
        print(f"{'':<32} localization: max error {max(loc):.2f} m, yaw {max(a.get('loc_yaw_err_max_deg') or 0 for a in att):.1f} deg "
              f"(believed pose vs ground truth)")
v = [verdict(c) for c in run["cases"]]
known = [c for c in run["cases"] if c.get("known_issue")]
print(f"{v.count('PASS')}/{len(v) - len(known)} pass (excluding known issues); NEW FAILURES: {v.count('FAIL')}; "
      f"known issues: {v.count('KNOWN')} still failing, {v.count('FIXED?')} now passing")
print(f"median total {med([a.get('latency_ms') for c in run['cases'] for a in c['attempts']])} ms, "
      f"llm {med([(a.get('timing') or {}).get('llm_ms') for c in run['cases'] for a in c['attempts']])} ms")
