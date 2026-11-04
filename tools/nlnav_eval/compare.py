#!/usr/bin/env python3
"""Compare eval runs side by side.   python3 tools/nlnav_eval/compare.py run_a.json [run_b.json ...]
Per case: pass/fail per run and the median total latency; overall: pass rate, median total / LLM / resolve ms,
deltas vs the first run. Drive runs also get each case's localization error and raw odometry drift."""
import json
import statistics
import sys


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def summarize(run):
    out = {}
    for c in run["cases"]:
        att = c["attempts"]
        out[c["name"]] = {
            "pass": all(a["pass"] for a in att) if att else False,
            "status": att[0]["status"] if att else "-",
            "total": med([a.get("latency_ms") for a in att]),
            "llm": med([(a.get("timing") or {}).get("llm_ms") for a in att]),
            "resolve": med([(a.get("timing") or {}).get("resolve_ms") for a in att]),
            "why": att[-1]["why"] if att else "",
            "loc": max([a["loc_err_max_m"] for a in att if a.get("loc_err_max_m") is not None], default=None),
            "odom": max([a["odom_err_max_m"] for a in att if a.get("odom_err_max_m") is not None], default=None),
        }
    return out


def fmt(v, unit=""):
    return "-" if v is None else f"{v:.0f}{unit}"


def fmt2(v):
    return "-" if v is None else f"{v:.2f}"


def main():
    runs = [json.load(open(p)) for p in sys.argv[1:]]
    if not runs:
        print(__doc__); sys.exit(2)
    sums = [summarize(r) for r in runs]
    tags = [r.get("tag", "?") for r in runs]
    print("run        | " + " | ".join(f"{t:>14}" for t in tags))
    print("config     | " + " | ".join(f"{str((r.get('config') or {}).get('model') or '?')[:14]:>14}" for r in runs))
    print("localiz.   | " + " | ".join(f"{(r.get('config') or {}).get('localization', 'gt'):>14}" for r in runs))
    names = [c["name"] for c in runs[0]["cases"]]
    print("-" * (13 + 17 * len(runs)))
    for n in names:
        cells = []
        for s in sums:
            c = s.get(n)
            cells.append("     (absent)  " if not c else f"{'PASS' if c['pass'] else 'FAIL':>4} {fmt(c['total'], 'ms'):>9}")
        print(f"{n[:10]:<10} | " + " | ".join(cells))
    print("-" * (13 + 17 * len(runs)))
    for label, key in (("pass rate", None), ("median total", "total"), ("median llm", "llm"), ("median resolve", "resolve")):
        cells = []
        for s in sums:
            if key is None:
                n_pass = sum(c["pass"] for c in s.values()); cells.append(f"{n_pass}/{len(s)}")
            else:
                cells.append(fmt(med([c[key] for c in s.values()]), " ms"))
        print(f"{label:<10} | " + " | ".join(f"{c:>14}" for c in cells))
    loc_cases = [n for n in names if any((s.get(n) or {}).get("loc") is not None for s in sums)]
    if loc_cases:                             # drive runs: believed-pose error and raw odometry drift, metres
        print("\nlocalization: max error believed vs true / raw odometry drift (m)")
        for n in loc_cases:
            cells = [f"{fmt2((s.get(n) or {}).get('loc'))} / {fmt2((s.get(n) or {}).get('odom'))}" for s in sums]
            print(f"{n[:10]:<10} | " + " | ".join(f"{c:>14}" for c in cells))
    if len(runs) > 1:
        print("\nchanges vs", tags[0])
        for i, s in enumerate(sums[1:], 1):
            for n in names:
                a, b = sums[0].get(n), s.get(n)
                if a and b and a["pass"] != b["pass"]:
                    print(f"  {tags[i]}: {n}: {'PASS' if a['pass'] else 'FAIL'} -> {'PASS' if b['pass'] else 'FAIL'}  ({b['why']})")
            ta, tb = med([c["total"] for c in sums[0].values()]), med([c["total"] for c in s.values()])
            if ta and tb:
                print(f"  {tags[i]}: median total {ta:.0f} -> {tb:.0f} ms ({(tb - ta) / ta * 100:+.0f}%)")


if __name__ == "__main__":
    main()
