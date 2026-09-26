#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Time three representative cells and extrapolate the full run before
committing to it. Budget by measurement, not by guess.

    python scripts/00_pilot.py --root acs_data --kappa results/kappa.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PILOT = [("WY", 2022), ("CO", 2022), ("CA", 2022)]   # small, medium, largest


def main():
    from study2lib.cellrun import run_cell
    from study2lib.acs import STATE_FIPS, load_cell
    from study2lib import specs
    from study2lib.runlog import write_json
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="acs_data")
    ap.add_argument("--kappa", default=None, help="results/kappa.json; 0.75 assumed for timing if absent")
    a = ap.parse_args()
    kap = 0.75
    if a.kappa and os.path.exists(a.kappa):
        kap = json.load(open(a.kappa, encoding="utf-8"))["lit"]["overall"]["kappa"]

    rows_per_s, records = [], []
    for st, yr in PILOT:
        t0 = time.time()
        res = run_cell(a.root, st, yr, kap)
        secs = time.time() - t0
        n = max(r["n"] for r in res["spec_results"])
        rows_per_s.append(n / secs)
        records.append({"state": st, "year": yr, "n": n, "seconds": round(secs, 1),
                        "skipped_arms": res["skipped_arms"]})
        print(f"  {st} {yr}: n={n:,}  {secs:6.1f}s"
              + (f"  skipped: {[s['learner'] for s in res['skipped_arms'][:3]]}" if res["skipped_arms"] else ""))

    # extrapolate: total rows in the priority grid approx 3.2M per year x 5 years
    per_row = 1.0 / (sum(rows_per_s) / len(rows_per_s))
    total_rows = 3_200_000 * len(specs.PRIORITY_YEARS)
    serial_h = total_rows * per_row / 3600
    workers = max(2, (os.cpu_count() or 8) - 2)
    est = {"pilot": records, "serial_hours_est": round(serial_h, 1),
           "workers": workers, "wall_hours_est": round(serial_h / workers, 1)}
    write_json("results/pilot.json", est)
    print(f"\nestimated wall time for the priority grid with {workers} workers: "
          f"~{est['wall_hours_est']} h (serial {est['serial_hours_est']} h)")
    print("written results/pilot.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
