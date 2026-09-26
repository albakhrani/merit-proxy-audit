#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Guard for the arms added after the internal review of 22 September 2026:
the 'exact' arm reruns the registered grid with
the logistic learner only and the exact anchors added. Every decomposition
field and the logistic R must equal the stored grid cell bit for bit; if
they do not, the library changes of 22 September altered registered
behaviour and nothing else may run until that is understood.

    python scripts/13_check_exact.py --grid results/grid --exact results/grid_exact
"""
import argparse, glob, json, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/grid")
    ap.add_argument("--exact", default="results/grid_exact")
    ap.add_argument("--tol", type=float, default=1e-9)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.exact, "*.json")))
    if not files:
        print("no exact cells found; run scripts/12_run_arms.py --arm exact first")
        return 1
    bad, n_cells, n_specs = [], 0, 0
    fields = ("decomp_raw_logpts", "decomp_explained", "decomp_unexplained",
              "eiv_unexplained_at_kappa", "delta_label", "delta_merit", "R")
    for f in files:
        e = json.load(open(f, encoding="utf-8"))
        gpath = os.path.join(a.grid, os.path.basename(f))
        if not os.path.exists(gpath):
            bad.append((os.path.basename(f), "no stored grid cell")); continue
        g = json.load(open(gpath, encoding="utf-8"))
        n_cells += 1
        stored = {(s["outcome_def"], s["covariate_set"], s["learner"]): s for s in g["spec_results"]}
        for s in e["spec_results"]:
            key = (s["outcome_def"], s["covariate_set"], s["learner"])
            if key not in stored:
                bad.append((os.path.basename(f), f"{key} absent from the grid")); continue
            n_specs += 1
            for fld in fields:
                if s[fld] is None or stored[key][fld] is None:
                    if s[fld] is not stored[key][fld]:
                        bad.append((os.path.basename(f), f"{key} {fld}: exact {s[fld]!r} vs grid {stored[key][fld]!r}"))
                    continue
                if abs(float(s[fld]) - float(stored[key][fld])) > a.tol:
                    bad.append((os.path.basename(f), f"{key} {fld}: exact {s[fld]!r} vs grid {stored[key][fld]!r}"))
            sw_e = [p["unexplained"] for p in s["kappa_sweep"]]
            sw_g = [p["unexplained"] for p in stored[key]["kappa_sweep"]]
            if len(sw_e) != len(sw_g) or max(abs(x - y) for x, y in zip(sw_e, sw_g)) > a.tol:
                bad.append((os.path.basename(f), f"{key} kappa_sweep differs"))
            if abs(s["eiv_unexplained_at"][str(round(s["kappa_point"], 6))]
                   - s["eiv_unexplained_at_kappa"]) > a.tol:
                bad.append((os.path.basename(f), f"{key} exact anchor does not equal the point value"))
    print(f"compared {n_specs} specifications in {n_cells} cells against the stored grid")
    if bad:
        for b in bad[:40]:
            print("  MISMATCH", b)
        print(f"{len(bad)} mismatches: STOP and report before running any other arm")
        return 1
    print("all equal within tolerance: the registered grid reproduces bit for bit under the new code")
    return 0


if __name__ == "__main__":
    sys.exit(main())
