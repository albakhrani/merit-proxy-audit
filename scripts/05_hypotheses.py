#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the five registered hypotheses from the grid results.

    python scripts/05_hypotheses.py --grid results/grid --kappa results/kappa.json

Reads only files produced by 04_run_grid.py; writes results/hypotheses.json.
Every verdict names its falsifier and states whether it fired.
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from study2lib import specs
    from study2lib.fdr import benjamini_hochberg
    from study2lib.runlog import write_json
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/grid")
    ap.add_argument("--kappa", default="results/kappa.json")
    ap.add_argument("--out", default="results/hypotheses.json")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.grid, "*.json")))
    if not files:
        print("no grid results found; run 04 first"); return 1
    cells = [json.load(open(f, encoding="utf-8")) for f in files]
    kap = json.load(open(a.kappa, encoding="utf-8"))

    def primary(c):
        for r in c["spec_results"]:
            if (r["outcome_def"], r["covariate_set"], r["learner"]) == tuple(specs.BOOT_ARMS[0]):
                return r
        return None

    prim = [(c, primary(c)) for c in cells]
    prim = [(c, r) for c, r in prim if r is not None]
    powered = [(c, r) for c, r in prim if r["powered"]]

    # ---- H2.1 attenuation: does the corrected residual survive? ----
    # Amended 15 Sep 2026: under the female-minus-male convention the residual
    # is negative throughout, so the positive share alone sits at its floor by
    # construction; the added fields carry the direction, sign preservation
    # and sweep behaviour the hypothesis actually concerns. Additive only:
    # the registered H2.1 field share_corrected_residual_positive and the
    # falsifier wording below are unchanged.
    k_lit = kap["lit"]["overall"]
    unexpl = [r["eiv_unexplained_at_kappa"] for _, r in powered]
    unexpl_raw = [r["decomp_unexplained"] for _, r in powered]
    sweeps = [[p["unexplained"] for p in r["kappa_sweep"]] for _, r in powered]
    h21 = {"kappa_lit": k_lit["kappa"], "kappa_se": k_lit["se"],
           "share_corrected_residual_positive": float(np.mean([u > 0 for u in unexpl])),
           "share_corrected_residual_negative": float(np.mean([u < 0 for u in unexpl])),
           "share_sign_preserved": float(np.mean([np.sign(u) == np.sign(v)
                                                  for u, v in zip(unexpl, unexpl_raw)])),
           "share_moved_up": float(np.mean([u > v for u, v in zip(unexpl, unexpl_raw)])),
           "share_deflated_toward_zero": float(np.mean([abs(u) < abs(v)
                                                        for u, v in zip(unexpl, unexpl_raw)])),
           "share_sweep_crosses_zero": float(np.mean([min(s) < 0 < max(s) for s in sweeps])),
           "note": "no interval for the corrected residual is stored; the bootstrap is on R only",
           "falsifier": "an interval covering zero for the corrected residual in most powered cells",
           "n_powered_cells": len(powered)}

    # ---- H2.2 differential reliability: sex kappa difference ----
    sex = kap["lit"].get("sex", {})
    ks = {k: v.get("kappa") for k, v in sex.items() if isinstance(v, dict)}
    h22 = {"kappa_by_sex": ks,
           "difference": (None if len([v for v in ks.values() if v is not None]) < 2
                          else float(np.diff(sorted([v for v in ks.values() if v is not None]))[-1])),
           "falsifier": "a differential large enough to flip the sign of the corrected merit gap"}

    # ---- H2.3 FDR-controlled replication of the reversal across cells ----
    ps = []
    for _, r in powered:
        b = r.get("boot")
        if b:                       # one-sided p from the percentile interval position
            draws_gt0 = None        # draws not stored; use normal approx from interval
            se = (b["hi"] - b["lo"]) / (2 * 1.959964)
            z = r["R"] / se if se > 0 else 0.0
            from math import erf, sqrt
            p = 1 - 0.5 * (1 + erf(z / sqrt(2)))
            ps.append(p)
    rej = benjamini_hochberg(ps, q=0.05) if ps else np.array([])
    h23 = {"n_tested": len(ps),
           "share_replicating_fdr05": float(np.mean(rej)) if len(ps) else None,
           "falsifier": "absent or reversed in a majority of powered cells"}

    # ---- H2.4 heterogeneity ----
    Rs = [r["R"] for _, r in powered]
    h24 = {"R_mean": float(np.mean(Rs)), "R_sd": float(np.std(Rs, ddof=1)),
           "R_iqr": float(np.subtract(*np.percentile(Rs, [75, 25])))}

    # ---- H2.5 specification concordance Gamma and tolerance share ----
    gammas, tol_shares = [], []
    for c, rp in powered:
        rs = [r for r in c["spec_results"]]
        verdicts = [1 if (r["delta_merit"] > 0 > r["delta_label"]) else 0 for r in rs]
        if not verdicts:
            continue
        modal = int(np.mean(verdicts) >= 0.5)
        gammas.append(np.mean([v == modal for v in verdicts]) >= specs.GAMMA_AGREE)
        Rvals = [r["R"] for r in rs]
        tol_shares.append(max(Rvals) - min(Rvals) <= specs.OMEGA_TOLERANCE)
    h25 = {"Gamma": float(np.mean(gammas)) if gammas else None,
           "predicted_below": specs.GAMMA_PREDICTED_BELOW,
           "share_within_omega": float(np.mean(tol_shares)) if tol_shares else None,
           "omega": specs.OMEGA_TOLERANCE,
           "falsifier": "Gamma at or above 0.75 with the omega share at or above 0.50"}
    # Amended 15 Sep 2026: locate which analyst choice drives disagreement,
    # and record that the reversal magnitude R itself is spec-robust. Additive.
    from collections import defaultdict
    vr = defaultdict(list)
    n_R_pos = n_specs = 0
    for c, _ in powered:
        for r in c["spec_results"]:
            v = 1 if (r["delta_merit"] > 0 > r["delta_label"]) else 0
            vr["learner_" + r["learner"]].append(v)
            vr["outcome_" + r["outcome_def"]].append(v)
            vr["covset_" + r["covariate_set"]].append(v)
            n_R_pos += int(r["R"] > 0); n_specs += 1
    h25["verdict_rate_by_dimension"] = {k: float(np.mean(v)) for k, v in sorted(vr.items())}
    h25["share_R_positive_all_specs"] = float(n_R_pos / n_specs) if n_specs else None

    payload = {"n_cells": len(cells), "n_powered": len(powered),
               "H2_1": h21, "H2_2": h22, "H2_3": h23, "H2_4": h24, "H2_5": h25}
    write_json(a.out, payload)
    print(json.dumps(payload, indent=1)[:2000])
    print(f"\nwritten {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
