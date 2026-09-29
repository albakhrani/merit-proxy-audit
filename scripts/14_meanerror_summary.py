#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarise the mean-error arms against the years-coded arm
(REGISTERED_CHANGES R11, 22 Sep 2026, post-results).

For every market and covariate set, three unexplained components are read
from the stored cells:

    U_y(1)    years-coded arm, no correction (kappa = 1)
    U_y(k)    years-coded arm, reliability correction at the anchor
    U_m(1)    mean-error arm, women's schooling lowered by delta, kappa = 1
    U_m(k)    mean-error arm, both corrections

and the sweep of each arm. The script reports, per delta: how many markets
the reliability correction deepens or attenuates once the mean error is
removed, the sign of U_m at the anchor and at the sweep floor, and pooled
means with a state-block bootstrap (states drawn with replacement, all
years of a state kept together, seeded from the primary seed). Nothing is
re-estimated. Writes results/meanerror_summary.json.

    python scripts/14_meanerror_summary.py --results results --tags lit lit_lo lit_hi num

Added 28 Sep 2026 (REGISTERED_CHANGES R14 and R15): a tag names either
results/grid_meanerror_<tag> or, when that folder does not exist,
results/grid_<tag>, so the proportional arms (grid_prop_lit and so on) are
summarised with the same quantities. For a proportional arm delta_years is
None, the share is recorded, and the per-market applied shift (from each
cell's load_options) is pooled as applied_shift_years:

    python scripts/14_meanerror_summary.py --results results --tags prop_lit prop_num         --out results/meanerror_summary_prop.json
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from study2lib import specs
from study2lib.runlog import write_json, config_hash

N_BOOT = 10000


def load_arm(d):
    """{(state, year): {covset: record}} for the first learner of each covset."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        c = json.load(open(f, encoding="utf-8"))
        cell = {}
        for s in c["spec_results"]:
            if s["outcome_def"] != specs.OUTCOME_DEFS[0]:
                continue                       # the decomposition does not depend on the outcome
            cell.setdefault(s["covariate_set"], s)
        out[(c["state"], c["year"])] = {"specs": cell, "options": c.get("arm_options", {}),
                                        "load_options": c.get("load_options", {})}
    return out


def block_mean(values_by_state, rng, n_boot):
    states = sorted(values_by_state)
    blocks = [np.array(values_by_state[s], float) for s in states]
    allv = np.concatenate(blocks)
    k = len(blocks)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, k, k)
        means[b] = np.concatenate([blocks[i] for i in idx]).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(allv.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "min": float(allv.min()), "max": float(allv.max()), "n": int(len(allv))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--years-dir", default=None, help="default results/grid_years")
    ap.add_argument("--tags", nargs="*", default=["lit", "lit_lo", "lit_hi", "num"],
                    help="suffixes of results/grid_meanerror_<tag>")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default=None,
                    help="output file; default results/meanerror_summary.json (use a separate "
                         "file for the ages 25 to 65 arms, for example "
                         "results/meanerror_summary_workers.json)")
    a = ap.parse_args()
    years = load_arm(a.years_dir or os.path.join(a.results, "grid_years"))
    if not years:
        print("no years-coded cells found"); return 1
    out = {"seed": specs.PRIMARY_SEED, "n_boot": a.n_boot, "arms": {}}
    print(f"years-coded arm: {len(years)} markets")
    for tag in a.tags:
        d = os.path.join(a.results, f"grid_meanerror_{tag}")
        if not os.path.isdir(d) and os.path.isdir(os.path.join(a.results, f"grid_{tag}")):
            d = os.path.join(a.results, f"grid_{tag}")
        me = load_arm(d)
        if not me:
            print(f"  {tag}: no cells in {d}, skipped"); continue
        common = sorted(set(years) & set(me))
        first = next(iter(me.values()))
        proportional = "shift_female_share" in first["options"]
        delta = None if proportional else -float(first["options"]["shift_female_schl"])
        rec = {"delta_years": delta, "n_markets": len(common), "covsets": {}, "arm_dir": d}
        if proportional:
            rec["share_of_schooling_lead"] = float(first["options"]["shift_female_share"])
        for cs in sorted(next(iter(years.values()))["specs"]):
            rows = []
            quantities = ["U_y1", "U_yk", "U_m1", "U_mk", "deep_y", "deep_m", "meanerror_shift"]
            if proportional:
                quantities += ["applied_shift_years", "schooling_lead_years"]
            by_state = {q: {} for q in quantities}
            for key in common:
                sy = years[key]["specs"].get(cs); sm = me[key]["specs"].get(cs)
                if sy is None or sm is None:
                    continue
                uy1, uyk = sy["decomp_unexplained"], sy["eiv_unexplained_at_kappa"]
                um1, umk = sm["decomp_unexplained"], sm["eiv_unexplained_at_kappa"]
                sweep_m = [p["unexplained"] for p in sm["kappa_sweep"]]
                sweep_y = [p["unexplained"] for p in sy["kappa_sweep"]]
                floor_m = sweep_m[0]
                rows.append({"market": f"{key[0]} {key[1]}", "U_y1": uy1, "U_yk": uyk,
                             "U_m1": um1, "U_mk": umk, "U_m_floor": floor_m,
                             "deep_y": uyk - uy1, "deep_m": umk - um1,
                             "meanerror_shift": um1 - uy1,
                             "sign_flip_in_sweep_m": (min(sweep_m) < 0 < max(sweep_m)),
                             "sweep_m_max": max(sweep_m), "sweep_y_max": max(sweep_y)})
                if proportional:
                    lo = me[key]["load_options"]
                    rows[-1]["applied_shift_years"] = -float(lo["shift_female_schl_applied"])
                    rows[-1]["schooling_lead_years"] = float(lo["schooling_lead_years"])
                st = key[0]
                for q in by_state:
                    by_state[q].setdefault(st, []).append(rows[-1][q])
            if not rows:
                continue
            n = len(rows)
            summ = {
                "n": n,
                "reliability_correction_deepens_years_arm": int(sum(r["deep_y"] < 0 for r in rows)),
                "reliability_correction_deepens_after_meanerror": int(sum(r["deep_m"] < 0 for r in rows)),
                "reliability_correction_attenuates_after_meanerror": int(sum(r["deep_m"] > 0 for r in rows)),
                "U_m_anchor_negative": int(sum(r["U_mk"] < 0 for r in rows)),
                "U_m_anchor_positive": int(sum(r["U_mk"] > 0 for r in rows)),
                "U_m_floor_negative": int(sum(r["U_m_floor"] < 0 for r in rows)),
                "U_m_sign_flips_in_sweep": int(sum(r["sign_flip_in_sweep_m"] for r in rows)),
                "U_y_sign_flips_in_sweep": int(sum(r["sweep_y_max"] > 0 for r in rows)),
                "pooled": {},
            }
            for q in quantities:
                rng = np.random.default_rng(specs.PRIMARY_SEED)
                summ["pooled"][q] = block_mean(by_state[q], rng, a.n_boot)
            summ["markets"] = rows
            rec["covsets"][cs] = summ
            p = summ["pooled"]
            label = (f"share {rec['share_of_schooling_lead']:.6f} (mean shift {p['applied_shift_years']['mean']:.3f} yrs)"
                     if proportional else f"delta {delta:.3f} yrs")
            print(f"  {label}, {cs:12s} n={n:3d} | pooled U: uncorrected {p['U_y1']['mean']:+.3f}, "
                  f"reliability only {p['U_yk']['mean']:+.3f}, mean error only {p['U_m1']['mean']:+.3f}, "
                  f"both {p['U_mk']['mean']:+.3f} [{p['U_mk']['ci_lo']:+.3f}, {p['U_mk']['ci_hi']:+.3f}] | "
                  f"deepens after mean error {summ['reliability_correction_deepens_after_meanerror']}/{n}, "
                  f"attenuates {summ['reliability_correction_attenuates_after_meanerror']}/{n} | "
                  f"U both negative at anchor {summ['U_m_anchor_negative']}/{n}, at floor {summ['U_m_floor_negative']}/{n}, "
                  f"sign flips in sweep {summ['U_m_sign_flips_in_sweep']}/{n}")
        out["arms"][tag] = rec
    out["config_hash"] = config_hash({"tags": a.tags, "n_boot": a.n_boot, "seed": specs.PRIMARY_SEED})
    out["years_dir"] = a.years_dir or os.path.join(a.results, "grid_years")
    dest = a.out or os.path.join(a.results, "meanerror_summary.json")
    write_json(dest, out)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
