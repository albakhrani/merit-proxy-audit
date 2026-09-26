# -*- coding: utf-8 -*-
"""Pooled cross-market estimates of the decomposition residual.

Added 22 September 2026 after a simulated referee report asked for a pooled
estimate of the corrected gap with an interval that respects the dependence
across markets, with the per-market results reported as heterogeneity. The
manuscript had reported the state-clustered interval for R only.

Reads results/boot_primary/*.json (one file per market, primary arm) and
results/grid/*.json (the reliability sweep of the premarket decomposition).
Writes results/pooled_gap.json. Nothing is re-estimated: the per-market point
values are the stored ones, and the pooling is the equal-weight mean across
the 250 markets, the same convention as the mean of R in the manuscript.

  1. mean across markets of the uncorrected residual, the corrected residual
     at the propagated anchor, and the deepening (corrected minus uncorrected),
     each with the leave-one-state-out range and a state-block bootstrap
     interval (states drawn with replacement, all years of a drawn state kept,
     seeded from the primary seed);
  2. the pooled mean of the corrected residual at every value of the stored
     reliability sweep, with the same block interval, so that the sweep can
     be read at the national level;
  3. the between-sex difference in weighted mean schooling (women minus men)
     across markets, with the three Utah values, for the reader's check that
     the Utah exception is a near-tie in mean schooling rather than an
     artefact.

Run: python scripts/10_pooled_gap.py [--results results]
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from study2lib import specs
from study2lib.runlog import write_json, config_hash

N_BLOCK_BOOT = 10000
FIELDS = ("uncorrected_residual", "corrected_residual", "deepening")


def block_summary(by_state, rng, n_boot):
    """Mean, LOSO range and state-block bootstrap interval for one quantity."""
    states = sorted(by_state)
    allv = np.array([v for s in states for v in by_state[s]], float)
    loso = {}
    for s in states:
        keep = np.array([v for t in states if t != s for v in by_state[t]], float)
        loso[s] = float(keep.mean())
    lo_vals = np.array(list(loso.values()))
    blocks = [np.array(by_state[s], float) for s in states]
    k = len(blocks)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, k, k)
        means[b] = np.concatenate([blocks[i] for i in idx]).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    naive_se = allv.std(ddof=1) / np.sqrt(len(allv))
    return {
        "mean": float(allv.mean()), "sd": float(allv.std(ddof=1)),
        "min": float(allv.min()), "max": float(allv.max()),
        "n_cells": int(len(allv)), "n_states": k,
        "loso": {"min": float(lo_vals.min()), "min_state": min(loso, key=loso.get),
                 "max": float(lo_vals.max()), "max_state": max(loso, key=loso.get)},
        "state_block_bootstrap": {"n_boot": n_boot, "ci_lo": float(lo), "ci_hi": float(hi),
                                  "se": float(means.std(ddof=1)),
                                  "share_lt0": float((means < 0).mean())},
        "naive_iid_se": float(naive_se),
        "se_ratio_block_over_iid": float(means.std(ddof=1) / naive_se),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--n-boot", type=int, default=N_BLOCK_BOOT)
    a = ap.parse_args()

    # 1. per-market point values from the bootstrap files (primary arm)
    by_state = {f: {} for f in FIELDS}
    sgap = {}
    kappa_points, arms = set(), set()
    for f in sorted(glob.glob(os.path.join(a.results, "boot_primary", "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        arms.add(tuple(d["arm"])); kappa_points.add(round(float(d["kappa_point"]), 9))
        for q in FIELDS:
            by_state[q].setdefault(d["state"], []).append(float(d["point"][q]))
        gm = d["group_moments"]["SCHL"]
        sgap["%s %d" % (d["state"], d["year"])] = float(gm["female"]["mean"] - gm["male"]["mean"])
    assert len(arms) == 1, arms
    assert len(kappa_points) == 1, kappa_points
    n_markets = sum(len(v) for v in by_state["corrected_residual"].values())
    print("primary arm %s, %d markets, kappa_point %.6f" % (arms.pop(), n_markets, kappa_points.pop()))

    out = {"arm": list(json.load(open(sorted(glob.glob(os.path.join(a.results, "boot_primary", "*.json")))[0]))["arm"]),
           "n_markets": n_markets, "seed": specs.PRIMARY_SEED,
           "pooling": "equal-weight mean across markets, the convention used for the mean of R",
           "resampling": "states with replacement, all years of a drawn state kept",
           "kappa_point": float(json.load(open(sorted(glob.glob(os.path.join(a.results, "boot_primary", "*.json")))[0]))["kappa_point"])}

    for q in FIELDS:
        rng = np.random.default_rng(specs.PRIMARY_SEED)   # same seed per quantity: draws are comparable
        out[q] = block_summary(by_state[q], rng, a.n_boot)
        s = out[q]
        print("%-22s mean %+.4f  sd %.4f  LOSO [%+.4f %s, %+.4f %s]  block 95%% [%+.4f, %+.4f]  se %.4f (iid %.4f, ratio %.2f)"
              % (q, s["mean"], s["sd"], s["loso"]["min"], s["loso"]["min_state"], s["loso"]["max"], s["loso"]["max_state"],
                 s["state_block_bootstrap"]["ci_lo"], s["state_block_bootstrap"]["ci_hi"],
                 s["state_block_bootstrap"]["se"], s["naive_iid_se"], s["se_ratio_block_over_iid"]))

    # 2. pooled sweep: mean corrected residual at each stored kappa, premarket decomposition
    sweep_by_state = {}
    kappas = None
    for f in sorted(glob.glob(os.path.join(a.results, "grid", "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        s = next(x for x in d["spec_results"] if x["covariate_set"] == "premarket")
        ks = [round(float(p["kappa"]), 6) for p in s["kappa_sweep"]]
        if kappas is None:
            kappas = ks
        assert ks == kappas, (f, ks)
        sweep_by_state.setdefault(d["state"], []).append([float(p["unexplained"]) for p in s["kappa_sweep"]])
    states = sorted(sweep_by_state)
    blocks = [np.array(sweep_by_state[s], float) for s in states]      # each (years, n_kappa)
    allrows = np.concatenate(blocks)
    rng = np.random.default_rng(specs.PRIMARY_SEED)
    k = len(blocks)
    boot = np.empty((a.n_boot, len(kappas)))
    for b in range(a.n_boot):
        idx = rng.integers(0, k, k)
        boot[b] = np.concatenate([blocks[i] for i in idx]).mean(axis=0)
    lo = np.percentile(boot, 2.5, axis=0); hi = np.percentile(boot, 97.5, axis=0)
    out["pooled_sweep"] = {
        "covariate_set": "premarket", "n_markets": int(allrows.shape[0]),
        "kappa": kappas,
        "mean": [float(v) for v in allrows.mean(axis=0)],
        "ci_lo": [float(v) for v in lo], "ci_hi": [float(v) for v in hi],
        "max_over_sweep": float(allrows.mean(axis=0).max()),
        "max_ci_hi_over_sweep": float(hi.max()),
        "note": "equal-weight mean across the 250 markets of the corrected residual at each stored sweep value; block interval as above"}
    print("pooled sweep: kappa %.2f mean %+.4f [%+.4f, %+.4f] ... kappa %.2f mean %+.4f [%+.4f, %+.4f]; largest upper limit %+.4f"
          % (kappas[0], allrows.mean(axis=0)[0], lo[0], hi[0], kappas[-1], allrows.mean(axis=0)[-1], lo[-1], hi[-1], hi.max()))

    # 3. schooling gap, women minus men, weighted means of the attainment code
    g = np.array(list(sgap.values()))
    utah = {m: v for m, v in sgap.items() if m.startswith("UT ")}
    others = np.array([v for m, v in sgap.items() if v >= 0])   # the 247 markets in which women lead
    out["schooling_gap_women_minus_men"] = {
        "unit": "ACS attainment code, weighted means", "n": int(len(g)),
        "median": float(np.median(g)), "min": float(g.min()), "min_market": min(sgap, key=sgap.get),
        "max": float(g.max()), "max_market": max(sgap, key=sgap.get),
        "n_negative": int((g < 0).sum()), "negative_markets": {m: float(v) for m, v in sgap.items() if v < 0},
        "utah": {m: float(v) for m, v in utah.items()},
        "min_where_women_lead": float(others.min()), "max_where_women_lead": float(others.max()),
        "median_where_women_lead": float(np.median(others)), "n_where_women_lead": int(len(others))}
    print("schooling gap women-men: median %+.3f, range [%+.3f (%s), %+.3f (%s)], negative in %d; Utah %s; where women lead (n=%d): min %+.4f, median %+.4f, max %+.4f"
          % (np.median(g), g.min(), min(sgap, key=sgap.get), g.max(), max(sgap, key=sgap.get), (g < 0).sum(),
             {m: round(v, 4) for m, v in utah.items()}, len(others), others.min(), np.median(others), others.max()))

    cfg = {"n_boot": a.n_boot, "seed": specs.PRIMARY_SEED, "fields": list(FIELDS), "pooling": out["pooling"]}
    out["config_hash"] = config_hash(cfg)
    write_json(os.path.join(a.results, "pooled_gap.json"), out)
    print("wrote %s/pooled_gap.json" % a.results)


if __name__ == "__main__":
    main()
