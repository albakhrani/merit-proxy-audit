#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replicate-weight standard errors for the primary decomposition
(REGISTERED_CHANGES R16, 28 Sep 2026, post-results).

For each of the 250 markets the registered primary sample is loaded
(registered attainment coding, premarket covariates, the grid's own filter)
and the decomposition quantities the paper reports are computed with the
full person weight PWGTP and with each of the 80 replicate weights PWGTP1
to PWGTP80, using the library functions the grid uses: the raw log-income
gap, the explained and unexplained components of the pooled-reference
decomposition, the corrected unexplained component at the literacy anchor,
and the deepening (corrected minus uncorrected). The successive-difference
replication variance is (4 / 80) times the sum over replicates of the
squared difference from the full-weight estimate; the standard error is its
square root (bootstrap.successive_difference_variance).

For comparison the script also reads, per market, the bootstrap standard
error implied by the stored percentile interval in results/boot_primary
((hi - lo) / 3.92) for the uncorrected residual, the corrected residual and
the deepening. Replicate weights address the sampling variance of the ACS
design and nothing else: they do not propagate the uncertainty of the
reliability anchor, which the version 2 bootstrap (results/boot_primary_v2)
does through its per-draw kappa.

    python scripts/16_replicate_variance.py --root acs_data --workers 3

Checks first that the cached person files carry PWGTP1 to PWGTP80 and stops
if they do not. Writes results/replicate_se.json: one record per market and
a summary block with, for each quantity, the median and range across
markets of the ratio replicate SE / bootstrap SE, and the number of markets
in which the point estimate plus or minus 1.96 replicate SE excludes zero
for the corrected residual and for the deepening.
"""
import argparse, glob, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPLICATES = [f"PWGTP{i}" for i in range(1, 81)]
QUANTITIES = ("raw", "explained", "unexplained", "corrected_unexplained", "deepening")


def _init_worker():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "2"


def _quantities(X, names, ly, g, w, kappa):
    """The five decomposition quantities under one weight vector."""
    from study2lib.decomposition import twofold_neumark
    from study2lib.cellrun import _corrected_unexplained
    dec = twofold_neumark(X[g == 1], ly[g == 1], X[g == 0], ly[g == 0], w[g == 1], w[g == 0])
    corrected = _corrected_unexplained(X, names, ly, g, w, kappa)[2]
    return np.array([dec["raw"], dec["explained"], dec["unexplained"], corrected,
                     corrected - dec["unexplained"]])


def _one(args):
    root, state, year, kappa, covariates, outcome_def = args
    from study2lib import specs
    from study2lib.acs import load_cell
    from study2lib.bootstrap import successive_difference_variance
    t0 = time.time()
    cell = load_cell(root, state, year, covariates, outcome_def, extra_columns=REPLICATES)
    X, names, g, w = cell["X"], cell["names"], cell["g"], cell["w"]
    ly = np.log(cell["y_cont"])
    full = _quantities(X, names, ly, g, w, kappa)
    reps = np.empty((len(REPLICATES), len(QUANTITIES)))
    negative = 0
    for r in range(len(REPLICATES)):
        wr = cell["extra"][:, r]
        if (wr < 0).any():
            negative += int((wr < 0).sum())
            wr = np.clip(wr, 0.0, None)
        reps[r] = _quantities(X, names, ly, g, wr, kappa)
    se = {q: float(np.sqrt(successive_difference_variance(full[i], reps[:, i], len(REPLICATES))))
          for i, q in enumerate(QUANTITIES)}
    return {"state": state, "year": year, "n": int(cell["n"]),
            "n_female": int((g == 1).sum()), "n_male": int((g == 0).sum()),
            "point": {q: float(full[i]) for i, q in enumerate(QUANTITIES)},
            "replicate_se": se,
            "negative_replicate_weights_clipped": negative,
            "timing_s": round(time.time() - t0, 1)}


def main():
    from study2lib import specs
    from study2lib.acs import STATE_FIPS, fetch_acs_cell
    from study2lib.runlog import write_json, config_hash
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="acs_data")
    ap.add_argument("--kappa", default="results/kappa.json")
    ap.add_argument("--skill", default="lit", choices=["lit", "num"])
    ap.add_argument("--boot", default="results/boot_primary")
    ap.add_argument("--out", default="results/replicate_se.json")
    ap.add_argument("--states", nargs="*", default=sorted(STATE_FIPS))
    ap.add_argument("--years", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    years = a.years or specs.PRIORITY_YEARS
    kappa = float(json.load(open(a.kappa, encoding="utf-8"))[a.skill]["overall"]["kappa"])
    covariates = specs.COVARIATE_SETS["premarket"]
    outcome_def = specs.OUTCOME_DEFS[0]

    # the cached files must carry the replicate weights; never refetch here
    missing = []
    for y in years:
        for s in a.states:
            path = os.path.join(a.root, str(y), "1-Year", f"psam_p{STATE_FIPS[s]}.csv")
            if not os.path.exists(path):
                missing.append(f"{s} {y}: file absent")
                continue
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                header = set(c.strip() for c in fh.readline().split(","))
            absent = [c for c in REPLICATES if c not in header]
            if absent:
                missing.append(f"{s} {y}: {len(absent)} replicate columns absent")
    if missing:
        print("replicate weights are not available in the cached files; stopping:")
        for m in missing[:20]:
            print("  " + m)
        return 2
    print(f"all {len(years) * len(a.states)} cached files carry PWGTP1 to PWGTP80")

    chash = config_hash({"arm": "replicate_variance", "covariates": list(covariates),
                         "outcome_def": outcome_def, "kappa": round(kappa, 6), "skill": a.skill,
                         "n_replicates": len(REPLICATES), "variance_factor": 4.0 / len(REPLICATES),
                         "seed": specs.PRIMARY_SEED})
    jobs = [(a.root, s, y, kappa, list(covariates), outcome_def) for y in years for s in a.states]
    markets = {}
    done = fail = 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker,
                             max_tasks_per_child=1) as ex:
        futures = {ex.submit(_one, j): j for j in jobs}
        for f in as_completed(futures):
            _, s, y, *_ = futures[f]
            try:
                r = f.result()
                done += 1
                markets[f"{s} {y}"] = r
                print(f"  [{done+fail}/{len(jobs)}] {s} {y}  {r['timing_s']:6.1f}s  "
                      f"corrected {r['point']['corrected_unexplained']:+.4f} (rep se {r['replicate_se']['corrected_unexplained']:.4f})")
            except Exception as e:
                fail += 1
                print(f"  [{done+fail}/{len(jobs)}] {s} {y}  FAILED  {e!r}")
    if fail:
        print(f"{fail} markets failed; nothing written")
        return 1

    # bootstrap standard errors implied by the stored percentile intervals
    boot_map = {"unexplained": "uncorrected_residual", "corrected_unexplained": "corrected_residual",
                "deepening": "deepening"}
    for f in sorted(glob.glob(os.path.join(a.boot, "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        key = f"{d['state']} {d['year']}"
        if key not in markets:
            continue
        m = markets[key]
        m["bootstrap_se"] = {}
        m["bootstrap_point"] = {}
        for q, bq in boot_map.items():
            iv = d["intervals"][bq]
            m["bootstrap_se"][q] = float((iv["hi"] - iv["lo"]) / 3.92)
            m["bootstrap_point"][q] = float(d["point"][bq])
        m["ratio_replicate_over_bootstrap"] = {
            q: (m["replicate_se"][q] / m["bootstrap_se"][q] if m["bootstrap_se"][q] > 0 else None)
            for q in boot_map}

    summary = {"n_markets": len(markets), "ratio_replicate_se_over_bootstrap_se": {},
               "excludes_zero_at_1_96_replicate_se": {}}
    for q in boot_map:
        ratios = np.array([m["ratio_replicate_over_bootstrap"][q] for m in markets.values()
                           if m.get("ratio_replicate_over_bootstrap", {}).get(q) is not None])
        summary["ratio_replicate_se_over_bootstrap_se"][q] = {
            "median": float(np.median(ratios)), "min": float(ratios.min()),
            "max": float(ratios.max()), "n": int(len(ratios))}
    for q in ("corrected_unexplained", "deepening"):
        cnt = sum(1 for m in markets.values()
                  if abs(m["point"][q]) > 1.96 * m["replicate_se"][q])
        summary["excludes_zero_at_1_96_replicate_se"][q] = int(cnt)
    summary["replicate_se"] = {q: {"median": float(np.median([m["replicate_se"][q] for m in markets.values()])),
                                   "min": float(min(m["replicate_se"][q] for m in markets.values())),
                                   "max": float(max(m["replicate_se"][q] for m in markets.values()))}
                               for q in QUANTITIES}
    summary["point_matches_bootstrap_point"] = int(sum(
        1 for m in markets.values() if "bootstrap_point" in m
        and all(abs(m["point"][q] - m["bootstrap_point"][q]) < 1e-9 for q in boot_map)))
    summary["negative_replicate_weights_clipped_total"] = int(sum(
        m["negative_replicate_weights_clipped"] for m in markets.values()))

    out = {"arm": "replicate_variance", "seed": specs.PRIMARY_SEED, "decomp_only": True,
           "arm_options": {"covariates": list(covariates), "outcome_def": outcome_def,
                           "schooling": "code", "kappa_point": kappa, "skill": a.skill,
                           "replicate_weights": "PWGTP1 to PWGTP80",
                           "variance": "(4 / 80) * sum over r of (theta_r - theta_full) ** 2",
                           "bootstrap_se": "(hi - lo) / 3.92 from the stored percentile interval in "
                                           + a.boot,
                           "note": "replicate weights cover the sampling variance of the ACS design "
                                   "and do not propagate the anchor's uncertainty, which the "
                                   "version 2 bootstrap does"},
           "kappa_point": kappa, "n_replicates": len(REPLICATES),
           "markets": dict(sorted(markets.items())), "summary": summary, "config_hash": chash}
    write_json(a.out, out)
    print("\nsummary over %d markets" % summary["n_markets"])
    for q, r in summary["ratio_replicate_se_over_bootstrap_se"].items():
        print("  ratio replicate SE / bootstrap SE, %-22s median %.3f  range [%.3f, %.3f]  (n %d)"
              % (q, r["median"], r["min"], r["max"], r["n"]))
    for q, c in summary["excludes_zero_at_1_96_replicate_se"].items():
        print("  point +/- 1.96 replicate SE excludes zero, %-22s %d of %d markets" % (q, c, summary["n_markets"]))
    print("  replicate SE medians: " + ", ".join("%s %.4f" % (q, r["median"]) for q, r in summary["replicate_se"].items()))
    print("  full-weight points equal the stored bootstrap points in %d markets; negative replicate weights clipped: %d"
          % (summary["point_matches_bootstrap_point"], summary["negative_replicate_weights_clipped_total"]))
    print(f"wrote {a.out}; config_hash {chash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
