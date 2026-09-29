#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attainment-indicator arm for the primary decomposition
(REGISTERED_CHANGES R18, 28 Sep 2026, post-results).

For each of the 250 markets, on the full sample with the premarket
covariates, the pooled-reference twofold decomposition of log income is
computed with the attainment code SCHL entered as a set of indicator
columns, one per value observed in the market with the lowest dropped
(decomposition.indicator_columns), plus age. The errors-in-variables
correction is not defined for a set of indicators, so the arm is
uncorrected; the run log says so in arm_options. Each cell stores the raw
gap and the explained and unexplained components, together with the
unexplained component of the linear coding from the same market in
results/grid for comparison. Learners, the DML estimate and the R
bootstrap are not run.

    python scripts/18_dummies.py --root acs_data --workers 3

Writes results/grid_dummies/{STATE}_{YEAR}.json (resumable; rerun to retry
failures) and, once every cell is present, results/dummies_summary.json:
the pooled mean and state-block interval of the unexplained component (the
pooling code of scripts/10_pooled_gap.py), the count of markets in which it
is negative, and the per-market difference from the linear-coding
unexplained component (mean, range, block interval).
"""
import argparse, glob, importlib.util, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _init_worker():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "2"


def _one(args):
    root, state, year, outdir, chash, covariates, outcome_def, griddir = args
    from study2lib import specs
    from study2lib.acs import load_cell
    from study2lib.decomposition import twofold_neumark, indicator_columns
    from study2lib.runlog import write_json
    t0 = time.time()
    cell = load_cell(root, state, year, covariates, outcome_def)
    X, names, g, w = cell["X"], cell["names"], cell["g"], cell["w"]
    ly = np.log(cell["y_cont"])
    j = names.index(specs.PROXY_COLUMN)
    dummies, levels = indicator_columns(X[:, j].astype(int))
    others = [i for i in range(X.shape[1]) if i != j]
    Xd = np.column_stack([X[:, others], dummies])
    a, b = g == 1, g == 0
    dec = twofold_neumark(Xd[a], ly[a], Xd[b], ly[b], w[a], w[b])
    linear = None
    p = os.path.join(griddir, f"{state}_{year}.json")
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8"))
        s = next(x for x in d["spec_results"] if x["covariate_set"] == "premarket")
        linear = {"decomp_raw_logpts": s["decomp_raw_logpts"], "decomp_explained": s["decomp_explained"],
                  "decomp_unexplained": s["decomp_unexplained"], "config_hash": d.get("config_hash")}
    res = {"state": state, "year": year, "seed": specs.PRIMARY_SEED, "timing_s": round(time.time() - t0, 1),
           "arm": "dummies", "decomp_only": True,
           "arm_options": {"covariates": list(covariates), "outcome_def": outcome_def,
                           "schooling": "indicator columns, one per observed SCHL value, lowest dropped",
                           "correction": "none: the errors-in-variables correction is not defined for a "
                                         "set of indicators, so the arm is uncorrected",
                           "reference": "pooled"},
           "n": int(cell["n"]), "n_female": int(a.sum()), "n_male": int(b.sum()),
           "n_schl_levels_observed": int(len(levels) + 1), "indicator_levels": levels,
           "n_covariate_columns": int(Xd.shape[1]),
           "decomp_raw_logpts": dec["raw"], "decomp_explained": dec["explained"],
           "decomp_unexplained": dec["unexplained"],
           "linear_coding_from_grid": linear,
           "difference_from_linear_unexplained": (dec["unexplained"] - linear["decomp_unexplained"]) if linear else None,
           "config_hash": chash}
    write_json(os.path.join(outdir, f"{state}_{year}.json"), res)
    return state, year, res["timing_s"], res["n_schl_levels_observed"]


def summarise(outdir, out_path, n_boot, chash):
    from study2lib import specs
    from study2lib.runlog import write_json, config_hash
    spec = importlib.util.spec_from_file_location(
        "pooled_gap", os.path.join(os.path.dirname(os.path.abspath(__file__)), "10_pooled_gap.py"))
    pooled_gap = importlib.util.module_from_spec(spec); spec.loader.exec_module(pooled_gap)
    cells = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(outdir, "*.json")))]
    cells = [c for c in cells if c.get("config_hash") == chash and c.get("difference_from_linear_unexplained") is not None]
    quantities = ("raw", "explained", "unexplained", "linear_unexplained", "difference_from_linear")
    by_state = {q: {} for q in quantities}
    markets = {}
    for c in cells:
        rec = {"raw": c["decomp_raw_logpts"], "explained": c["decomp_explained"],
               "unexplained": c["decomp_unexplained"],
               "linear_unexplained": c["linear_coding_from_grid"]["decomp_unexplained"],
               "difference_from_linear": c["difference_from_linear_unexplained"],
               "n_schl_levels_observed": c["n_schl_levels_observed"]}
        markets[f"{c['state']} {c['year']}"] = rec
        for q in quantities:
            by_state[q].setdefault(c["state"], []).append(float(rec[q]))
    out = {"arm": "dummies", "seed": specs.PRIMARY_SEED, "n_boot": n_boot, "n_markets": len(markets),
           "pooling": "equal-weight mean across markets; state-block bootstrap, states drawn with "
                      "replacement, all years of a drawn state kept",
           "pooled": {}, "counts": {}, "markets": markets, "cell_config_hash": chash}
    for q in quantities:
        rng = np.random.default_rng(specs.PRIMARY_SEED)
        out["pooled"][q] = pooled_gap.block_summary(by_state[q], rng, n_boot)
    u = np.array([m["unexplained"] for m in markets.values()])
    d = np.array([m["difference_from_linear"] for m in markets.values()])
    out["counts"] = {"unexplained_negative": int((u < 0).sum()),
                     "difference_from_linear_positive": int((d > 0).sum()),
                     "difference_from_linear_negative": int((d < 0).sum())}
    out["config_hash"] = config_hash({"arm": "dummies_summary", "n_boot": n_boot, "seed": specs.PRIMARY_SEED,
                                      "cell_config_hash": chash})
    write_json(out_path, out)
    print("\ndummies summary over %d markets (block bootstrap %d draws)" % (len(markets), n_boot))
    for q in quantities:
        s = out["pooled"][q]; b = s["state_block_bootstrap"]
        print("  %-24s mean %+.4f  block 95%% [%+.4f, %+.4f]  range [%+.4f, %+.4f]"
              % (q, s["mean"], b["ci_lo"], b["ci_hi"], s["min"], s["max"]))
    print("  unexplained negative in %d of %d markets; difference from linear coding positive in %d, negative in %d"
          % (out["counts"]["unexplained_negative"], len(markets),
             out["counts"]["difference_from_linear_positive"], out["counts"]["difference_from_linear_negative"]))
    print(f"wrote {out_path}; summary config_hash {out['config_hash']}")


def main():
    from study2lib import specs
    from study2lib.acs import STATE_FIPS
    from study2lib.runlog import config_hash
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="acs_data")
    ap.add_argument("--griddir", default="results/grid")
    ap.add_argument("--outdir", default="results/grid_dummies")
    ap.add_argument("--summary", default="results/dummies_summary.json")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--states", nargs="*", default=sorted(STATE_FIPS))
    ap.add_argument("--years", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    years = a.years or specs.PRIORITY_YEARS
    covariates = specs.COVARIATE_SETS["premarket"]; outcome_def = specs.OUTCOME_DEFS[0]
    chash = config_hash({"arm": "dummies", "covariates": list(covariates), "outcome_def": outcome_def,
                         "schooling": "indicators_lowest_dropped", "correction": None,
                         "seed": specs.PRIMARY_SEED})
    os.makedirs(a.outdir, exist_ok=True)
    todo = []
    for y in years:
        for s in a.states:
            p = os.path.join(a.outdir, f"{s}_{y}.json")
            if os.path.exists(p):
                try:
                    if json.load(open(p, encoding="utf-8")).get("config_hash") == chash:
                        continue
                except Exception:
                    pass
            todo.append((a.root, s, y, a.outdir, chash, list(covariates), outcome_def, a.griddir))
    print(f"arm dummies: {len(todo)} cells to run, {len(years)*len(a.states)-len(todo)} already current; "
          f"workers {a.workers}; outdir {a.outdir}; config_hash {chash}")
    done = fail = 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker, max_tasks_per_child=1) as ex:
        futures = {ex.submit(_one, t): t for t in todo}
        for f in as_completed(futures):
            _, s, y, *_ = futures[f]
            try:
                st, yr, secs, nlev = f.result()
                done += 1
                print(f"  [{done+fail}/{len(todo)}] {st} {yr}  {secs:6.1f}s  {nlev} attainment levels")
            except Exception as e:
                fail += 1
                print(f"  [{done+fail}/{len(todo)}] {s} {y}  FAILED  {e!r}")
    print(f"\n{done} cells written, {fail} failed")
    if fail:
        return 1
    if len(glob.glob(os.path.join(a.outdir, "*.json"))) >= len(years) * len(a.states):
        summarise(a.outdir, a.summary, a.n_boot, chash)
    return 0


if __name__ == "__main__":
    sys.exit(main())
