#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Group-specific reference structure for the primary decomposition
(REGISTERED_CHANGES R17, 28 Sep 2026, post-results).

For each of the 250 markets, on the full sample with the registered
attainment coding and the premarket covariates, the outcome equation
(log income) is fitted separately by sex by weighted least squares and the
twofold decomposition is reported under (i) the male coefficients and (ii)
the female coefficients as the reference structure
(decomposition.twofold_groupref), each uncorrected and corrected. The
errors-in-variables correction is applied to the schooling column inside
each sex's regression, once with the common literacy anchor in both
regressions and once with the sex-specific anchors of results/kappa.json
(literacy, sex strata: GENDER_R 1 men, GENDER_R 2 women). Each cell also
stores the pooled-reference values of the same market from results/grid
for comparison. Learners, the DML estimate and the R bootstrap are not run.

    python scripts/17_groupref.py --root acs_data --workers 3

Writes results/grid_groupref/{STATE}_{YEAR}.json (resumable; rerun to
retry failures) and, once every cell is present, results/groupref_summary.json
with pooled means and state-block intervals (the pooling code of
scripts/10_pooled_gap.py) for every stored quantity and for the difference
between the sex-specific and the common-anchor corrected components under
each reference, with the count of markets in which that difference is
positive and in which the corrected component is negative.
"""
import argparse, glob, importlib.util, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QUANTITIES = ("raw",
              "male_ref_uncorrected", "male_ref_corrected_common", "male_ref_corrected_sexspecific",
              "female_ref_uncorrected", "female_ref_corrected_common", "female_ref_corrected_sexspecific",
              "pooled_ref_uncorrected", "pooled_ref_corrected")
DIFFERENCES = ("male_ref_sexspecific_minus_common", "female_ref_sexspecific_minus_common")


def _init_worker():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "2"


def _grid_values(griddir, state, year):
    """Pooled-reference values of the same market from the registered grid."""
    p = os.path.join(griddir, f"{state}_{year}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    s = next(x for x in d["spec_results"] if x["covariate_set"] == "premarket")
    return {"decomp_raw_logpts": s["decomp_raw_logpts"], "decomp_explained": s["decomp_explained"],
            "decomp_unexplained": s["decomp_unexplained"],
            "eiv_unexplained_at_kappa": s["eiv_unexplained_at_kappa"], "kappa_point": s["kappa_point"],
            "n": s["n"], "config_hash": d.get("config_hash")}


def _one(args):
    root, state, year, outdir, chash, covariates, outcome_def, k_common, k_men, k_women, griddir = args
    from study2lib import specs
    from study2lib.acs import load_cell
    from study2lib.decomposition import twofold_groupref
    from study2lib.runlog import write_json
    t0 = time.time()
    cell = load_cell(root, state, year, covariates, outcome_def)
    X, names, g, w = cell["X"], cell["names"], cell["g"], cell["w"]
    ly = np.log(cell["y_cont"])
    j = names.index(specs.PROXY_COLUMN)
    a, b = g == 1, g == 0          # a = women, b = men

    def run(ka, kb):
        try:
            r = twofold_groupref(X[a], ly[a], X[b], ly[b], w[a], w[b], proxy_index=j, kappa_a=ka, kappa_b=kb)
        except np.linalg.LinAlgError:
            return None
        return {"raw": r["raw"], "explained_male_ref": r["explained_ref_b"],
                "unexplained_male_ref": r["unexplained_ref_b"],
                "explained_female_ref": r["explained_ref_a"], "unexplained_female_ref": r["unexplained_ref_a"],
                "beta_women": [float(v) for v in r["beta_a"]], "beta_men": [float(v) for v in r["beta_b"]]}

    unc = run(None, None)
    com = run(k_common, k_common)
    sex = run(k_women, k_men)
    grid = _grid_values(griddir, state, year)
    res = {"state": state, "year": year, "seed": specs.PRIMARY_SEED, "timing_s": round(time.time() - t0, 1),
           "arm": "groupref", "decomp_only": True,
           "arm_options": {"covariates": list(covariates), "outcome_def": outcome_def, "schooling": "code",
                           "kappa_common": k_common, "kappa_men": k_men, "kappa_women": k_women,
                           "reference": "group-specific coefficients from separate weighted least squares by sex",
                           "correction": "errors-in-variables on the schooling column inside each sex's regression"},
           "n": int(cell["n"]), "n_female": int(a.sum()), "n_male": int(b.sum()),
           "covariate_names": names,
           "uncorrected": unc, "corrected_common_anchor": com, "corrected_sex_specific_anchors": sex,
           "unexplained": {
               "raw": unc["raw"],
               "male_ref_uncorrected": unc["unexplained_male_ref"],
               "male_ref_corrected_common": com["unexplained_male_ref"] if com else None,
               "male_ref_corrected_sexspecific": sex["unexplained_male_ref"] if sex else None,
               "female_ref_uncorrected": unc["unexplained_female_ref"],
               "female_ref_corrected_common": com["unexplained_female_ref"] if com else None,
               "female_ref_corrected_sexspecific": sex["unexplained_female_ref"] if sex else None,
               "pooled_ref_uncorrected": grid["decomp_unexplained"] if grid else None,
               "pooled_ref_corrected": grid["eiv_unexplained_at_kappa"] if grid else None},
           "pooled_reference_from_grid": grid,
           "config_hash": chash}
    write_json(os.path.join(outdir, f"{state}_{year}.json"), res)
    return state, year, res["timing_s"], com is not None and sex is not None


def summarise(outdir, out_path, n_boot, chash):
    from study2lib import specs
    from study2lib.runlog import write_json
    spec = importlib.util.spec_from_file_location(
        "pooled_gap", os.path.join(os.path.dirname(os.path.abspath(__file__)), "10_pooled_gap.py"))
    pooled_gap = importlib.util.module_from_spec(spec); spec.loader.exec_module(pooled_gap)
    cells = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(outdir, "*.json")))]
    cells = [c for c in cells if c.get("config_hash") == chash]
    by_state = {q: {} for q in QUANTITIES + DIFFERENCES}
    markets = {}
    for c in cells:
        u = c["unexplained"]
        if any(u[q] is None for q in QUANTITIES):
            continue
        d = {"male_ref_sexspecific_minus_common": u["male_ref_corrected_sexspecific"] - u["male_ref_corrected_common"],
             "female_ref_sexspecific_minus_common": u["female_ref_corrected_sexspecific"] - u["female_ref_corrected_common"]}
        markets[f"{c['state']} {c['year']}"] = {**u, **d}
        for q in QUANTITIES:
            by_state[q].setdefault(c["state"], []).append(float(u[q]))
        for q in DIFFERENCES:
            by_state[q].setdefault(c["state"], []).append(float(d[q]))
    out = {"arm": "groupref", "seed": specs.PRIMARY_SEED, "n_boot": n_boot, "n_markets": len(markets),
           "pooling": "equal-weight mean across markets; state-block bootstrap, states drawn with "
                      "replacement, all years of a drawn state kept",
           "pooled": {}, "counts": {}, "markets": markets, "cell_config_hash": chash}
    for q in QUANTITIES + DIFFERENCES:
        rng = np.random.default_rng(specs.PRIMARY_SEED)
        out["pooled"][q] = pooled_gap.block_summary(by_state[q], rng, n_boot)
    vals = {q: np.array([m[q] for m in markets.values()]) for q in QUANTITIES + DIFFERENCES}
    for q in DIFFERENCES:
        out["counts"][q + "_positive"] = int((vals[q] > 0).sum())
    for q in QUANTITIES[1:]:
        out["counts"][q + "_negative"] = int((vals[q] < 0).sum())
    from study2lib.runlog import config_hash
    out["config_hash"] = config_hash({"arm": "groupref_summary", "n_boot": n_boot, "seed": specs.PRIMARY_SEED,
                                      "cell_config_hash": chash})
    write_json(out_path, out)
    print("\ngroupref summary over %d markets (block bootstrap %d draws)" % (len(markets), n_boot))
    for q in QUANTITIES + DIFFERENCES:
        s = out["pooled"][q]; b = s["state_block_bootstrap"]
        line = "  %-36s mean %+.4f  block 95%% [%+.4f, %+.4f]  range [%+.4f, %+.4f]" % (
            q, s["mean"], b["ci_lo"], b["ci_hi"], s["min"], s["max"])
        if q in DIFFERENCES:
            line += "  positive in %d" % out["counts"][q + "_positive"]
        elif q != "raw":
            line += "  negative in %d" % out["counts"][q + "_negative"]
        print(line)
    print(f"wrote {out_path}; summary config_hash {out['config_hash']}")


def main():
    from study2lib import specs
    from study2lib.acs import STATE_FIPS
    from study2lib.runlog import config_hash
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="acs_data")
    ap.add_argument("--kappa", default="results/kappa.json")
    ap.add_argument("--skill", default="lit", choices=["lit", "num"])
    ap.add_argument("--griddir", default="results/grid")
    ap.add_argument("--outdir", default="results/grid_groupref")
    ap.add_argument("--summary", default="results/groupref_summary.json")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--states", nargs="*", default=sorted(STATE_FIPS))
    ap.add_argument("--years", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    years = a.years or specs.PRIORITY_YEARS
    kap = json.load(open(a.kappa, encoding="utf-8"))[a.skill]
    k_common = float(kap["overall"]["kappa"])
    k_men = float(kap["sex"]["1"]["kappa"]); k_women = float(kap["sex"]["2"]["kappa"])
    covariates = specs.COVARIATE_SETS["premarket"]; outcome_def = specs.OUTCOME_DEFS[0]
    chash = config_hash({"arm": "groupref", "covariates": list(covariates), "outcome_def": outcome_def,
                         "skill": a.skill, "kappa_common": round(k_common, 6),
                         "kappa_men": round(k_men, 6), "kappa_women": round(k_women, 6),
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
            todo.append((a.root, s, y, a.outdir, chash, list(covariates), outcome_def,
                         k_common, k_men, k_women, a.griddir))
    print(f"arm groupref: anchors common {k_common:.6f}, men {k_men:.6f}, women {k_women:.6f}; "
          f"{len(todo)} cells to run, {len(years)*len(a.states)-len(todo)} already current; "
          f"workers {a.workers}; outdir {a.outdir}; config_hash {chash}")
    done = fail = notcorr = 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker, max_tasks_per_child=1) as ex:
        futures = {ex.submit(_one, t): t for t in todo}
        for f in as_completed(futures):
            _, s, y, *_ = futures[f]
            try:
                st, yr, secs, ok = f.result()
                done += 1; notcorr += (not ok)
                print(f"  [{done+fail}/{len(todo)}] {st} {yr}  {secs:6.1f}s" + ("" if ok else "  [not correctable]"))
            except Exception as e:
                fail += 1
                print(f"  [{done+fail}/{len(todo)}] {s} {y}  FAILED  {e!r}")
    print(f"\n{done} cells written, {fail} failed" + (f"; {notcorr} not correctable" if notcorr else ""))
    if fail:
        return 1
    present = len(glob.glob(os.path.join(a.outdir, "*.json")))
    if present >= len(years) * len(a.states):
        summarise(a.outdir, a.summary, a.n_boot, chash)
    return 0


if __name__ == "__main__":
    sys.exit(main())
