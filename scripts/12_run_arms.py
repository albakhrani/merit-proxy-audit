#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arms added after the internal review of 22 September 2026
(REGISTERED_CHANGES R4-R9 and R11, post-results).

One driver for every arm added after the simulated referee report. Each arm
is the registered cell run with one option changed in load_cell, written to
its own directory so results/grid and results/grid_h are never touched:

    python scripts/12_run_arms.py --arm years    --root acs_data --workers 3
    python scripts/12_run_arms.py --arm wage     --root acs_data --workers 3
    python scripts/12_run_arms.py --arm occ      --root acs_data --workers 3
    python scripts/12_run_arms.py --arm adjinc   --root acs_data --workers 3
    python scripts/12_run_arms.py --arm wmedian  --root acs_data --workers 3
    python scripts/12_run_arms.py --arm age2565  --root acs_data --workers 3
    python scripts/12_run_arms.py --arm exact    --root acs_data --workers 3
    python scripts/12_run_arms.py --arm meanerror --delta 0.297 --outdir results/grid_meanerror_lit --root acs_data

Learners default to logistic only (minutes per arm); add --learners
logistic xgboost tabpfn for the full set on an arm that may replace the
primary specification (years). Every cell stores the corrected residual at
the exact anchors 0.187 (all adults) and 0.243 (earners 25 to 65), read
from results/kappa.json and results/kappa_workers.json, in
eiv_unexplained_at. The age2565 arm propagates the earners anchor as its
kappa_point; every other arm propagates the registered full-sample anchor.

The occ arm runs the extended covariates with occupation major groups
appended, under the covariate-set name extended_occ; the premarket set is
unchanged by occupation and is not rerun here.

The meanerror arm (R11) is the years-coded arm with women's schooling
LOWERED by --delta years before the decomposition and its correction, where
delta is the sex difference in the mean of the proxy error estimated by
11_piaac_validity.py (positive where women's schooling overstates their
assessed skill; the script reports it in years of qualification, which is
why this arm uses the years coding, so the units match). Only the
decomposition fields of this arm are meaningful: the screening model must
see the proxy as reported, so its R and delta values here are not used.
Run it once per delta of interest with its own --outdir.

Added 23 Sep 2026: --decomp-only skips the screening learners, the DML
estimate and the R bootstrap (learner "none" in the records), which is all
a mean-error arm needs and turns three hours into minutes; it is refused
for the exact arm, whose purpose is the full comparison. --age2565 (mean-
error arm only) restricts the ACS sample to ages 25 to 65 and propagates
the earners anchor from results/kappa_workers.json, so that a delta
estimated on the PIAAC earners sample is applied to the matching ACS
sample:

    python scripts/12_run_arms.py --arm meanerror --delta 0.580 --age2565 --decomp-only \
        --outdir results/grid_meanerror_lit_workers --root acs_data

Added 28 Sep 2026 (REGISTERED_CHANGES R14 and R15): --share lowers women's
years-coded schooling by that share of the cell's own person-weighted
schooling lead instead of a fixed --delta (the two are mutually exclusive;
the lead and the applied shift are recorded in each cell's load_options);
--skill composite reads the composite anchor from a kappa file written with
02_estimate_kappa.py --construct composite; --extra-kappa adds further
reliabilities at which the corrected residual is stored exactly:

    python scripts/12_run_arms.py --arm meanerror --share 0.807388 --decomp-only         --outdir results/grid_prop_lit --root acs_data
    python scripts/12_run_arms.py --arm meanerror --delta 0.4 --skill composite         --kappa results/kappa_composite.json --kappa-workers results/kappa_composite_workers.json         --extra-kappa 0.186729 --decomp-only --outdir results/grid_meanerror_composite --root acs_data

Resumable; rerun the same command to retry failures. Writes
results/grid_<arm>/{STATE}_{YEAR}.json.
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _init_worker():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "2"


def _one(args):
    (root, state, year, kappa_point, outdir, chash, arm, load_kwargs, covsets, learners, extra,
     decomp_only) = args
    from study2lib.cellrun import run_cell
    from study2lib.runlog import write_json
    res = run_cell(root, state, year, kappa_point, learners=learners,
                   covsets=covsets, load_kwargs=load_kwargs, extra_kappas=extra,
                   decomp_only=decomp_only)
    res["arm"] = arm
    res["arm_options"] = load_kwargs
    res["decomp_only"] = bool(decomp_only)
    if arm == "meanerror":
        shift_txt = (f"{-load_kwargs['shift_female_schl']:.4f} years" if "shift_female_schl" in load_kwargs
                     else f"{load_kwargs['shift_female_share']:.6f} times the cell's schooling lead "
                          f"({-res['load_options'].get('shift_female_schl_applied', 0.0):.4f} years here)")
        res["note"] = ("decomposition fields only; women's years-coded schooling lowered by "
                       f"{shift_txt} before the correction; "
                       "R and delta values of this arm are not to be used"
                       + ("; ages 25 to 65 with the earners anchor" if "age_range" in load_kwargs else "")
                       + ("; learners, DML and the R bootstrap skipped" if decomp_only else ""))
    res["config_hash"] = chash
    write_json(os.path.join(outdir, f"{state}_{year}.json"), res)
    bad = sum(1 for s in res["spec_results"] if s.get("eiv_correctable") is False)
    return (state, year, res["timing_s"], len(res["spec_results"]), len(res["skipped_arms"]), bad)


def main():
    from study2lib.acs import STATE_FIPS
    from study2lib import specs
    from study2lib.runlog import config_hash
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(list(specs.ARMS) + ["meanerror"]))
    ap.add_argument("--root", default="acs_data")
    ap.add_argument("--kappa", default="results/kappa.json")
    ap.add_argument("--kappa-workers", default="results/kappa_workers.json")
    ap.add_argument("--skill", default="lit", choices=["lit", "num", "composite"])
    ap.add_argument("--learners", nargs="*", default=["logistic"])
    ap.add_argument("--delta", type=float, default=None,
                    help="meanerror arm: sex difference in mean proxy error in years "
                         "(positive = women's schooling overstates); subtracted from "
                         "women's years-coded schooling")
    ap.add_argument("--share", type=float, default=None,
                    help="meanerror arm: lower women's years-coded schooling by this share of "
                         "the cell's own person-weighted schooling lead (R14); excludes --delta")
    ap.add_argument("--extra-kappa", nargs="*", type=float, default=[],
                    help="further reliabilities at which the corrected residual is stored exactly")
    ap.add_argument("--states", nargs="*", default=sorted(STATE_FIPS))
    ap.add_argument("--years", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--decomp-only", action="store_true",
                    help="decomposition fields only: no learners, DML or R bootstrap "
                         "(refused for the exact arm)")
    ap.add_argument("--age2565", action="store_true",
                    help="meanerror arm: ACS ages 25 to 65 with the earners anchor, for a "
                         "delta estimated on the PIAAC earners sample")
    a = ap.parse_args()
    if a.decomp_only and a.arm == "exact":
        print("--decomp-only is refused for the exact arm (it exists to compare full cells)")
        return 1
    if a.age2565 and a.arm != "meanerror":
        print("--age2565 belongs to the meanerror arm; use --arm age2565 for the plain arm")
        return 1

    k_all = json.load(open(a.kappa, encoding="utf-8"))[a.skill]["overall"]["kappa"]
    k_work = json.load(open(a.kappa_workers, encoding="utf-8"))[a.skill]["overall"]["kappa"]
    extra = sorted({float(k_all), float(k_work)} | {float(k) for k in a.extra_kappa})  # full precision; keyed to 6 decimals in the cell

    if a.arm == "meanerror":
        if (a.delta is None) == (a.share is None):
            print("meanerror arm needs exactly one of --delta (sex difference in mean proxy error, "
                  "years, from results/piaac_validity.json) and --share (that delta divided by "
                  "the PIAAC schooling lead)")
            return 1
        if a.outdir is None:
            print("meanerror arm needs --outdir (one directory per delta)")
            return 1
        load_kwargs = ({"schooling": "years", "shift_female_schl": -float(a.delta)} if a.delta is not None
                       else {"schooling": "years", "shift_female_share": float(a.share)})
        if a.age2565:
            load_kwargs["age_range"] = (25, 65)
    else:
        load_kwargs = dict(specs.ARMS[a.arm])
    if "age_range" in load_kwargs:
        load_kwargs["age_range"] = tuple(load_kwargs["age_range"])
    kappa_point = k_work if (a.arm == "age2565" or a.age2565) else k_all
    covsets = ({"extended_occ": specs.COVARIATE_SETS["extended"]} if a.arm == "occ"
               else dict(specs.COVARIATE_SETS))
    outdir = a.outdir or f"results/grid_{a.arm}"
    years = a.years or specs.PRIORITY_YEARS
    chash = config_hash({"arm": a.arm, "load_kwargs": {k: (list(v) if isinstance(v, tuple) else v)
                                                        for k, v in load_kwargs.items()},
                         "covsets": covsets, "learners": (["none"] if a.decomp_only else a.learners),
                         "kappa": round(kappa_point, 6), "skill": a.skill,
                         "extra_kappas": [round(k, 6) for k in extra], "outcomes": specs.OUTCOME_DEFS,
                         "seed": specs.PRIMARY_SEED, "grid": specs.KAPPA_GRID,
                         "schl_to_years": specs.SCHL_TO_YEARS if a.arm == "years" else None,
                         "occp_groups": specs.OCCP_MAJOR_GROUPS if a.arm == "occ" else None,
                         "tabpfn_model_version": specs.TABPFN_MODEL_VERSION})
    os.makedirs(outdir, exist_ok=True)

    todo = []
    for y in years:
        for s in a.states:
            p = os.path.join(outdir, f"{s}_{y}.json")
            if os.path.exists(p):
                try:
                    if json.load(open(p, encoding="utf-8")).get("config_hash") == chash:
                        continue
                except Exception:
                    pass
            todo.append((a.root, s, y, kappa_point, outdir, chash, a.arm, load_kwargs,
                         covsets, a.learners, extra, a.decomp_only))
    print(f"arm {a.arm}: options {load_kwargs}; covsets {list(covsets)}; "
          f"learners {'none (decomposition only)' if a.decomp_only else a.learners}; "
          f"kappa_point {kappa_point:.4f}; exact anchors {[round(k, 4) for k in extra]}; "
          f"{len(todo)} cells to run, {len(years)*len(a.states)-len(todo)} already current; "
          f"workers {a.workers}; outdir {outdir}")

    done = fail = noncorr = 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker,
                             max_tasks_per_child=1) as ex:
        futures = {ex.submit(_one, t): t for t in todo}
        for f in as_completed(futures):
            _, s, y, *_ = futures[f]
            try:
                st, yr, secs, nspec, nskip, bad = f.result()
                done += 1
                noncorr += bad
                print(f"  [{done+fail}/{len(todo)}] {st} {yr}  {secs:7.1f}s  {nspec} specs"
                      + (f"  ({nskip} arms skipped)" if nskip else "")
                      + (f"  [{bad} not correctable at the anchor]" if bad else ""))
            except Exception as e:
                fail += 1
                print(f"  [{done+fail}/{len(todo)}] {s} {y}  FAILED  {e!r}")
    print(f"\n{done} cells written, {fail} failed"
          + ("; rerun to retry failures" if fail else ""))
    if noncorr:
        print(f"{noncorr} specification(s) are not correctable at this anchor: the assumed "
              f"reliability is below the design's admissible minimum (eiv_min_kappa in each "
              f"record). The uncorrected decomposition and R are stored; the corrected fields "
              f"are null by design, and the sweep carries values above the threshold only.")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
