#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harmonised premarket arm (REGISTERED_CHANGES R1, 15 Sep 2026, post-results).

Runs the premarket covariates on the WKHP/COW-complete sample in every
state-year cell, so the premarket-vs-extended verdict flip can be decomposed
into a sample component (premarket on its own sample vs premarket on the
harmonised sample) and a conditioning component (premarket vs extended on
the same harmonised sample). No bootstrap: the harmonised arm is point
estimates by design, like every non-primary arm.

    python scripts/06_harmonised.py --root acs_data --kappa results/kappa.json --workers 3

Resumable; rerun the same command to retry failures until "0 failed".
Writes results/grid_h/{STATE}_{YEAR}.json; never touches results/grid.
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _init_worker():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "2"


def _one(args):
    root, state, year, kappa_point, outdir, chash = args
    from study2lib import specs
    from study2lib.cellrun import run_cell
    from study2lib.runlog import write_json
    res = run_cell(root, state, year, kappa_point,
                   covsets={"premarket_h": specs.COVARIATE_SETS["premarket"]},
                   sample_requires=specs.HARMONISED_SAMPLE_REQUIRES)
    res["arm"] = "harmonised_premarket"
    res["sample_requires"] = specs.HARMONISED_SAMPLE_REQUIRES
    res["config_hash"] = chash
    path = os.path.join(outdir, f"{state}_{year}.json")
    write_json(path, res)
    return state, year, res["timing_s"], len(res["spec_results"])


def main():
    from study2lib.acs import STATE_FIPS
    from study2lib import specs
    from study2lib.runlog import config_hash
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="acs_data")
    ap.add_argument("--kappa", required=True)
    ap.add_argument("--skill", default="lit", choices=["lit", "num"])
    ap.add_argument("--states", nargs="*", default=sorted(STATE_FIPS))
    ap.add_argument("--years", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--outdir", default="results/grid_h")
    a = ap.parse_args()

    kap = json.load(open(a.kappa, encoding="utf-8"))[a.skill]["overall"]["kappa"]
    years = a.years or specs.PRIORITY_YEARS
    chash = config_hash({"arm": "harmonised_premarket",
                         "covariates": specs.COVARIATE_SETS["premarket"],
                         "sample_requires": specs.HARMONISED_SAMPLE_REQUIRES,
                         "kappa": round(kap, 6), "skill": a.skill,
                         "outcomes": specs.OUTCOME_DEFS,
                         "learners": specs.LEARNERS, "seed": specs.PRIMARY_SEED,
                         "grid": specs.KAPPA_GRID,
                         "tabpfn_model_version": specs.TABPFN_MODEL_VERSION})
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
            todo.append((a.root, s, y, kap, a.outdir, chash))
    print(f"harmonised arm: kappa({a.skill}) = {kap:.4f}; {len(todo)} cells to run, "
          f"{len(years)*len(a.states)-len(todo)} already current; workers {a.workers}")

    done = fail = 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker,
                             max_tasks_per_child=1) as ex:
        futures = {ex.submit(_one, t): t for t in todo}
        for f in as_completed(futures):
            _, s, y, *_ = futures[f]
            try:
                st, yr, secs, nspec = f.result()
                done += 1
                print(f"  [{done+fail}/{len(todo)}] {st} {yr}  {secs:7.1f}s  {nspec} specs")
            except Exception as e:
                fail += 1
                print(f"  [{done+fail}/{len(todo)}] {s} {y}  FAILED  {e!r}")
    print(f"\n{done} cells written, {fail} failed"
          + ("; rerun to retry failures" if fail else ""))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
