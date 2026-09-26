#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the state-year grid in parallel. Resumable: cells whose result file
already matches the current config hash are skipped.

    python scripts/04_run_grid.py --root acs_data --kappa results/kappa.json
    python scripts/04_run_grid.py --root acs_data --kappa results/kappa.json --full
    python scripts/04_run_grid.py ... --states WY CO CA --years 2022 --workers 8
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _init_worker():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "2"


def _one(args):
    root, state, year, kappa_point, outdir, chash = args
    from study2lib.cellrun import run_cell
    from study2lib.runlog import write_json
    res = run_cell(root, state, year, kappa_point)
    res["config_hash"] = chash
    path = os.path.join(outdir, f"{state}_{year}.json")
    write_json(path, res)
    return state, year, res["timing_s"], len(res["spec_results"]), len(res["skipped_arms"])


def main():
    from study2lib.acs import STATE_FIPS
    from study2lib import specs
    from study2lib.runlog import config_hash
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="acs_data")
    ap.add_argument("--kappa", required=True, help="results/kappa.json from step 02")
    ap.add_argument("--skill", default="lit", choices=["lit", "num"])
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--states", nargs="*", default=sorted(STATE_FIPS))
    ap.add_argument("--years", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 8) - 2))
    ap.add_argument("--outdir", default="results/grid")
    a = ap.parse_args()

    kap = json.load(open(a.kappa, encoding="utf-8"))[a.skill]["overall"]["kappa"]
    years = a.years or (specs.FULL_YEARS if a.full else specs.PRIORITY_YEARS)
    chash = config_hash({"kappa": round(kap, 6), "skill": a.skill,
                         "outcomes": specs.OUTCOME_DEFS,
                         "covsets": specs.COVARIATE_SETS,
                         "learners": specs.LEARNERS, "seed": specs.PRIMARY_SEED,
                         "n_boot": specs.N_BOOT, "grid": specs.KAPPA_GRID,
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
    print(f"kappa({a.skill}) = {kap:.4f}; {len(todo)} cells to run, "
          f"{len(years)*len(a.states)-len(todo)} already current; workers {a.workers}")

    done = fail = 0
    # max_tasks_per_child=1: every cell gets a fresh process and a fresh CUDA
    # context. Without it, a single CUDA launch failure leaves a worker with a
    # sticky poisoned context that instantly fails every cell it is handed
    # while healthy workers grind, so one fault devours the queue (observed on
    # the first T6 attempt, 14 Sep 2026: 1 real fault at LA 2017, then 19
    # instant sticky failures among the next 37 completions).
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker,
                             max_tasks_per_child=1) as ex:
        futures = {ex.submit(_one, t): t for t in todo}
        for f in as_completed(futures):
            _, s, y, *_ = futures[f]
            try:
                st, yr, secs, nspec, nskip = f.result()
                done += 1
                print(f"  [{done+fail}/{len(todo)}] {st} {yr}  {secs:7.1f}s  "
                      f"{nspec} specs" + (f"  ({nskip} arms skipped)" if nskip else ""))
            except Exception as e:
                fail += 1
                print(f"  [{done+fail}/{len(todo)}] {s} {y}  FAILED  {e!r}")
    print(f"\n{done} cells written, {fail} failed"
          + ("; rerun to retry failures (resume skips finished cells)" if fail else ""))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
