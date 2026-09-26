#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Joint weighted bootstrap on the registered primary arm
(REGISTERED_CHANGES R2, 15 Sep 2026, post-results).

Per state-year cell, B seeded draws sample persons within sex with
probability proportional to the person weight PWGTP (a pps bootstrap; the
drawn rows then enter every estimator with unit weight), and each draw
computes four statistics jointly: delta_label, delta_merit (out-of-fold
logistic refit), R, and the EIV-corrected residual at the anchor kappa.
Percentile intervals for all four; positive-definiteness failures of
S_XX - Lambda are caught and counted per cell, never silently dropped.
Also exports the weighted group moments (mean and variance of SCHL and
AGEP by sex, group weight shares) needed for the analytic H2.2 flip
thresholds, and checks the stored grid cell's R for consistency.

    python scripts/07_boot_primary.py --root acs_data --kappa results/kappa.json --workers 3

Resumable; writes results/boot_primary/{STATE}_{YEAR}.json.

Bootstrap v2 (REGISTERED_CHANGES R12, 22 Sep 2026, post-results):

    python scripts/07_boot_primary.py --root acs_data --kappa results/kappa.json \
        --folds-by-person --kappa-draw --outdir results/boot_primary_v2 --workers 3

--folds-by-person assigns the out-of-fold split by original person, so a
person drawn twice never sits in both the training and the evaluation
partition (v1 leaked duplicates across folds, stated as a limitation).
--kappa-draw draws the reliability in every draw from a normal with the
anchor's replicate-based standard error, truncated to the sweep range, from
a separate seeded stream so that the person draws are identical to v1; the
kappa-fixed intervals are still stored alongside the kappa-drawn ones.
The v1 outputs are never overwritten: v2 goes to its own directory.

Mean-error and years-coded arms (R11; 22 Sep 2026, evening): --schooling
years applies the crosswalk, --shift-female-years D lowers women's
years-coded schooling by D before the decomposition (D = the PIAAC
estimate of the sex difference in mean proxy error). With either option
the grid consistency check compares against --griddir only if it is given
explicitly; use --griddir results/grid_years for the years arm and
--no-grid-check for a shifted arm, whose R is not the reported one.

--shift-se S (23 Sep 2026) draws the shift itself in every draw from a
normal with mean --shift-female-years and SD S (the replicate SE of the
PIAAC estimate), from its own seeded stream, and stores the corrected
residual at the drawn shift (and at the drawn shift and drawn kappa when
--kappa-draw is on). It also stores the uncorrected residual with the shift
undone, i.e. the as-reported gap, so that the interval of the net effect of
both corrections relative to the reported gap is available per market:

    python scripts/07_boot_primary.py --root acs_data --kappa results/kappa.json \
        --folds-by-person --kappa-draw --schooling years --shift-female-years 0.297 \
        --shift-se 0.069 --no-grid-check --outdir results/boot_meanerror_lit --workers 3

    python scripts/07_boot_primary.py --root acs_data --kappa results/kappa.json \
        --folds-by-person --kappa-draw --schooling years --shift-female-years 0.297 \
        --no-grid-check --outdir results/boot_meanerror_lit --workers 3
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

B_DRAWS = 500          # matches the registered N_BOOT
ALPHA = 0.05


def _init_worker():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "2"


def _one(args):
    import numpy as np
    from study2lib import specs
    from study2lib.acs import load_cell
    from study2lib.decomposition import twofold_neumark
    from study2lib.cellrun import _corrected_unexplained
    from study2lib.mrr import mrr
    from study2lib.runlog import write_json
    (root, state, year, kappa_point, outdir, griddir, chash, folds_by_person, kappa_se,
     load_kwargs, shift_se) = args
    from study2lib.learners import folds_by_person as _folds
    shift_base = -float(load_kwargs.get("shift_female_schl", 0.0))   # years lowered, as loaded

    od, cs_name, lr = specs.BOOT_ARMS[0]
    cell = load_cell(root, state, year, specs.COVARIATE_SETS[cs_name], od, **load_kwargs)
    X, names = cell["X"], cell["names"]
    y, y_cont, g, w = cell["y"], cell["y_cont"], cell["g"], cell["w"]
    ly = np.log(y_cont)

    def stats(Xs, ys, lys, gs, ws, folds=None, kappa_b=None):
        m = mrr(Xs, ys, gs, ws, seed=specs.PRIMARY_SEED, method=lr, folds=folds)
        a_, b_ = gs == 1, gs == 0
        dec = twofold_neumark(Xs[a_], lys[a_], Xs[b_], lys[b_], ws[a_], ws[b_])
        try:
            corr = _corrected_unexplained(Xs, names, lys, gs, ws, kappa_point)[2]
            pd_fail = 0
        except np.linalg.LinAlgError:
            corr, pd_fail = None, 1
        corr_k = None
        if kappa_b is not None:
            try:
                corr_k = _corrected_unexplained(Xs, names, lys, gs, ws, kappa_b)[2]
            except np.linalg.LinAlgError:
                pd_fail += 1
        return (m["delta_label"], m["delta_merit"], m["R"],
                dec["unexplained"], corr, pd_fail, corr_k)

    j_schl = names.index("SCHL")

    def shifted(Xs, gs, extra_years):
        """Copy of Xs with women's schooling lowered by extra_years more."""
        Z = Xs.copy()
        Z[gs == 1, j_schl] -= extra_years
        return Z

    def decomp(Xs, lys, gs, ws, kappas):
        """Uncorrected unexplained component and the corrected one at each kappa
        (None where the EIV solve is not positive definite)."""
        a_, b_ = gs == 1, gs == 0
        dec = twofold_neumark(Xs[a_], lys[a_], Xs[b_], lys[b_], ws[a_], ws[b_])
        out = []
        for k in kappas:
            try:
                out.append(_corrected_unexplained(Xs, names, lys, gs, ws, k)[2])
            except np.linalg.LinAlgError:
                out.append(None)
        return dec["unexplained"], out

    # point estimates on the full weighted sample (identical to the grid cell)
    p_lab, p_mer, p_R, p_unc, p_corr, _, _ = stats(X, y, ly, g, w)
    p_unc0 = decomp(shifted(X, g, -shift_base), ly, g, w, [])[0] if shift_base else p_unc

    from study2lib.acs import STATE_FIPS
    seed_offset = int(STATE_FIPS[state]) * 10000 + int(year)   # deterministic
    rng = np.random.default_rng(specs.PRIMARY_SEED + 700000 + seed_offset)
    idx_f, idx_m = np.flatnonzero(g == 1), np.flatnonzero(g == 0)
    prob_f = w[idx_f] / w[idx_f].sum()
    prob_m = w[idx_m] / w[idx_m].sum()
    ones = None
    draws = {"delta_label": [], "delta_merit": [], "R": [],
             "uncorrected_residual": [], "corrected_residual": [],
             "deepening": []}
    if kappa_se is not None:
        draws["kappa_drawn"] = []
        draws["corrected_residual_kappa_drawn"] = []
        draws["deepening_kappa_drawn"] = []
        rng_k = np.random.default_rng(specs.PRIMARY_SEED + 710000 + seed_offset)
    if shift_se is not None:
        draws["shift_drawn"] = []
        draws["uncorrected_residual_unshifted"] = []
        draws["corrected_residual_shift_drawn"] = []
        draws["net_vs_reported"] = []
        draws["net_vs_reported_shift_drawn"] = []
        if kappa_se is not None:
            draws["corrected_residual_both_drawn"] = []
            draws["net_vs_reported_both_drawn"] = []
        rng_d = np.random.default_rng(specs.PRIMARY_SEED + 730000 + seed_offset)
    pd_failures = 0
    for b in range(B_DRAWS):
        take = np.concatenate([rng.choice(idx_f, len(idx_f), replace=True, p=prob_f),
                               rng.choice(idx_m, len(idx_m), replace=True, p=prob_m)])
        if ones is None or len(ones) != len(take):
            ones = np.ones(len(take))
        folds = (_folds(take, n_folds=5, seed=specs.PRIMARY_SEED + 720000 + seed_offset + b)
                 if folds_by_person else None)
        kappa_b = None
        if kappa_se is not None:
            kappa_b = float(np.clip(rng_k.normal(kappa_point, kappa_se),
                                    specs.KAPPA_GRID[0], specs.KAPPA_GRID[-1]))
        dl, dm, dR, du, dc, pf, dck = stats(X[take], y[take], ly[take], g[take], ones,
                                            folds=folds, kappa_b=kappa_b)
        pd_failures += pf
        draws["delta_label"].append(dl); draws["delta_merit"].append(dm)
        draws["R"].append(dR); draws["uncorrected_residual"].append(du)
        if dc is not None:
            draws["corrected_residual"].append(dc)
            draws["deepening"].append(dc - du)   # negative = deepened
        if kappa_b is not None:
            draws["kappa_drawn"].append(kappa_b)
            if dck is not None:
                draws["corrected_residual_kappa_drawn"].append(dck)
                draws["deepening_kappa_drawn"].append(dck - du)
        if shift_se is not None:
            d_b = float(rng_d.normal(shift_base, shift_se))
            Xt, lt, gt = X[take], ly[take], g[take]
            du0 = decomp(shifted(Xt, gt, -shift_base), lt, gt, ones, [])[0]   # as reported
            kap_list = [kappa_point] + ([kappa_b] if kappa_b is not None else [])
            _, cs = decomp(shifted(Xt, gt, d_b - shift_base), lt, gt, ones, kap_list)
            draws["shift_drawn"].append(d_b)
            draws["uncorrected_residual_unshifted"].append(du0)
            if dc is not None:
                draws["net_vs_reported"].append(dc - du0)
            if cs[0] is not None:
                draws["corrected_residual_shift_drawn"].append(cs[0])
                draws["net_vs_reported_shift_drawn"].append(cs[0] - du0)
            else:
                pd_failures += 1
            if kappa_b is not None:
                if cs[1] is not None:
                    draws["corrected_residual_both_drawn"].append(cs[1])
                    draws["net_vs_reported_both_drawn"].append(cs[1] - du0)
                else:
                    pd_failures += 1

    def interval(vals):
        v = np.asarray(vals, float)
        lo, hi = np.quantile(v, [ALPHA / 2, 1 - ALPHA / 2])
        return {"lo": float(lo), "hi": float(hi), "n_draws": int(len(v)),
                "share_gt0": float(np.mean(v > 0)), "share_lt0": float(np.mean(v < 0))}

    # weighted group moments for the analytic H2.2 flip thresholds
    def moments(col):
        j = names.index(col)
        out = {}
        for tag, mask in (("female", g == 1), ("male", g == 0)):
            mu = float(np.average(X[mask, j], weights=w[mask]))
            va = float(np.average((X[mask, j] - mu) ** 2, weights=w[mask]))
            out[tag] = {"mean": mu, "var": va}
        return out

    # consistency: the stored grid cell's primary R must match the point here
    stored = os.path.join(griddir, f"{state}_{year}.json") if griddir else None
    stored_R = None
    if stored and os.path.exists(stored):
        for r in json.load(open(stored, encoding="utf-8"))["spec_results"]:
            if (r["outcome_def"], r["covariate_set"], r["learner"]) == (od, cs_name, lr):
                stored_R = r["R"]
    consistent = (griddir is None) or ((stored_R is not None) and abs(stored_R - p_R) < 1e-9)

    res = {"state": state, "year": year, "arm": [od, cs_name, lr],
           "seed": specs.PRIMARY_SEED, "seed_offset": seed_offset,
           "kappa_point": kappa_point,
           "resampling": "pps within sex (p proportional to PWGTP), unit "
                         "weights on drawn rows; point estimates weighted"
                         + ("; folds assigned by original person" if folds_by_person else "")
                         + (f"; kappa drawn per draw, normal(kappa_point, {kappa_se:.4f}) "
                            f"truncated to the sweep range" if kappa_se is not None else "")
                         + (f"; shift drawn per draw, normal({shift_base:.4f}, {shift_se:.4f}) years"
                            if shift_se is not None else ""),
           "version": 2 if (folds_by_person or kappa_se is not None or shift_se is not None) else 1,
           "folds_by_person": bool(folds_by_person),
           "kappa_se": kappa_se,
           "shift_female_years": shift_base if shift_base else None,
           "shift_se": shift_se,
           "load_options": cell.get("options", {}),
           "point": {"delta_label": p_lab, "delta_merit": p_mer, "R": p_R,
                     "uncorrected_residual": p_unc,
                     "corrected_residual": p_corr,
                     "deepening": (None if p_corr is None else p_corr - p_unc),
                     "uncorrected_residual_unshifted": p_unc0,
                     "net_vs_reported": (None if p_corr is None else p_corr - p_unc0)},
           "intervals": {k: interval(v) for k, v in draws.items()},
           "pd_failures": pd_failures,
           "group_moments": {c: moments(c) for c in ("SCHL", "AGEP")},
           "group_weight_share_female": float(w[g == 1].sum() / w.sum()),
           "n": cell["n"], "n_female": int((g == 1).sum()),
           "n_male": int((g == 0).sum()),
           "stored_grid_R": stored_R, "consistent_with_grid": bool(consistent),
           "config_hash": chash}
    write_json(os.path.join(outdir, f"{state}_{year}.json"), res)
    return state, year, consistent, pd_failures


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
    ap.add_argument("--griddir", default="results/grid")
    ap.add_argument("--outdir", default="results/boot_primary")
    ap.add_argument("--folds-by-person", action="store_true",
                    help="v2: out-of-fold split by original person (R12)")
    ap.add_argument("--kappa-draw", action="store_true",
                    help="v2: draw kappa per draw from its replicate-based normal (R12)")
    ap.add_argument("--schooling", default="code", choices=["code", "years"],
                    help="years: apply the SCHL_TO_YEARS crosswalk (R4)")
    ap.add_argument("--shift-female-years", type=float, default=0.0,
                    help="lower women's years-coded schooling by this many years (R11)")
    ap.add_argument("--shift-se", type=float, default=None,
                    help="draw the shift per draw from normal(--shift-female-years, this SD) (R11)")
    ap.add_argument("--no-grid-check", action="store_true",
                    help="skip the consistency check against a stored grid cell")
    a = ap.parse_args()
    if a.shift_se is not None and not a.shift_female_years:
        print("--shift-se needs --shift-female-years (the mean of the drawn shift)")
        return 1
    load_kwargs = {}
    if a.schooling == "years":
        load_kwargs["schooling"] = "years"
    if a.shift_female_years:
        if a.schooling != "years":
            print("--shift-female-years needs --schooling years (the shift is in years)")
            return 1
        load_kwargs["shift_female_schl"] = -float(a.shift_female_years)
    griddir = None if a.no_grid_check else a.griddir
    if load_kwargs and not a.no_grid_check and a.griddir == "results/grid":
        print("a years-coded or shifted arm cannot be checked against results/grid; "
              "pass --griddir results/grid_years (years arm) or --no-grid-check")
        return 1
    if load_kwargs and a.outdir in ("results/boot_primary", "results/boot_primary_v2"):
        print("a years-coded or shifted arm needs its own --outdir")
        return 1

    kfile = json.load(open(a.kappa, encoding="utf-8"))[a.skill]["overall"]
    kap = kfile["kappa"]
    kappa_se = float(kfile["se"]) if a.kappa_draw else None
    if (a.folds_by_person or a.kappa_draw) and a.outdir == "results/boot_primary":
        print("v2 options need their own --outdir (for example results/boot_primary_v2); "
              "the v1 outputs are not overwritten")
        return 1
    years = a.years or specs.PRIORITY_YEARS
    cfg = {"arm": "boot_primary_joint", "B": B_DRAWS,
           "resampling": "pps_within_sex",
           "kappa": round(kap, 6), "skill": a.skill,
           "boot_arm": specs.BOOT_ARMS[0],
           "seed": specs.PRIMARY_SEED,
           "tabpfn_model_version": specs.TABPFN_MODEL_VERSION}
    if a.folds_by_person or a.kappa_draw:
        cfg.update({"version": 2, "folds_by_person": a.folds_by_person,
                    "kappa_se": (round(kappa_se, 6) if kappa_se is not None else None)})
    if load_kwargs:
        cfg["load_kwargs"] = load_kwargs
        if a.schooling == "years":
            cfg["schl_to_years"] = specs.SCHL_TO_YEARS
    if a.shift_se is not None:
        cfg["shift_se"] = round(float(a.shift_se), 6)
    chash = config_hash(cfg)
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
            todo.append((a.root, s, y, kap, a.outdir, griddir, chash,
                         a.folds_by_person, kappa_se, load_kwargs, a.shift_se))
    print(f"joint bootstrap{' v2' if (a.folds_by_person or a.kappa_draw) else ''}: "
          f"kappa({a.skill}) = {kap:.4f}"
          + (f" (se {kappa_se:.4f}, drawn per draw)" if kappa_se is not None else "")
          + (", folds by person" if a.folds_by_person else "")
          + (f"; load options {load_kwargs}" if load_kwargs else "")
          + (f"; shift drawn per draw (se {a.shift_se:.4f} years)" if a.shift_se is not None else "")
          + f"; {len(todo)} cells to run, "
          f"{len(years)*len(a.states)-len(todo)} already current; workers {a.workers}; "
          f"outdir {a.outdir}")

    done = fail = 0
    inconsistent = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker,
                             max_tasks_per_child=1) as ex:
        futures = {ex.submit(_one, t): t for t in todo}
        for f in as_completed(futures):
            _, s, y, *_ = futures[f]
            try:
                st, yr, ok, pdf = f.result()
                done += 1
                note = "" if ok else "  !! R differs from stored grid cell"
                note += f"  ({pdf} PD failures)" if pdf else ""
                print(f"  [{done+fail}/{len(todo)}] {st} {yr}{note}")
                if not ok:
                    inconsistent.append(f"{st}_{yr}")
            except Exception as e:
                fail += 1
                print(f"  [{done+fail}/{len(todo)}] {s} {y}  FAILED  {e!r}")
    print(f"\n{done} cells written, {fail} failed"
          + ("; rerun to retry failures" if fail else ""))
    if inconsistent:
        print(f"INCONSISTENT WITH GRID (stop and report): {inconsistent}")
        return 1
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
