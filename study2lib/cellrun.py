# -*- coding: utf-8 -*-
"""Everything estimated on one state-year cell, across the registered
specification grid. This is the unit of work the grid runner parallelises."""
from __future__ import annotations
import time
import numpy as np
from . import specs
from .acs import load_cell
from .decomposition import twofold_neumark
from .eiv import eiv_correct, min_admissible_kappa
from .mrr import mrr
from .dml import dml_gap
from .bootstrap import stratified_bootstrap
from .learners import ArmUnavailable


def _corrected_unexplained(X, names, y_cont, g, w, kappa):
    """EIV-corrected explained/unexplained split of the raw continuous gap
    under the same reference structure as the uncorrected decomposition:
    the pooled regression including a group indicator whose coefficient is
    discarded (Neumark, as in Study 1). Only the proxy column carries the
    reliability kappa; demographics and the group indicator carry 1.
    At kappa = 1 this reproduces twofold_neumark exactly."""
    kap = np.array([kappa if n == specs.PROXY_COLUMN else 1.0 for n in names] + [1.0])
    Xg = np.column_stack([X, g.astype(float)])
    beta_full = eiv_correct(Xg, y_cont, kap, w)
    beta_star = np.concatenate([beta_full[:1], beta_full[1:-1]])   # drop the group dummy
    a, b = g == 1, g == 0
    xbar_a = np.concatenate([[1.0], np.average(X[a], axis=0, weights=w[a])])
    xbar_b = np.concatenate([[1.0], np.average(X[b], axis=0, weights=w[b])])
    raw = float(np.average(y_cont[a], weights=w[a]) - np.average(y_cont[b], weights=w[b]))
    explained = float((xbar_a - xbar_b) @ beta_star)
    return raw, explained, raw - explained


def _eiv_min_kappa(X, names, g, w):
    """Smallest reliability of the proxy column for which the corrected
    system exists in this design (25 Sep 2026; see eiv.min_admissible_kappa).
    Computed on the same augmented matrix the correction uses."""
    Xg = np.column_stack([X, g.astype(float)])
    return min_admissible_kappa(Xg, names.index(specs.PROXY_COLUMN), w)


def _corrected_or_none(X, names, y_cont, g, w, kappa):
    """The corrected unexplained component, or None where the assumed
    reliability is below the design's admissible minimum. A None is a
    diagnosis, never a zero: the cell records that the correction does not
    exist at that kappa and the uncorrected decomposition stands."""
    try:
        return _corrected_unexplained(X, names, y_cont, g, w, kappa)[2]
    except np.linalg.LinAlgError:
        return None


def run_cell(root, state, year, kappa_point, seed=None, n_boot=None,
             learners=None, kappa_grid=None, covsets=None, sample_requires=None,
             load_kwargs=None, extra_kappas=None, decomp_only=False):
    """Returns the cell's full result dict (JSON-serialisable).

    covsets and sample_requires (added 15 Sep 2026 for the harmonised arm)
    default to the registered grid and to no extra sample restriction, so
    every call without them reproduces the T6 outputs bit for bit.

    load_kwargs (22 Sep 2026, arms R4-R12): keyword options passed through to
    load_cell (schooling, income_var, fulltime, occupation, adjinc,
    weighted_median_threshold, age_range, shift_female_schl); None means
    the registered behaviour. extra_kappas: reliabilities at which the
    corrected residual is also stored exactly (eiv_unexplained_at), in
    addition to kappa_point and the sweep.

    decomp_only (23 Sep 2026): True skips the screening learners, the DML
    estimate and the R bootstrap and writes one record per outcome and
    covariate set with learner "none" and the decomposition fields only.
    For the mean-error arms, whose R is not reported, this turns hours into
    minutes; the decomposition fields are computed by the same code and are
    identical to a full run's."""
    seed = specs.PRIMARY_SEED if seed is None else seed
    n_boot = specs.N_BOOT if n_boot is None else n_boot
    learners = learners or specs.LEARNERS
    kappa_grid = kappa_grid or specs.KAPPA_GRID
    covsets = covsets or specs.COVARIATE_SETS
    load_kwargs = dict(load_kwargs or {})
    extra_kappas = list(extra_kappas or [])
    t0 = time.time()
    results, skips = [], []
    load_options = None

    for od in specs.OUTCOME_DEFS:
        for cs_name, cs_cols in covsets.items():
            cell = load_cell(root, state, year, cs_cols, od,
                             sample_requires=sample_requires, **load_kwargs)
            load_options = cell.get("options", {})
            X, names = cell["X"], cell["names"]
            y, y_cont, g, w = cell["y"], cell["y_cont"], cell["g"], cell["w"]
            n_f, n_m = int((g == 1).sum()), int((g == 0).sum())
            powered = n_f >= specs.MIN_GROUP_N and n_m >= specs.MIN_GROUP_N

            # decomposition on log income (raw gap in log points), once per od x cs
            ly = np.log(y_cont)
            dec = twofold_neumark(X[g == 1], ly[g == 1], X[g == 0], ly[g == 0],
                                  w[g == 1], w[g == 0])
            kmin = _eiv_min_kappa(X, names, g, w)
            unexpl_c = _corrected_or_none(X, names, ly, g, w, kappa_point)
            sweep = [{"kappa": k,
                      "unexplained": _corrected_or_none(X, names, ly, g, w, k)}
                     for k in kappa_grid]
            exact = {str(round(k, 6)): _corrected_or_none(X, names, ly, g, w, k)
                     for k in extra_kappas}
            if decomp_only:
                results.append({"outcome_def": od, "covariate_set": cs_name, "learner": "none",
                                "n": cell["n"], "n_female": n_f, "n_male": n_m,
                                "powered": powered,
                                "delta_label": None, "delta_merit": None, "R": None,
                                "decomp_raw_logpts": dec["raw"],
                                "decomp_explained": dec["explained"],
                                "decomp_unexplained": dec["unexplained"],
                                "eiv_unexplained_at_kappa": unexpl_c,
                                "kappa_point": kappa_point,
                                "kappa_sweep": sweep,
                                "eiv_unexplained_at": exact,
                                "eiv_min_kappa": kmin,
                                "eiv_correctable": unexpl_c is not None,
                                "n_covariate_columns": int(X.shape[1]),
                                "dml_theta_logpts": None, "dml_se": None,
                                "dml_nuisance": None, "boot": None,
                                "decomp_only": True})
                continue
            dml = dml_gap(X, ly, g, seed=seed)

            for lr in learners:
                try:
                    m = mrr(X, y, g, w, seed=seed, method=lr)
                except ArmUnavailable as e:
                    skips.append({"outcome_def": od, "covariate_set": cs_name,
                                  "learner": lr, "reason": str(e)})
                    continue
                rec = {"outcome_def": od, "covariate_set": cs_name, "learner": lr,
                       "n": cell["n"], "n_female": n_f, "n_male": n_m,
                       "powered": powered,
                       "delta_label": m["delta_label"], "delta_merit": m["delta_merit"],
                       "R": m["R"],
                       "decomp_raw_logpts": dec["raw"],
                       "decomp_explained": dec["explained"],
                       "decomp_unexplained": dec["unexplained"],
                       "eiv_unexplained_at_kappa": unexpl_c,
                       "kappa_point": kappa_point,
                       "kappa_sweep": sweep,
                       "eiv_unexplained_at": exact,
                       "eiv_min_kappa": kmin,
                       "eiv_correctable": unexpl_c is not None,
                       "n_covariate_columns": int(X.shape[1]),
                       "dml_theta_logpts": dml["theta"], "dml_se": dml["se"],
                       "dml_nuisance": dml["nuisance"],
                       "boot": None}
                # bootstrap on the primary arm only (registered in specs)
                if (od, cs_name, lr) in map(tuple, specs.BOOT_ARMS) and powered:
                    def stat(idx):
                        mm = mrr(X[idx], y[idx], g[idx], w[idx],
                                 seed=seed, method=lr)
                        return mm["R"]
                    b = stratified_bootstrap(stat, g, n_boot=n_boot, seed=seed)
                    rec["boot"] = {"lo": b["lo"], "hi": b["hi"], "n_boot": n_boot,
                                   "stat": "R", "learner_in_boot": lr}
                results.append(rec)

    return {"state": state, "year": year, "seed": seed,
            "tabpfn_model_version": specs.TABPFN_MODEL_VERSION,
            "timing_s": round(time.time() - t0, 1),
            "load_options": load_options or {},
            "spec_results": results, "skipped_arms": skips}
