# -*- coding: utf-8 -*-
"""Stratified bootstrap, seeded, resampling within group strata."""
from __future__ import annotations
import numpy as np


def stratified_bootstrap(stat_fn, strata, n_boot=1000, seed=0, alpha=0.05):
    """stat_fn(indices) -> float; strata: integer array defining resampling
    strata (the group attribute in Study 1's convention). Returns point
    estimate on the full sample, the percentile interval, and the draws."""
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    point = stat_fn(np.arange(len(strata)))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        draws[b] = stat_fn(idx)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "draws": draws}
