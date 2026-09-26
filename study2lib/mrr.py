# -*- coding: utf-8 -*-
"""Merit-Reward Reversal on a cell, with a pluggable merit-only learner."""
from __future__ import annotations
import numpy as np
from .learners import oof_predictions


def mrr(X, y, group, w=None, n_folds=5, seed=0, method="logistic", folds=None, **kw):
    """Delta_label, Delta_merit and R on one cell.

    group: 1 for the group whose gap is reported (female minus male in the
    Study 1 convention). The merit-only learner never sees the group
    attribute; predictions are out of fold; Delta_merit compares mean
    out-of-fold scores between groups, matching the Study 1 estimator.
    """
    X = np.asarray(X, float); y = np.asarray(y, float); g = np.asarray(group, int)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    oof = oof_predictions(X, y, w, method=method, n_folds=n_folds, seed=seed,
                          folds=folds, **kw)
    d_label = (np.average(y[g == 1], weights=w[g == 1])
               - np.average(y[g == 0], weights=w[g == 0]))
    d_merit = (np.average(oof[g == 1], weights=w[g == 1])
               - np.average(oof[g == 0], weights=w[g == 0]))
    return {"delta_label": float(d_label), "delta_merit": float(d_merit),
            "R": float(d_merit - d_label), "oof": oof}
