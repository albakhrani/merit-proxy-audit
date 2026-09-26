# -*- coding: utf-8 -*-
"""Double/debiased machine learning arm: the group gap with the merit
covariates partialled out flexibly (partialling-out estimator with
cross-fitting, Chernozhukov et al. 2018).

Estimand: theta in  y = theta * g + f(X) + e,  with f learned by ML.
This is the flexible-form analogue of the decomposition's unexplained
component; the mapping is stated in the manuscript, not assumed here.
"""
from __future__ import annotations
import numpy as np


def _nuisance_factory(kind, seed):
    if kind == "auto":
        try:
            import xgboost  # noqa: F401
            kind = "xgboost"
        except ImportError:
            try:
                import sklearn  # noqa: F401
                kind = "sklearn_gbm"
            except ImportError:
                kind = "ridge_poly"
    if kind == "xgboost":
        import xgboost as xgb

        def fit_predict(Xtr, ytr, Xte):
            d = xgb.DMatrix(Xtr, label=ytr)
            bst = xgb.train(dict(objective="reg:squarederror", max_depth=4,
                                 eta=0.1, subsample=0.9, tree_method="hist",
                                 nthread=2, seed=seed), d, num_boost_round=200)
            return bst.predict(xgb.DMatrix(Xte))
        return fit_predict, kind
    if kind == "sklearn_gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor

        def fit_predict(Xtr, ytr, Xte):
            m = HistGradientBoostingRegressor(random_state=seed, max_depth=4)
            m.fit(Xtr, ytr)
            return m.predict(Xte)
        return fit_predict, kind
    if kind == "ridge_poly":
        def fit_predict(Xtr, ytr, Xte):
            def expand(Z):
                return np.column_stack([np.ones(len(Z)), Z, Z ** 2,
                                        Z[:, :1] * Z[:, 1:2] if Z.shape[1] > 1 else Z[:, :1] ** 3])
            A = expand(Xtr); lam = 1e-3 * np.eye(A.shape[1])
            beta = np.linalg.solve(A.T @ A + lam, A.T @ ytr)
            return expand(Xte) @ beta
        return fit_predict, kind
    raise ValueError(kind)


def dml_gap(X, y, g, n_folds=5, seed=0, nuisance="auto"):
    """Cross-fitted partialling-out estimate of the adjusted group gap.

    Returns theta, its influence-function standard error, and the nuisance
    learner actually used. Deterministic given seed.
    """
    X = np.asarray(X, float); y = np.asarray(y, float); g = np.asarray(g, float)
    fit_predict, used = _nuisance_factory(nuisance, seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, n_folds)
    ry = np.full(len(y), np.nan); rg = np.full(len(y), np.nan)
    for i in range(n_folds):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        ry[te] = y[te] - fit_predict(X[tr], y[tr], X[te])
        rg[te] = g[te] - fit_predict(X[tr], g[tr], X[te])
    denom = float(np.mean(rg * rg))
    theta = float(np.mean(rg * ry) / denom)
    psi = (ry - theta * rg) * rg / denom          # influence function
    se = float(np.sqrt(np.var(psi, ddof=1) / len(y)))
    return {"theta": theta, "se": se, "nuisance": used}
