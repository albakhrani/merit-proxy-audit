# -*- coding: utf-8 -*-
"""Unified learner interface for the merit-only classifier.

Three registered families: 'logistic' (in-house, always available),
'xgboost' (gradient-boosted trees), 'tabpfn' (tabular foundation model,
GPU when available). Optional arms raise ArmUnavailable with a reason, and
the grid runner logs the skip instead of failing the cell.
"""
from __future__ import annotations
import numpy as np


class ArmUnavailable(RuntimeError):
    pass


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _logit_fit(X, y, w, l2=1e-6, iters=200):
    Xc = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(Xc.shape[1])
    for _ in range(iters):
        p = _sigmoid(Xc @ beta)
        g = Xc.T @ (w * (y - p)) - l2 * beta
        Wd = w * p * (1 - p) + 1e-12
        H = (Xc * Wd[:, None]).T @ Xc + l2 * np.eye(Xc.shape[1])
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def folds_by_person(person, n_folds=5, seed=0):
    """Fold membership for rows that may repeat a person (a bootstrap
    resample): every row of the same person lands in the same fold, so a
    duplicated person never sits in both the training and the evaluation
    partition. person: integer id per row. Deterministic given seed.
    Added 22 Sep 2026 (REGISTERED_CHANGES R12)."""
    person = np.asarray(person)
    uniq, inv = np.unique(person, return_inverse=True)
    rng = np.random.default_rng(seed)
    fold_of_person = rng.permutation(len(uniq)) % n_folds
    fold_of_row = fold_of_person[inv]
    return [np.flatnonzero(fold_of_row == i) for i in range(n_folds)]


def oof_predictions(X, y, w, method="logistic", n_folds=5, seed=0,
                    tabpfn_max_train=10000, device="auto", folds=None):
    """Out-of-fold predicted probabilities from a merit-only classifier.

    Deterministic given (method, seed). The group attribute must not be a
    column of X; the caller is responsible for that exclusion. folds: an
    optional list of index arrays replacing the seeded random split (used
    by the bootstrap v2 to keep a resampled person inside one fold).
    """
    X = np.asarray(X, float); y = np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    if folds is None:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        folds = np.array_split(idx, n_folds)
    else:
        folds = [np.asarray(f, int) for f in folds]
        n_folds = len(folds)
    oof = np.full(len(y), np.nan)

    if method == "logistic":
        for i in range(n_folds):
            te = folds[i]; tr = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            beta = _logit_fit(X[tr], y[tr], w[tr])
            oof[te] = _sigmoid(np.column_stack([np.ones(len(te)), X[te]]) @ beta)
        return oof

    if method == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ArmUnavailable("xgboost not installed") from e
        params = dict(objective="binary:logistic", max_depth=4, eta=0.1,
                      subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                      nthread=2, seed=seed, eval_metric="logloss")
        for i in range(n_folds):
            te = folds[i]; tr = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            dtr = xgb.DMatrix(X[tr], label=y[tr], weight=w[tr])
            dte = xgb.DMatrix(X[te])
            bst = xgb.train(params, dtr, num_boost_round=200)
            oof[te] = bst.predict(dte)
        return oof

    if method == "tabpfn":
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as e:
            raise ArmUnavailable("tabpfn not installed") from e
        try:
            import torch
            dev = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            dev = "cpu"
        for i in range(n_folds):
            te = folds[i]; tr = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            if len(tr) > tabpfn_max_train:
                # registered, seeded subsampling to the model's context limit
                sub = np.random.default_rng(seed + 1000 + i).choice(len(tr), tabpfn_max_train, replace=False)
                tr = tr[sub]
            # The weights generation is pinned in specs.TABPFN_MODEL_VERSION and
            # recorded in every cell result.
            from .specs import TABPFN_MODEL_VERSION as model_version
            clf = TabPFNClassifier.create_default_for_version(
                model_version, device=dev, random_state=seed)
            clf.fit(X[tr], y[tr].astype(int))
            oof[te] = clf.predict_proba(X[te])[:, 1]
        return oof

    raise ValueError(f"unknown learner {method!r}")
