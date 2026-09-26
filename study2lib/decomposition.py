# -*- coding: utf-8 -*-
"""Twofold Oaxaca-Blinder decomposition under a pooled reference structure."""
from __future__ import annotations
import numpy as np


def _ols(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    Xc = np.column_stack([np.ones(len(X)), X])
    if w is None:
        beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    else:
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(Xc * sw[:, None], y * sw, rcond=None)
    return beta


def twofold_neumark(X_a, y_a, X_b, y_b, w_a=None, w_b=None):
    """Twofold decomposition of mean(y_a) - mean(y_b).

    The reference coefficient vector is the pooled regression including a
    group indicator whose coefficient is discarded (the pooled variant of the
    Neumark reference used throughout Study 1).

    Returns dict with raw, explained, unexplained, beta_star, beta_a, beta_b.
    The identity raw = explained + unexplained holds to numerical precision
    when weights are the same objects passed here.
    """
    X_a, y_a, X_b, y_b = map(np.asarray, (X_a, y_a, X_b, y_b))
    w_a = np.ones(len(y_a)) if w_a is None else np.asarray(w_a, float)
    w_b = np.ones(len(y_b)) if w_b is None else np.asarray(w_b, float)

    Xp = np.vstack([X_a, X_b])
    yp = np.concatenate([y_a, y_b])
    wp = np.concatenate([w_a, w_b])
    g = np.concatenate([np.ones(len(y_a)), np.zeros(len(y_b))])
    beta_pool = _ols(np.column_stack([Xp, g]), yp, wp)
    beta_star = np.concatenate([beta_pool[:1], beta_pool[1:-1]])  # drop the group dummy

    beta_a = _ols(X_a, y_a, w_a)
    beta_b = _ols(X_b, y_b, w_b)

    xbar_a = np.concatenate([[1.0], np.average(X_a, axis=0, weights=w_a)])
    xbar_b = np.concatenate([[1.0], np.average(X_b, axis=0, weights=w_b)])

    raw = float(np.average(y_a, weights=w_a) - np.average(y_b, weights=w_b))
    explained = float((xbar_a - xbar_b) @ beta_star)
    unexplained = raw - explained
    return {"raw": raw, "explained": explained, "unexplained": unexplained,
            "beta_star": beta_star, "beta_a": beta_a, "beta_b": beta_b}
