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


# ---- Group-specific reference structure and indicator coding (28 Sep 2026,
# REGISTERED_CHANGES R17 and R18). The pooled estimator above is unchanged. ----

def _eiv_beta(X, y, w, kappa, proxy_index):
    """Coefficients of y on X (intercept first) with the errors-in-variables
    correction applied to one column; kappa = 1 or proxy_index None gives
    weighted least squares."""
    if kappa is None or proxy_index is None or kappa >= 1.0:
        return _ols(X, y, w)
    from .eiv import eiv_correct
    kap = np.ones(X.shape[1]); kap[proxy_index] = kappa
    return eiv_correct(X, y, kap, w)


def twofold_groupref(X_a, y_a, X_b, y_b, w_a=None, w_b=None,
                     proxy_index=None, kappa_a=None, kappa_b=None):
    """Twofold decomposition of mean(y_a) - mean(y_b) under each group's own
    coefficients as the reference structure.

    The outcome equation is fitted separately in group a and group b by
    weighted least squares. With the b coefficients as reference,
        explained = (xbar_a - xbar_b) . beta_b,
    and with the a coefficients as reference the same expression with
    beta_a; unexplained = raw - explained in each case, so explained plus
    unexplained equals the raw gap by construction. When the two groups
    have identical coefficients both references give the pooled result.

    kappa_a and kappa_b (None means 1) apply the errors-in-variables
    correction of eiv.eiv_correct to column proxy_index inside each group's
    regression; a common anchor passes the same value twice, sex-specific
    anchors pass one value per group. The correction is applied to the
    coefficients only; group means enter as observed.

    Returns raw, explained_ref_b, unexplained_ref_b, explained_ref_a,
    unexplained_ref_a, beta_a, beta_b.
    """
    X_a, y_a, X_b, y_b = map(np.asarray, (X_a, y_a, X_b, y_b))
    w_a = np.ones(len(y_a)) if w_a is None else np.asarray(w_a, float)
    w_b = np.ones(len(y_b)) if w_b is None else np.asarray(w_b, float)
    beta_a = _eiv_beta(X_a, y_a, w_a, kappa_a, proxy_index)
    beta_b = _eiv_beta(X_b, y_b, w_b, kappa_b, proxy_index)
    xbar_a = np.concatenate([[1.0], np.average(X_a, axis=0, weights=w_a)])
    xbar_b = np.concatenate([[1.0], np.average(X_b, axis=0, weights=w_b)])
    raw = float(np.average(y_a, weights=w_a) - np.average(y_b, weights=w_b))
    exp_b = float((xbar_a - xbar_b) @ beta_b)
    exp_a = float((xbar_a - xbar_b) @ beta_a)
    return {"raw": raw,
            "explained_ref_b": exp_b, "unexplained_ref_b": raw - exp_b,
            "explained_ref_a": exp_a, "unexplained_ref_a": raw - exp_a,
            "beta_a": beta_a, "beta_b": beta_b}


def indicator_columns(values):
    """Indicator columns for an integer-coded variable: one column per
    observed value in ascending order with the lowest value dropped, so that
    the columns plus an intercept span the same space as the full set.
    Returns (matrix, list of the values represented by the columns)."""
    v = np.asarray(values)
    levels = np.unique(v)
    cols = [(v == lv).astype(float) for lv in levels[1:]]
    if not cols:
        return np.empty((len(v), 0)), []
    return np.column_stack(cols), [float(x) for x in levels[1:]]
