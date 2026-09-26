# -*- coding: utf-8 -*-
"""Errors-in-variables: reliability estimation and moment-based correction."""
from __future__ import annotations
import numpy as np


def weighted_correlation(proxy, assessed, w=None) -> float:
    """Signed weighted Pearson correlation on complete cases.

    Returns NaN when fewer than 2 complete cases remain or either variable
    is constant; callers must report such cells as None with a reason,
    never as a number.
    """
    x = np.asarray(proxy, float); a = np.asarray(assessed, float)
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    keep = np.isfinite(x) & np.isfinite(a) & np.isfinite(w) & (w > 0)
    x, a, w = x[keep], a[keep], w[keep]
    if len(x) < 2:
        return float("nan")
    xm = np.average(x, weights=w); am = np.average(a, weights=w)
    vx = np.average((x - xm) ** 2, weights=w)
    va = np.average((a - am) ** 2, weights=w)
    # 1e-20 threshold (15 Sep 2026): a proxy constant within a stratum can
    # leave rounding dust of order 1e-30 instead of exactly zero; real
    # variances in use are at least 1e-2, so this cannot bite a true signal.
    if vx < 1e-20 or va < 1e-20:
        return float("nan")
    cov = np.average((x - xm) * (a - am), weights=w)
    return float(cov / np.sqrt(vx * va))


def estimate_kappa(proxy, assessed, w=None) -> float:
    """Reliability of a proxy for a latent construct, using a directly
    assessed measure of that construct as the validation instrument.

    Estimator: the weighted squared correlation between proxy X and the
    assessed measure A. Under the classical model X = M* + nu with error
    independent of M*, rho(X, A)^2 equals Var(M*)/Var(X) whenever A is any
    error-free linear transform of M*, so on a shared scale this is the
    textbook reliability ratio, and it is invariant to the scales of both
    variables by construction. Redefined 7 September 2026 after the first
    real T3 run: the previous form Cov(X, A)/Var(X) is the regression
    slope of A on X and carries A's units per unit of X; on the PIAAC
    cycle-2 file (years of schooling against a 500-point proficiency
    scale) it returned 8.25, which is not a reliability. The two forms
    coincide exactly in the shared-scale special case, which is why every
    synthetic test passed; the scale-invariance test now guards the class.

    Assumptions this does NOT test, stated in the manuscript's
    identification section: linearity of the proxy in M*, and A carrying
    no measurement error of its own. For PIAAC, plausible values handle
    the second when the proxy is in the conditioning model; when A is
    instead a noisy measure of M*, rho(X, A)^2 understates the
    reliability by the factor rho(A, M*)^2. Correction of 15 September
    2026: in a dataset whose explained component is positive, an
    understated kappa ENLARGES the corrected residual in the direction of
    the study's own conclusion, so the attenuation is anti-conservative
    there; never describe the anchor as conservative. The robust statement
    is the sign across the whole sweep, kappa up to 1.

    Returns NaN under the conditions listed for weighted_correlation.
    """
    r = weighted_correlation(proxy, assessed, w)
    return float(r * r)


def eiv_correct(X, y, kappas, w=None):
    """Method-of-moments errors-in-variables regression.

    kappas: per-column reliability in (0, 1]; 1.0 marks a column measured
    without error. The corrected coefficients solve
        (S_XX - Lambda) b = S_Xy
    where Lambda is diagonal with (1 - kappa_j) * Var(X_j). Reduces to OLS
    when every kappa is 1, and to beta_hat / kappa in the univariate case.

    Raises np.linalg.LinAlgError when S_XX - Lambda is singular or not
    positive definite, which is a diagnosis (reliability too low for this
    design matrix), not a bug; the caller records the cell as not
    correctable at these kappas rather than reporting coefficients from an
    indefinite system.
    """
    X = np.asarray(X, float); y = np.asarray(y, float)
    k = np.asarray(kappas, float)
    if np.any((k <= 0) | (k > 1)):
        raise ValueError("each kappa must lie in (0, 1]")
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    wm = w / w.sum()
    xbar = wm @ X; ybar = float(wm @ y)
    Xc = X - xbar; yc = y - ybar
    Sxx = (Xc * wm[:, None]).T @ Xc
    Sxy = (Xc * wm[:, None]).T @ yc
    # Var(X) = Var(M*)/kappa  =>  error variance = (1 - kappa) * Var(X)
    lam = np.diag((1.0 - k) * np.diag(Sxx))
    S = Sxx - lam
    eigmin = float(np.linalg.eigvalsh(S).min())
    if eigmin <= 0:
        raise np.linalg.LinAlgError(
            f"S_XX - Lambda not positive definite (min eigenvalue {eigmin:.3e});"
            " the assumed reliabilities are too low for this design matrix")
    b = np.linalg.solve(S, Sxy)
    intercept = ybar - float(xbar @ b)
    return np.concatenate([[intercept], b])


def min_admissible_kappa(X, j, w=None) -> float:
    """Smallest reliability of column j for which S_XX - Lambda stays
    positive definite, when only that column carries measurement error.

    Added 25 September 2026 after the occupation arm failed in all 250
    markets. Writing S for the weighted centred second-moment matrix, the
    corrected system subtracts lam = (1 - kappa) * S_jj from entry (j, j),
    and S - lam e_j e_j' is positive definite exactly when
    lam < 1 / (S^-1)_jj. Substituting gives

        kappa_min = 1 - 1 / (S_jj * (S^-1)_jj) = R^2_j,

    the weighted share of column j's variance that the other columns
    already explain. The correction therefore exists only when the assumed
    reliability exceeds that share: a proxy the other covariates can
    reconstruct leaves no room for the error variance the correction wants
    to remove. Returns 1.0 for a design matrix that is already singular.
    """
    X = np.asarray(X, float)
    w = np.ones(len(X)) if w is None else np.asarray(w, float)
    wm = w / w.sum()
    Xc = X - wm @ X
    S = (Xc * wm[:, None]).T @ Xc
    try:
        inv_jj = float(np.linalg.inv(S)[j, j])
    except np.linalg.LinAlgError:
        return 1.0
    denom = float(S[j, j]) * inv_jj
    if not np.isfinite(denom) or denom <= 0:
        return 1.0
    return float(min(max(1.0 - 1.0 / denom, 0.0), 1.0))

