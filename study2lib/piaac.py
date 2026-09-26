# -*- coding: utf-8 -*-
"""PIAAC loading, plausible-value combining, replicate-weight variance, and
reliability estimation by stratum.

Methodology: for a statistic computed on assessed skill, PIAAC requires
combining the 10 plausible values by Rubin's rules and estimating sampling
variance with the replicate weights (SPFWT1..SPFWTn), with the replication
factor implied by the variance-estimation variables in the file. Total
variance = W + (1 + 1/M) * B, where W is the mean replication variance
across plausible values and B the between-plausible-value variance.

Replication factor resolution, in order, never silently:
  1. A populated VEFAYFAC in (0, 1) means Fay's BRR with that factor k and
     the variance factor is 1 / (R * (1 - k)^2). The US cycle-2 file
     carries VEFAYFAC = 0.3 with VENREPS = 44 (NCES 2023 technical notes:
     balanced repeated replication with Fay's adjustment, factor 0.3).
  2. Otherwise the VEMETHOD label: JK1 gives (R-1)/R, JK2 gives 1.0, BRR
     gives 1/R. A label naming Fay without a usable VEFAYFAC raises: the
     caller must resolve k from the technical report, never a default.
  3. A numeric VEMETHODN with neither of the above raises: no numeric
     code-to-method mapping is guessed here.
  4. Nothing at all: the JK2 factor 1.0 is used and recorded as assumed.

R is the file's declared VENREPS when present. International PIAAC files
pad the replicate-weight block to 80 columns; every column beyond VENREPS
must equal SPFWT0 on all rows (pure padding) and is excluded only after
that identity is verified. A padding column that differs raises rather
than being dropped.

Cycle awareness: cycle 2 renames several variables with a C2 or _TC1
suffix and ships semicolon-delimited CSVs with a dot decimal separator.
load_piaac sniffs the delimiter and loads both generations of names.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .eiv import estimate_kappa, weighted_correlation

PV_LIT = [f"PVLIT{i}" for i in range(1, 11)]
PV_NUM = [f"PVNUM{i}" for i in range(1, 11)]

# cycle 1 name -> cycle 2 name; both generations are loaded when present
CYCLE_NAMES = {
    "YRSQUAL": "YRSQUALC2",
    "B_Q01A": "B2_Q01",
    "EDCAT8": "EDCAT8_TC1",
    "EDCAT7": "EDCAT7_TC1",
    "EARNMTHALLDCL": "EARNMTHALLDCLC2",
    "EARNHRDCL": "EARNHRDCLC2",
}


def sniff_delimiter(path):
    """Cycle-1 PUF CSVs are comma-delimited, cycle-2 semicolon-delimited."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        head = fh.readline()
    return ";" if head.count(";") > head.count(",") else ","


def resolve_column(df, name):
    """The column under its cycle-1 or cycle-2 name, or None."""
    for c in (name, CYCLE_NAMES.get(name, ""), *(k for k, v in CYCLE_NAMES.items() if v == name)):
        if c and c in df.columns:
            return c
    return None


def load_piaac(path, extra_cols=()):
    """Read the PUF with only the columns the study uses."""
    want = set(PV_LIT + PV_NUM + ["SPFWT0", "GENDER_R", "AGE_R", "AGEG10LFS",
                                  "CTRYRGN", "VEMETHOD", "VEMETHODN",
                                  "VENREPS", "VEFAYFAC", "VARSTRAT", "VARUNIT",
                                  "EDLEVEL3", "EDCAT6_TC1",
                                  "EARNFLAGC2"]) | set(extra_cols)
    want |= set(CYCLE_NAMES) | set(CYCLE_NAMES.values())
    want |= {f"SPFWT{i}" for i in range(1, 121)}
    df = pd.read_csv(path, sep=sniff_delimiter(path),
                     usecols=lambda c: c.strip().upper() in want,
                     low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def _single_value(df, col):
    """The column's single distinct numeric value, else None."""
    if col not in df.columns:
        return None
    v = pd.to_numeric(df[col], errors="coerce").dropna().unique()
    return float(v[0]) if len(v) == 1 else None


def replication_factor(df):
    """(factor, n_replicates, method_string). Resolution order and the
    padding rule are documented in the module docstring."""
    have = [c for c in df.columns if c.startswith("SPFWT") and c != "SPFWT0"]
    R = len(have)
    declared = _single_value(df, "VENREPS")
    if declared is not None and 0 < int(declared) <= R:
        R = int(declared)
        w0 = pd.to_numeric(df["SPFWT0"], errors="coerce").to_numpy(float)
        for c in have:
            if int(c[5:]) > R:
                wr = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
                if not np.allclose(wr, w0, rtol=0, atol=1e-9, equal_nan=True):
                    raise ValueError(
                        f"{c} differs from SPFWT0 yet exceeds VENREPS={R}; "
                        "refusing to drop a replicate that carries information")
    method = None
    if "VEMETHOD" in df.columns:
        vals = df["VEMETHOD"].dropna().astype(str).str.strip().str.upper()
        vals = vals[vals != ""].unique()
        method = vals[0] if len(vals) else None
    k = _single_value(df, "VEFAYFAC")
    if k is not None and 0 < k < 1:
        return 1.0 / (R * (1.0 - k) ** 2), R, f"FAY k={k:g}"
    if method and "JK1" in method:
        return (R - 1) / R, R, method
    if method and "FAY" in method:
        raise ValueError("VEMETHOD reports Fay but VEFAYFAC is unusable; "
                         "resolve k from the technical report before running")
    if method and "BRR" in method:
        return 1.0 / R, R, method
    if method and "JK2" in method:
        return 1.0, R, method
    code = _single_value(df, "VEMETHODN")
    if code is not None:
        raise ValueError(f"VEMETHODN={code:g} is present without a VEMETHOD "
                         "label or a VEFAYFAC; no numeric mapping is guessed")
    return 1.0, R, method or "ABSENT->JK2 assumed"


def kappa_with_uncertainty(df, proxy_col, pv_cols, weight_col="SPFWT0"):
    """Reliability of proxy_col for the construct measured by pv_cols,
    combined over plausible values with replicate-weight variance.

    kappa is the weighted squared correlation (see eiv.estimate_kappa);
    the signed correlation is returned alongside it as "rho". A cell where
    the proxy or criterion is constant, or with under 2 complete cases, is
    returned with kappa None and a reason, never as NaN."""
    factor, R, method = replication_factor(df)
    reps = [f"SPFWT{i}" for i in range(1, R + 1) if f"SPFWT{i}" in df.columns]
    kappas, rhos, wvars = [], [], []
    x = pd.to_numeric(df[proxy_col], errors="coerce").to_numpy(float)
    w0 = pd.to_numeric(df[weight_col], errors="coerce").to_numpy(float)
    n = int(np.isfinite(x).sum())
    for pv in pv_cols:
        a = pd.to_numeric(df[pv], errors="coerce").to_numpy(float)
        r0 = weighted_correlation(x, a, w0)
        if not np.isfinite(r0):
            return {"kappa": None, "n": n,
                    "note": "proxy or criterion constant, or under 2 complete cases",
                    "n_replicates": R, "replication": method, "factor": factor}
        k0 = r0 * r0
        kr = np.array([estimate_kappa(x, a, pd.to_numeric(df[r], errors="coerce").to_numpy(float))
                       for r in reps])
        wvars.append(factor * np.nansum((kr - k0) ** 2))
        kappas.append(k0); rhos.append(r0)
    kappas = np.array(kappas); M = len(kappas)
    point = float(np.mean(kappas))
    W = float(np.mean(wvars))
    B = float(np.var(kappas, ddof=1)) if M > 1 else 0.0
    var_total = W + (1 + 1 / M) * B
    return {"kappa": point, "rho": float(np.mean(rhos)),
            "se": float(np.sqrt(var_total)), "n": n,
            "n_replicates": R, "replication": method, "factor": factor}


def kappa_sex_difference(df, proxy_col, pv_cols, weight_col="SPFWT0",
                         sex_col="GENDER_R", male=1, female=2):
    """kappa(men) minus kappa(women) with the variance computed on the
    difference itself inside the shared replication design: both subgroups
    share one Fay BRR sample, so the correct SE differences the two kappas
    within each replicate weight and each plausible value, then combines by
    the Fay factor and Rubin's rules. The independent-SE approximation is
    not used. Added 15 September 2026 (post-results, disclosed)."""
    factor, R, method = replication_factor(df)
    reps = [f"SPFWT{i}" for i in range(1, R + 1) if f"SPFWT{i}" in df.columns]
    sx = pd.to_numeric(df[sex_col], errors="coerce").to_numpy(float)
    x = pd.to_numeric(df[proxy_col], errors="coerce").to_numpy(float)
    w0 = pd.to_numeric(df[weight_col], errors="coerce").to_numpy(float)
    mm, ff = sx == male, sx == female
    diffs, wvars = [], []
    for pv in pv_cols:
        a = pd.to_numeric(df[pv], errors="coerce").to_numpy(float)
        d0 = (estimate_kappa(x[mm], a[mm], w0[mm])
              - estimate_kappa(x[ff], a[ff], w0[ff]))
        if not np.isfinite(d0):
            return {"difference": None,
                    "note": "a subgroup kappa is not estimable"}
        dr = []
        for rcol in reps:
            wr = pd.to_numeric(df[rcol], errors="coerce").to_numpy(float)
            dr.append(estimate_kappa(x[mm], a[mm], wr[mm])
                      - estimate_kappa(x[ff], a[ff], wr[ff]))
        dr = np.array(dr)
        wvars.append(factor * np.nansum((dr - d0) ** 2))
        diffs.append(d0)
    diffs = np.array(diffs); M = len(diffs)
    point = float(np.mean(diffs))
    B = float(np.var(diffs, ddof=1)) if M > 1 else 0.0
    var_total = float(np.mean(wvars)) + (1 + 1 / M) * B
    se = float(np.sqrt(var_total))
    return {"difference": point, "se": se,
            "z": (point / se if se > 0 else None),
            "convention": f"{sex_col} {male} minus {sex_col} {female}",
            "n_replicates": R, "replication": method, "factor": factor}


def kappa_by_stratum(df, proxy_col, pv_cols, strata):
    """kappa in every cell of every stratum definition.

    strata: dict name -> list of columns, e.g. {"sex": ["GENDER_R"],
    "sex_educ": ["GENDER_R", "EDCAT8_TC1"]}. Returns nested dict with the
    overall estimate under key "overall". Cells under 30 observations are
    reported with kappa None and their n, never silently dropped. Stratum
    columns are coerced to numeric before grouping, so string missing
    codes such as "." fall out of the grouping instead of forming their
    own stratum.
    """
    gdf = df.copy()
    for cols in strata.values():
        for c in cols:
            gdf[c] = pd.to_numeric(gdf[c], errors="coerce")
    out = {"overall": kappa_with_uncertainty(df, proxy_col, pv_cols)}
    for name, cols in strata.items():
        cells = {}
        for key, sub in gdf.groupby(cols, dropna=True):
            label = "_".join(str(int(k)) if isinstance(k, float) and k == int(k)
                             else str(k) for k in (key if isinstance(key, tuple) else (key,)))
            if len(sub) < 30:
                cells[label] = {"kappa": None, "n": int(len(sub)),
                                "note": "below minimum stratum size 30"}
            else:
                cells[label] = kappa_with_uncertainty(sub, proxy_col, pv_cols)
        out[name] = cells
    return out


# ---- Differential validity by sex (22 Sep 2026, REGISTERED_CHANGES R10) ----

def _wls(Xmat, y, w):
    """Weighted least squares coefficients, intercept first."""
    sw = np.sqrt(w)
    Xd = np.column_stack([np.ones(len(y)), Xmat])
    beta, *_ = np.linalg.lstsq(Xd * sw[:, None], y * sw, rcond=None)
    return beta


def _rubin(estimates, factor):
    """estimates: array (M plausible values, 1 + R weight vectors), column 0
    the full-weight estimate, columns 1..R the replicate estimates. Returns
    point, se by Rubin's rules over PVs with the replicate variance inside."""
    est = np.asarray(estimates, float)
    M = est.shape[0]
    point = float(np.mean(est[:, 0]))
    W = float(np.mean([factor * np.nansum((est[i, 1:] - est[i, 0]) ** 2) for i in range(M)]))
    B = float(np.var(est[:, 0], ddof=1)) if M > 1 else 0.0
    se = float(np.sqrt(W + (1 + 1 / M) * B))
    return {"point": point, "se": se, "z": (point / se if se > 0 else None)}


def differential_validity(df, proxy_col, pv_cols, weight_col="SPFWT0",
                          sex_col="GENDER_R", male=1, female=2,
                          age=None, proxy_dummies=False):
    """Does the proxy carry a sex difference in the mean of its error?

    Regresses each plausible value of assessed skill on the proxy (linear,
    or category dummies when proxy_dummies is True), optional age terms and
    a female indicator, with the full weight and every replicate weight,
    and combines by Rubin's rules with the file's replication factor.

    The classical-error benchmark for the female coefficient is not zero.
    With X = M* + nu, A = c0 + c1 M*, one within-sex reliability kappa and
    equal error means,
        E[A | X, sex] = c0 + c1 kappa X + c1 (1 - kappa) mu_sex,
    so the female coefficient equals b_X (1 - kappa) / kappa times women's
    lead in mean X, where b_X = c1 kappa is the fitted slope. The benchmark
    is computed inside every replicate and plausible value from the
    replicate's own b_X, within-sex kappa and mean difference, so that
    difference = b_female - benchmark carries its own standard error. With
    error means delta_f and delta_m the difference equals -c1 (delta_f -
    delta_m), so the sex difference in the mean of the proxy error, in
    proxy units, is delta = -difference * kappa / b_X. A positive delta
    means women's proxy overstates their assessed skill relative to men's,
    the case under which the correction could shrink the gap.

    age: None (no age terms), or an array-like of age values (linear), or
    a 2-D array of age-band dummies. With age terms the benchmark is the
    same expression and is an approximation (the derivation holds exactly
    without covariates); the no-age specification is the clean test.

    Returns a dict of Rubin-combined statistics; None where not defined.
    """
    factor, R, method = replication_factor(df)
    reps = [f"SPFWT{i}" for i in range(1, R + 1) if f"SPFWT{i}" in df.columns]
    sx = pd.to_numeric(df[sex_col], errors="coerce").to_numpy(float)
    w0 = pd.to_numeric(df[weight_col], errors="coerce").to_numpy(float)
    x = pd.to_numeric(df[proxy_col], errors="coerce").to_numpy(float)
    ff = sx == female
    keep = np.isfinite(x) & np.isfinite(w0) & (w0 > 0) & ((sx == male) | (sx == female))
    age_mat = None
    if age is not None:
        age_mat = np.asarray(age, float)
        if age_mat.ndim == 1:
            age_mat = age_mat[:, None]
        keep &= np.all(np.isfinite(age_mat), axis=1)
    if proxy_dummies:
        cats = np.unique(x[keep])
        Xp = np.column_stack([(x == c).astype(float) for c in cats[1:]])
    else:
        Xp = x[:, None]
    weights = [w0] + [pd.to_numeric(df[r], errors="coerce").to_numpy(float) for r in reps]
    stats = {k: [] for k in ("b_female", "b_x", "kappa_within", "dxbar", "benchmark",
                             "difference", "delta_proxy_units", "pv_sd")}
    n_used = None
    for pv in pv_cols:
        a = pd.to_numeric(df[pv], errors="coerce").to_numpy(float)
        k2 = keep & np.isfinite(a)
        n_used = int(k2.sum())
        rows = {k: [] for k in stats}
        for wv in weights:
            m = k2 & np.isfinite(wv) & (wv > 0)
            cols = [Xp[m]]
            if age_mat is not None:
                cols.append(age_mat[m])
            cols.append(ff[m].astype(float)[:, None])
            Xmat = np.column_stack(cols)
            beta = _wls(Xmat, a[m], wv[m])
            b_female = float(beta[-1])
            rows["b_female"].append(b_female)
            am = np.average(a[m], weights=wv[m])
            rows["pv_sd"].append(float(np.sqrt(np.average((a[m] - am) ** 2, weights=wv[m]))))
            if proxy_dummies:
                for k in ("b_x", "kappa_within", "dxbar", "benchmark", "difference",
                          "delta_proxy_units"):
                    rows[k].append(np.nan)
                continue
            b_x = float(beta[1])
            # within-sex reliability: squared weighted correlation of x and a
            # after removing each sex's weighted mean
            xd = x[m].copy(); ad = a[m].copy(); fm = ff[m]; wm_ = wv[m]
            for grp in (fm, ~fm):
                xd[grp] -= np.average(xd[grp], weights=wm_[grp])
                ad[grp] -= np.average(ad[grp], weights=wm_[grp])
            r_w = weighted_correlation(xd, ad, wm_)
            kap = float(r_w * r_w)
            dxbar = float(np.average(x[m][fm], weights=wm_[fm])
                          - np.average(x[m][~fm], weights=wm_[~fm]))
            bench = b_x * (1.0 - kap) / kap * dxbar if kap > 0 else np.nan
            diff = b_female - bench
            rows["b_x"].append(b_x); rows["kappa_within"].append(kap)
            rows["dxbar"].append(dxbar); rows["benchmark"].append(float(bench))
            rows["difference"].append(float(diff))
            rows["delta_proxy_units"].append(float(-diff * kap / b_x) if b_x != 0 else np.nan)
        for k in stats:
            stats[k].append(rows[k])
    out = {"n": n_used, "n_replicates": R, "replication": method, "factor": factor,
           "proxy_dummies": bool(proxy_dummies), "age_terms": age_mat is not None
           and int(age_mat.shape[1])}
    for k, v in stats.items():
        arr = np.asarray(v, float)
        out[k] = None if np.all(np.isnan(arr)) else _rubin(arr, factor)
    if out["b_female"] and out["pv_sd"]:
        out["b_female_in_pv_sd"] = out["b_female"]["point"] / out["pv_sd"]["point"]
    return out
