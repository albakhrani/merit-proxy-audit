# -*- coding: utf-8 -*-
"""PV combining and replicate-weight variance on synthetic PIAAC-shaped data."""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from study2lib.piaac import kappa_with_uncertainty, kappa_by_stratum


def _fake(n=30000, kappa=0.72, R=20, seed=21):
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 1, n)
    x = m + rng.normal(0, np.sqrt((1 - kappa) / kappa), n)
    df = pd.DataFrame({"YRSQUAL": x, "SPFWT0": np.ones(n),
                       "GENDER_R": rng.integers(1, 3, n),
                       "VEMETHOD": ["JK2"] * n})
    for i in range(1, 11):
        df[f"PVLIT{i}"] = m + rng.normal(0, 0.15, n)   # PV noise
    for r in range(1, R + 1):
        df[f"SPFWT{r}"] = rng.uniform(0.5, 1.5, n)     # replicate weights
    return df, kappa


def test_kappa_combining_recovers_truth():
    df, kappa = _fake()
    pv = [f"PVLIT{i}" for i in range(1, 11)]
    r = kappa_with_uncertainty(df, "YRSQUAL", pv)
    assert abs(r["kappa"] - kappa) < 0.03
    assert r["se"] > 0 and r["n_replicates"] == 20 and "JK2" in r["replication"]


def test_stratum_output_shape_and_small_cell_rule():
    df, _ = _fake(n=4000)
    df.loc[df.index[:20], "GENDER_R"] = 3          # a tiny stratum
    pv = [f"PVLIT{i}" for i in range(1, 11)]
    out = kappa_by_stratum(df, "YRSQUAL", pv, {"sex": ["GENDER_R"]})
    assert "overall" in out and "sex" in out
    assert out["sex"]["3"]["kappa"] is None        # below minimum, reported not dropped


def test_fay_factor_and_padding_rule():
    """The US cycle-2 layout: VEFAYFAC and VENREPS in the file, replicate
    columns padded to 80 with copies of SPFWT0. The factor must be
    1/(R*(1-k)^2) over the declared R, and a padding column that differs
    from SPFWT0 must raise, never be dropped."""
    from study2lib.piaac import replication_factor
    df, _ = _fake(n=2000, R=44)
    df = df.drop(columns=["VEMETHOD"])
    df["VEFAYFAC"] = 0.3
    df["VENREPS"] = 44
    for r in range(45, 81):
        df[f"SPFWT{r}"] = df["SPFWT0"]
    factor, R, method = replication_factor(df)
    assert R == 44 and "FAY" in method
    assert abs(factor - 1.0 / (44 * (1 - 0.3) ** 2)) < 1e-12
    pv = [f"PVLIT{i}" for i in range(1, 11)]
    r = kappa_with_uncertainty(df, "YRSQUAL", pv)
    assert r["n_replicates"] == 44 and r["se"] > 0
    df["SPFWT60"] = df["SPFWT0"] * 1.01            # a padding column that lies
    try:
        replication_factor(df)
        raise AssertionError("padding violation not caught")
    except ValueError:
        pass


def test_scale_invariance_and_bounds():
    """kappa must be unchanged by any affine rescaling of the criterion or
    of the proxy, and must lie in [0, 1]. The slope form failed this on
    the real cycle-2 file (years of schooling against a 500-point
    proficiency scale gave 8.25); this test guards the whole class."""
    df, _ = _fake(n=6000, seed=9)
    pv = [f"PVLIT{i}" for i in range(1, 11)]
    base = kappa_with_uncertainty(df, "YRSQUAL", pv)
    assert 0.0 <= base["kappa"] <= 1.0
    df2 = df.copy()
    for c in pv:
        df2[c] = 250.0 + 100.0 * df2[c]            # a 500-point-style scale
    scaled = kappa_with_uncertainty(df2, "YRSQUAL", pv)
    assert abs(base["kappa"] - scaled["kappa"]) < 1e-10
    assert abs(base["se"] - scaled["se"]) < 1e-10
    df3 = df.copy()
    df3["YRSQUAL"] = 3.0 * df3["YRSQUAL"] - 7.0    # rescale the proxy too
    reproxy = kappa_with_uncertainty(df3, "YRSQUAL", pv)
    assert abs(base["kappa"] - reproxy["kappa"]) < 1e-10
    assert abs(abs(base["rho"]) - abs(reproxy["rho"])) < 1e-10


def test_constant_proxy_reported_not_nan():
    """A cell where the proxy is constant has no estimable reliability; it
    must come back as None with a reason, never as NaN or a warning."""
    df, _ = _fake(n=1000, seed=11)
    df["YRSQUAL"] = 12.0
    pv = [f"PVLIT{i}" for i in range(1, 11)]
    r = kappa_with_uncertainty(df, "YRSQUAL", pv)
    assert r["kappa"] is None and "constant" in r["note"]


def test_rounding_dust_variance_reported_not_estimated():
    """R3 guard: a proxy constant up to rounding dust (variance ~1e-32,
    as in the real educ strata) must also return None, never a 1e-30
    'estimate'."""
    rng = np.random.default_rng(13)
    df, _ = _fake(n=1000, seed=13)
    df["YRSQUAL"] = 12.0 + rng.normal(0, 1e-16, len(df))
    pv = [f"PVLIT{i}" for i in range(1, 11)]
    r = kappa_with_uncertainty(df, "YRSQUAL", pv)
    assert r["kappa"] is None


def test_sex_difference_replicate_se():
    """R3: the sex difference in kappa with the SE computed on the
    difference inside the shared replication design. A real differential
    (0.85 vs 0.65) must be recovered and detected."""
    from study2lib.piaac import kappa_sex_difference
    rng = np.random.default_rng(17)
    n = 20000
    m = rng.normal(0, 1, n)
    sex = rng.integers(1, 3, n)
    nu_sd = np.where(sex == 1, np.sqrt((1 - 0.85) / 0.85),
                     np.sqrt((1 - 0.65) / 0.65))
    df = pd.DataFrame({"YRSQUAL": m + rng.normal(0, 1, n) * nu_sd,
                       "GENDER_R": sex, "SPFWT0": np.ones(n),
                       "VEMETHOD": ["JK2"] * n})
    for i in range(1, 11):
        df[f"PVLIT{i}"] = m + rng.normal(0, 0.10, n)
    for r_ in range(1, 21):
        df[f"SPFWT{r_}"] = rng.uniform(0.5, 1.5, n)
    d = kappa_sex_difference(df, "YRSQUAL", [f"PVLIT{i}" for i in range(1, 11)])
    assert abs(d["difference"] - 0.20) < 0.03
    assert d["se"] > 0 and d["z"] > 3
    assert d["convention"].startswith("GENDER_R 1 minus GENDER_R 2")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print(f"  pass  {f.__name__}")
    print(f"{len(fns)} tests passed")
