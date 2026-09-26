# -*- coding: utf-8 -*-
"""Differential validity by sex on synthetic PIAAC-shaped data with a known
answer (REGISTERED_CHANGES R10). Two checks:

  1. classical error with equal error means: the female coefficient equals
     the benchmark b_X (1 - kappa)/kappa * (women's lead in mean proxy), so
     the combined difference sits near zero and delta near zero;
  2. a sex difference in the mean proxy error of +0.8 units for women
     (women's schooling overstating their skill): delta recovers +0.8.

The analytic claim the manuscript rests on carries a test.
"""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from study2lib.piaac import differential_validity


def _fake(n=60000, kappa=0.25, mu_gap=0.6, delta_gap=0.0, R=44, seed=5):
    rng = np.random.default_rng(seed)
    sex = rng.integers(1, 3, n)                       # 1 male, 2 female
    f = sex == 2
    m_star = rng.normal(0, 1, n) + mu_gap * f         # women lead in true skill
    x = m_star + rng.normal(0, np.sqrt((1 - kappa) / kappa), n) + delta_gap * f
    a = 250 + 40 * m_star                             # error-free linear transform
    df = pd.DataFrame({"YRSQUALC2": x, "SPFWT0": np.ones(n), "GENDER_R": sex,
                       "AGE_R": rng.integers(16, 66, n), "VEMETHOD": ["JK1"] * n,
                       "VENREPS": R})
    for i in range(1, 11):
        df[f"PVLIT{i}"] = a + rng.normal(0, 3, n)     # small PV noise
    # delete-one-group jackknife replicates (JK1), so the replicate variance
    # is a calibrated estimate of the sampling variance in this simulation
    grp = rng.integers(0, R, n)
    for r in range(1, R + 1):
        df[f"SPFWT{r}"] = np.where(grp == r - 1, 0.0, R / (R - 1.0))
    return df, kappa


def test_equal_error_means_gives_zero_difference():
    df, kappa = _fake()
    pv = [f"PVLIT{i}" for i in range(1, 11)]
    r = differential_validity(df, "YRSQUALC2", pv)
    assert abs(r["kappa_within"]["point"] - kappa) < 0.02
    assert r["b_female"]["point"] > 0                 # women's lead makes it positive
    assert abs(r["difference"]["z"]) < 3.0, r["difference"]
    assert abs(r["delta_proxy_units"]["point"]) < 0.15, r["delta_proxy_units"]
    print(f"  ok: null case, difference z {r['difference']['z']:+.2f}, "
          f"delta {r['delta_proxy_units']['point']:+.3f}")


def test_error_mean_shift_is_recovered():
    df, kappa = _fake(delta_gap=0.8, seed=6)
    pv = [f"PVLIT{i}" for i in range(1, 11)]
    r = differential_validity(df, "YRSQUALC2", pv)
    assert r["difference"]["point"] < 0 and r["difference"]["z"] < -3
    assert abs(r["delta_proxy_units"]["point"] - 0.8) < 0.15, r["delta_proxy_units"]
    # with age terms the test still runs and points the same way
    r2 = differential_validity(df, "YRSQUALC2", pv, age=df["AGE_R"].to_numpy(float))
    assert r2["difference"]["point"] < 0
    print(f"  ok: shift case, delta {r['delta_proxy_units']['point']:+.3f} "
          f"(truth +0.800), z {r['difference']['z']:+.1f}")


def test_dummies_coding_returns_female_coefficient_only():
    df, _ = _fake(n=8000)
    df["EDCAT8_TC1"] = np.clip(np.round(df["YRSQUALC2"] / 2 + 4), 1, 8)
    pv = [f"PVLIT{i}" for i in range(1, 11)]
    r = differential_validity(df, "EDCAT8_TC1", pv, proxy_dummies=True)
    assert r["b_female"] is not None and r["benchmark"] is None
    print("  ok: dummy coding reports the female coefficient without a benchmark")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
