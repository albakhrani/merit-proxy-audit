# -*- coding: utf-8 -*-
"""The arms added after the pre-submission review of 28 Sep 2026
(REGISTERED_CHANGES R14 to R18) on synthetic data with known answers:
the proportional shift equals the fixed shift when share times lead equals
delta, the composite plausible values are the mean of the two skills, the
successive-difference replication variance matches a known replicate
spread, the two-reference decomposition adds up for each reference and
reduces to the pooled result when the sex regressions share their
coefficients, and the indicator-coded decomposition sums to the raw gap."""
import os, sys, tempfile
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from study2lib import specs
from study2lib.acs import load_cell
from study2lib.piaac import add_composite_pvs, PV_LIT, PV_NUM, PV_COMPOSITE
from study2lib.bootstrap import successive_difference_variance
from study2lib.decomposition import twofold_neumark, twofold_groupref, indicator_columns


def _fake_acs(root, state="WY", year=2022, n=5000, seed=31):
    rng = np.random.default_rng(seed)
    merit = rng.normal(0, 1, n)
    sex = rng.integers(1, 3, n)
    schl = np.clip(np.round(16 + 3 * merit + rng.normal(0, 2, n) + 0.4 * (sex == 2)), 1, 24)
    agep = rng.integers(18, 80, n)
    inc = np.exp(10.2 + 0.5 * merit + 0.01 * (agep - 40) - 0.03 * (sex == 2) + rng.normal(0, 0.6, n))
    df = pd.DataFrame({"AGEP": agep, "SCHL": schl, "SEX": sex, "PINCP": inc,
                       "PWGTP": rng.integers(1, 50, n), "WKHP": rng.integers(10, 60, n).astype(float),
                       "COW": rng.integers(1, 6, n).astype(float)})
    for r in range(1, 81):
        df[f"PWGTP{r}"] = df["PWGTP"] * rng.uniform(0.5, 1.5, n)
    d = os.path.join(root, str(year), "1-Year")
    os.makedirs(d, exist_ok=True)
    df.to_csv(os.path.join(d, "psam_p56.csv"), index=False)
    return df


def test_proportional_shift_equals_fixed_shift():
    with tempfile.TemporaryDirectory() as root:
        raw = _fake_acs(root)
        prop = load_cell(root, "WY", 2022, ["AGEP", "SCHL"], "acs50k", schooling="years",
                         shift_female_share=0.8)
        lead = prop["options"]["schooling_lead_years"]
        yrs = raw["SCHL"].astype(int).map(specs.SCHL_TO_YEARS).astype(float)
        keep = (raw["AGEP"] >= 18) & (raw["PINCP"] > 0)
        w = raw["PWGTP"].astype(float)
        expect = (np.average(yrs[keep & (raw["SEX"] == 2)], weights=w[keep & (raw["SEX"] == 2)])
                  - np.average(yrs[keep & (raw["SEX"] == 1)], weights=w[keep & (raw["SEX"] == 1)]))
        assert abs(lead - expect) < 1e-12
        assert abs(prop["options"]["shift_female_schl_applied"] + 0.8 * lead) < 1e-12
        fixed = load_cell(root, "WY", 2022, ["AGEP", "SCHL"], "acs50k", schooling="years",
                          shift_female_schl=-0.8 * lead)
        assert np.allclose(prop["X"], fixed["X"]) and prop["n"] == fixed["n"]
        base = load_cell(root, "WY", 2022, ["AGEP", "SCHL"], "acs50k", schooling="years")
        assert base["options"] == {"schooling": "years"}
        try:
            load_cell(root, "WY", 2022, ["AGEP", "SCHL"], "acs50k", schooling="years",
                      shift_female_share=0.5, shift_female_schl=-0.1)
            raise AssertionError("mutual exclusion not enforced")
        except ValueError:
            pass
        extra = load_cell(root, "WY", 2022, ["AGEP", "SCHL"], "acs50k", extra_columns=["PWGTP1", "PWGTP80"])
        assert extra["extra"].shape == (extra["n"], 2) and extra["options"] == {}
    print(f"  ok: share * lead reproduces the fixed shift (lead {lead:.4f} years); defaults unchanged")


def test_composite_plausible_values():
    rng = np.random.default_rng(5)
    n = 200
    df = pd.DataFrame({c: rng.normal(270, 40, n) for c in PV_LIT + PV_NUM})
    out = add_composite_pvs(df)
    for lit, num, comp in zip(PV_LIT, PV_NUM, PV_COMPOSITE):
        assert np.allclose(out[comp], (df[lit] + df[num]) / 2)
    assert list(df.columns) == PV_LIT + PV_NUM        # the input frame is not modified
    print("  ok: composite plausible value k equals the mean of PVLIT k and PVNUM k")


def test_successive_difference_variance():
    rng = np.random.default_rng(9)
    theta = 0.4
    reps = theta + rng.normal(0, 0.05, 80)
    v = successive_difference_variance(theta, reps, 80)
    assert abs(v - 4.0 / 80 * np.sum((reps - theta) ** 2)) < 1e-15
    # a replicate spread of exactly s in every replicate gives variance 4 s^2
    reps2 = theta + 0.02 * np.where(np.arange(80) % 2 == 0, 1.0, -1.0)
    assert abs(successive_difference_variance(theta, reps2, 80) - 4 * 0.02 ** 2) < 1e-15
    print("  ok: replication variance equals (4/80) times the summed squared replicate deviations")


def test_groupref_adds_up_and_reduces_to_pooled():
    rng = np.random.default_rng(2)
    n = 4000
    X = np.column_stack([rng.normal(40, 10, n), rng.integers(8, 22, n).astype(float)])
    g = rng.integers(0, 2, n); w = rng.integers(1, 30, n).astype(float)
    y = 9 + 0.01 * X[:, 0] + 0.08 * X[:, 1] - 0.1 * g + rng.normal(0, 0.5, n)
    r = twofold_groupref(X[g == 1], y[g == 1], X[g == 0], y[g == 0], w[g == 1], w[g == 0])
    assert abs(r["raw"] - (r["explained_ref_b"] + r["unexplained_ref_b"])) < 1e-10
    assert abs(r["raw"] - (r["explained_ref_a"] + r["unexplained_ref_a"])) < 1e-10
    # corrected: the identity holds for each reference and each anchor choice
    rc = twofold_groupref(X[g == 1], y[g == 1], X[g == 0], y[g == 0], w[g == 1], w[g == 0],
                          proxy_index=1, kappa_a=0.6, kappa_b=0.7)
    assert abs(rc["raw"] - (rc["explained_ref_b"] + rc["unexplained_ref_b"])) < 1e-10
    assert abs(rc["raw"] - (rc["explained_ref_a"] + rc["unexplained_ref_a"])) < 1e-10
    assert rc["beta_a"][2] > r["beta_a"][2] and rc["beta_b"][2] > r["beta_b"][2]   # attenuation removed
    # identical coefficients in both groups: every reference equals the pooled result
    y_exact = 9 + 0.01 * X[:, 0] + 0.08 * X[:, 1] - 0.1 * g
    r2 = twofold_groupref(X[g == 1], y_exact[g == 1], X[g == 0], y_exact[g == 0], w[g == 1], w[g == 0])
    pooled = twofold_neumark(X[g == 1], y_exact[g == 1], X[g == 0], y_exact[g == 0], w[g == 1], w[g == 0])
    for k in ("explained_ref_a", "explained_ref_b"):
        assert abs(r2[k] - pooled["explained"]) < 1e-8
    print("  ok: two-reference decomposition adds up under each reference and reduces to the pooled result")


def test_indicator_decomposition_sums_to_raw_gap():
    rng = np.random.default_rng(3)
    n = 3000
    schl = rng.integers(1, 25, n); age = rng.integers(18, 80, n).astype(float)
    g = rng.integers(0, 2, n); w = rng.integers(1, 20, n).astype(float)
    y = 9 + 0.05 * schl + 0.005 * age - 0.1 * g + rng.normal(0, 0.4, n)
    D, levels = indicator_columns(schl)
    assert D.shape == (n, len(np.unique(schl)) - 1) and levels == [float(v) for v in np.unique(schl)[1:]]
    assert np.all(D.sum(axis=1) <= 1) and np.all(D.sum(axis=1)[schl == schl.min()] == 0)
    Xd = np.column_stack([age, D])
    dec = twofold_neumark(Xd[g == 1], y[g == 1], Xd[g == 0], y[g == 0], w[g == 1], w[g == 0])
    raw = np.average(y[g == 1], weights=w[g == 1]) - np.average(y[g == 0], weights=w[g == 0])
    assert abs(dec["raw"] - raw) < 1e-12 and abs(dec["raw"] - (dec["explained"] + dec["unexplained"])) < 1e-10
    print(f"  ok: indicator-coded decomposition sums to the raw gap ({len(levels) + 1} levels)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
