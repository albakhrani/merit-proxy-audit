# -*- coding: utf-8 -*-
"""The arms added after the internal review of 22 Sep 2026 (REGISTERED_CHANGES
R4-R12) on a synthetic ACS-shaped file: every option changes what it should and nothing else, the
defaults reproduce the registered cell bit for bit, the person-level fold
assignment never splits a person, and the log-income decomposition is
invariant to the ADJINC factor."""
import json, os, sys, tempfile
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from study2lib import specs
from study2lib.acs import load_cell, weighted_median, occp_major_group
from study2lib.cellrun import run_cell
from study2lib.learners import folds_by_person
from study2lib.decomposition import twofold_neumark


def _fake_acs(root, state="WY", year=2022, n=6000, seed=31, wkwn=True):
    rng = np.random.default_rng(seed)
    merit = rng.normal(0, 1, n)
    sex = rng.integers(1, 3, n)
    schl = np.clip(np.round(16 + 3 * merit + rng.normal(0, 2, n)), 1, 24)
    agep = rng.integers(18, 80, n)
    inc = np.exp(10.2 + 0.5 * merit + 0.01 * (agep - 40)
                 - 0.03 * (sex == 2) + rng.normal(0, 0.6, n))
    wkhp = rng.integers(10, 60, n).astype(float)
    cow = rng.integers(1, 6, n).astype(float)
    occp = rng.choice([10, 430, 800, 1010, 2310, 3255, 4700, 5120, 6230, 9130, 9830], n).astype(float)
    df = pd.DataFrame({"AGEP": agep, "SCHL": schl, "SEX": sex, "PINCP": inc,
                       "WAGP": inc * rng.uniform(0.5, 1.0, n), "PWGTP": rng.integers(1, 50, n),
                       "WKHP": wkhp, "COW": cow, "OCCP": occp,
                       "ADJINC": np.full(n, 1052000.0)})
    if wkwn:
        df["WKWN"] = rng.integers(1, 53, n).astype(float)
    else:
        df["WKW"] = rng.integers(1, 7, n).astype(float)
    d = os.path.join(root, str(year), "1-Year")
    os.makedirs(d, exist_ok=True)
    df.to_csv(os.path.join(d, "psam_p56.csv"), index=False)
    return df


def test_defaults_unchanged_and_options_act():
    with tempfile.TemporaryDirectory() as root:
        raw = _fake_acs(root)
        base = load_cell(root, "WY", 2022, ["AGEP", "SCHL"], "acs50k")
        assert base["options"] == {}
        # years coding: same rows, SCHL values pass through the crosswalk
        yrs = load_cell(root, "WY", 2022, ["AGEP", "SCHL"], "acs50k", schooling="years")
        assert yrs["n"] == base["n"]
        j = base["names"].index("SCHL")
        mapped = np.array([specs.SCHL_TO_YEARS[int(c)] for c in base["X"][:, j]])
        assert np.array_equal(mapped, yrs["X"][:, j])
        assert yrs["options"] == {"schooling": "years"}
        # wage arm: WAGP outcome, full-time full-year sample
        wage = load_cell(root, "WY", 2022, ["AGEP", "SCHL"], "acs50k",
                         income_var="WAGP", fulltime=True)
        keep = (raw["AGEP"] >= 18) & (raw["WAGP"] > 0) & (raw["WKHP"] >= 35) & (raw["WKWN"] >= 50)
        assert wage["n"] == int(keep.sum())
        assert np.allclose(np.sort(wage["y_cont"]), np.sort(raw.loc[keep, "WAGP"]))
        # occupation arm: major-group dummies appended
        occ = load_cell(root, "WY", 2022, ["AGEP", "SCHL", "WKHP", "COW"], "acs50k",
                        occupation=True)
        occ_cols = [c for c in occ["names"] if c.startswith("OCC_")]
        assert len(occ_cols) == len(set(occp_major_group(c) for c in raw["OCCP"])) - 1
        assert occp_major_group(9830) == "military_or_none" and occp_major_group(2310) == "education"
        # ADJINC: threshold changes, sample does not
        adj = load_cell(root, "WY", 2022, ["AGEP", "SCHL"], "acs50k", adjinc=True)
        assert adj["n"] == base["n"]
        assert np.allclose(adj["y_cont"], base["y_cont"] * 1.052)
        assert adj["y"].sum() >= base["y"].sum()
        # weighted median
        wm = load_cell(root, "WY", 2022, ["AGEP", "SCHL"], "cellmedian",
                       weighted_median_threshold=True)
        thr = weighted_median(base["y_cont"], base["w"])
        assert np.array_equal(wm["y"], (base["y_cont"] > thr).astype(float))
        assert weighted_median([1, 2, 3, 100], [1, 1, 1, 1]) == 2.0
        assert weighted_median([1, 2, 3, 100], [1, 1, 1, 10]) == 100.0
        # age range
        ag = load_cell(root, "WY", 2022, ["AGEP", "SCHL"], "acs50k", age_range=(25, 65))
        i = base["names"].index("AGEP")
        assert ag["X"][:, i].min() >= 25 and ag["X"][:, i].max() <= 65
        # mean-error shift moves women only
        sh = load_cell(root, "WY", 2022, ["AGEP", "SCHL"], "acs50k", shift_female_schl=0.5)
        d = sh["X"][:, j] - base["X"][:, j]
        assert np.allclose(d[base["g"] == 1], 0.5) and np.allclose(d[base["g"] == 0], 0.0)
    print("  ok: load_cell options act as documented; defaults reproduce the registered frame")


def test_wkw_fallback_before_2019():
    with tempfile.TemporaryDirectory() as root:
        raw = _fake_acs(root, year=2018, wkwn=False)
        wage = load_cell(root, "WY", 2018, ["AGEP", "SCHL"], "acs50k",
                         income_var="WAGP", fulltime=True)
        keep = (raw["AGEP"] >= 18) & (raw["WAGP"] > 0) & (raw["WKHP"] >= 35) & (raw["WKW"] == 1)
        assert wage["n"] == int(keep.sum())
    print("  ok: WKW == 1 used where WKWN is absent")


def test_run_cell_with_options_and_exact_anchors():
    with tempfile.TemporaryDirectory() as root:
        _fake_acs(root)
        res = run_cell(root, "WY", 2022, kappa_point=0.187, n_boot=20, learners=["logistic"],
                       load_kwargs={"schooling": "years"}, extra_kappas=[0.187, 0.243])
        json.dumps(res)
        assert res["load_options"] == {"schooling": "years"}
        r = res["spec_results"][0]
        assert set(r["eiv_unexplained_at"]) == {"0.187", "0.243"}
        assert abs(r["eiv_unexplained_at"]["0.187"] - r["eiv_unexplained_at_kappa"]) < 1e-12
        # registered call unchanged in structure
        res0 = run_cell(root, "WY", 2022, kappa_point=0.187, n_boot=20, learners=["logistic"])
        assert res0["load_options"] == {} and res0["spec_results"][0]["eiv_unexplained_at"] == {}
    print("  ok: run_cell carries the options and the exact anchors")


def test_decomp_only_matches_full_cell():
    """23 Sep 2026: the decomposition-only cell (no learners, no DML, no R
    bootstrap) carries exactly the decomposition fields of the full cell."""
    with tempfile.TemporaryDirectory() as root:
        _fake_acs(root)
        kw = {"schooling": "years", "shift_female_schl": -0.3}
        full = run_cell(root, "WY", 2022, kappa_point=0.187, n_boot=20, learners=["logistic"],
                        load_kwargs=kw, extra_kappas=[0.187, 0.243])
        fast = run_cell(root, "WY", 2022, kappa_point=0.187, load_kwargs=kw,
                        extra_kappas=[0.187, 0.243], decomp_only=True)
        json.dumps(fast)
        assert len(fast["spec_results"]) == len(full["spec_results"])
        assert fast["load_options"] == full["load_options"]
        for a, b in zip(fast["spec_results"], full["spec_results"]):
            assert (a["outcome_def"], a["covariate_set"]) == (b["outcome_def"], b["covariate_set"])
            assert a["learner"] == "none" and a["R"] is None and a["dml_theta_logpts"] is None
            assert a["decomp_only"] is True
            for fld in ("n", "n_female", "n_male", "decomp_raw_logpts", "decomp_explained",
                        "decomp_unexplained", "eiv_unexplained_at_kappa", "kappa_point",
                        "n_covariate_columns"):
                assert a[fld] == b[fld], fld
            assert a["eiv_unexplained_at"] == b["eiv_unexplained_at"]
            assert all(p["unexplained"] == q["unexplained"] for p, q in zip(a["kappa_sweep"], b["kappa_sweep"]))
        assert fast["timing_s"] < full["timing_s"]
    print("  ok: decomposition-only cells equal the full cell's decomposition fields")


def test_eiv_admissibility_threshold_and_fallback():
    """25 Sep 2026: the correction exists only above the proxy's weighted R^2
    on the other covariates, and a cell below that threshold reports None with
    the threshold rather than failing."""
    from study2lib.eiv import min_admissible_kappa, eiv_correct
    rng = np.random.default_rng(17)
    n = 4000
    X = rng.normal(0, 1, (n, 5))
    X[:, 1] = 0.8 * X[:, 2] - 0.3 * X[:, 3] + 0.45 * rng.normal(0, 1, n)
    w = rng.integers(1, 25, n).astype(float)
    y = X @ np.array([0.2, 0.5, 0.1, -0.2, 0.3]) + rng.normal(0, 1, n)
    kmin = min_admissible_kappa(X, 1, w)
    # equals the weighted R^2 of column 1 on the others
    wm = w / w.sum()
    Xc = X - wm @ X
    A = np.column_stack([np.ones(n), np.delete(Xc, 1, axis=1)])
    sw = np.sqrt(wm)[:, None]
    beta = np.linalg.lstsq(A * sw, Xc[:, 1] * np.sqrt(wm), rcond=None)[0]
    r2 = 1 - np.average((Xc[:, 1] - A @ beta) ** 2, weights=wm) / np.average(Xc[:, 1] ** 2, weights=wm)
    assert abs(kmin - r2) < 1e-8, (kmin, r2)
    # just above the threshold the solve succeeds, just below it raises
    kap_hi = np.ones(5); kap_hi[1] = min(kmin + 1e-3, 1.0)
    eiv_correct(X, y, kap_hi, w)
    kap_lo = np.ones(5); kap_lo[1] = max(kmin - 1e-3, 1e-6)
    try:
        eiv_correct(X, y, kap_lo, w)
        raise AssertionError("expected LinAlgError below the threshold")
    except np.linalg.LinAlgError:
        pass
    # a cell whose proxy is collinear with the covariates records the diagnosis
    with tempfile.TemporaryDirectory() as root:
        raw = _fake_acs(root)
        df = raw.copy()
        # an extra covariate that nearly determines schooling
        df["WKHP"] = df["SCHL"] * 2.0 + np.random.default_rng(5).normal(0, 0.05, len(df))
        d = os.path.join(root, "2022", "1-Year")
        df.to_csv(os.path.join(d, "psam_p56.csv"), index=False)
        res = run_cell(root, "WY", 2022, kappa_point=0.187, load_kwargs={}, decomp_only=True,
                       covsets={"collinear": ["AGEP", "SCHL", "WKHP"]}, extra_kappas=[0.187])
        r = res["spec_results"][0]
        assert r["eiv_min_kappa"] > 0.187, r["eiv_min_kappa"]
        assert r["eiv_correctable"] is False
        assert r["eiv_unexplained_at_kappa"] is None
        assert r["eiv_unexplained_at"]["0.187"] is None
        assert r["decomp_unexplained"] == r["decomp_raw_logpts"] - r["decomp_explained"]
        above = [p for p in r["kappa_sweep"] if p["kappa"] > r["eiv_min_kappa"]]
        below = [p for p in r["kappa_sweep"] if p["kappa"] < r["eiv_min_kappa"]]
        assert all(p["unexplained"] is not None for p in above), "admissible sweep points must be computed"
        assert all(p["unexplained"] is None for p in below), "inadmissible sweep points must be None"
        json.dumps(res)
    print("  ok: the admissible reliability is the proxy's R^2 and an inadmissible cell is diagnosed, not failed")


def test_folds_by_person_never_split_a_person():
    rng = np.random.default_rng(3)
    take = rng.integers(0, 500, 2000)              # a resample with many duplicates
    folds = folds_by_person(take, n_folds=5, seed=11)
    assert sorted(np.concatenate(folds).tolist()) == list(range(2000))
    owner = {}
    for i, f in enumerate(folds):
        for row in f:
            p = take[row]
            assert owner.setdefault(p, i) == i, "a person appears in two folds"
    same = folds_by_person(take, n_folds=5, seed=11)
    assert all(np.array_equal(a, b) for a, b in zip(folds, same))   # deterministic
    print("  ok: person-level folds are exhaustive, disjoint by person and seeded")


def test_log_decomposition_invariant_to_adjinc():
    rng = np.random.default_rng(4)
    n = 3000
    X = rng.normal(0, 1, (n, 2)); g = rng.integers(0, 2, n)
    y = np.exp(10 + 0.3 * X[:, 0] + 0.1 * X[:, 1] - 0.05 * g + rng.normal(0, 0.5, n))
    w = rng.integers(1, 20, n).astype(float)
    a = twofold_neumark(X[g == 1], np.log(y[g == 1]), X[g == 0], np.log(y[g == 0]), w[g == 1], w[g == 0])
    b = twofold_neumark(X[g == 1], np.log(1.052 * y[g == 1]), X[g == 0], np.log(1.052 * y[g == 0]),
                        w[g == 1], w[g == 0])
    assert abs(a["unexplained"] - b["unexplained"]) < 1e-9 and abs(a["explained"] - b["explained"]) < 1e-9
    print("  ok: a year-constant income factor leaves the log decomposition unchanged")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
