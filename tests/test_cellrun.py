# -*- coding: utf-8 -*-
"""End-to-end cell run on a synthetic ACS-shaped file placed in the layout
fetch_acs_cell() expects, so no network is touched."""
import json, os, sys, tempfile
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from study2lib.cellrun import run_cell
from study2lib import specs


def _fake_acs(root, state="WY", year=2022, n=6000, seed=31, missing_share=0.0):
    rng = np.random.default_rng(seed)
    merit = rng.normal(0, 1, n)
    sex = rng.integers(1, 3, n)
    schl = np.clip(np.round(16 + 3 * merit + rng.normal(0, 2, n)), 1, 24)
    agep = rng.integers(18, 65, n)
    inc = np.exp(10.2 + 0.5 * merit + 0.01 * (agep - 40)
                 - 0.03 * (sex == 2) + rng.normal(0, 0.6, n))
    wkhp = rng.integers(20, 60, n).astype(float)
    cow = rng.integers(1, 6, n).astype(float)
    if missing_share > 0:                    # people with income but no work record
        gone = rng.random(n) < missing_share
        wkhp[gone] = np.nan
        cow[gone] = np.nan
    df = pd.DataFrame({"AGEP": agep, "SCHL": schl, "SEX": sex,
                       "PINCP": inc, "PWGTP": np.ones(n),
                       "WKHP": wkhp, "COW": cow})
    d = os.path.join(root, str(year), "1-Year")
    os.makedirs(d, exist_ok=True)
    df.to_csv(os.path.join(d, "psam_p56.csv"), index=False)


def test_cell_end_to_end_and_json_serialisable():
    with tempfile.TemporaryDirectory() as root:
        _fake_acs(root)
        res = run_cell(root, "WY", 2022, kappa_point=0.75, n_boot=50)
        json.dumps(res)                       # fully serialisable
        prim = [r for r in res["spec_results"]
                if (r["outcome_def"], r["covariate_set"], r["learner"])
                == tuple(specs.BOOT_ARMS[0])][0]
        # identities and structure
        assert abs(prim["decomp_raw_logpts"]
                   - (prim["decomp_explained"] + prim["decomp_unexplained"])) < 1e-9
        assert prim["boot"] is not None                      # primary arm carries the interval
        assert prim["boot"]["lo"] <= prim["R"] <= prim["boot"]["hi"]
        assert prim["boot"]["learner_in_boot"] == prim["learner"]
        assert len(prim["kappa_sweep"]) == len(specs.KAPPA_GRID)
        assert prim["dml_nuisance"] in ("xgboost", "sklearn_gbm", "ridge_poly")
        # kappa=1 sweep entry equals the uncorrected decomposition residual
        at1 = [s for s in prim["kappa_sweep"] if s["kappa"] == 1.0][0]
        assert abs(at1["unexplained"] - prim["decomp_unexplained"]) < 1e-8
        # arms without their library are skipped, not fatal
        skipped = {s["learner"] for s in res["skipped_arms"]}
        for arm in ("xgboost", "tabpfn"):
            try:
                __import__(arm)
            except ImportError:
                assert arm in skipped


def test_sample_requires_decouples_sample_from_covariates():
    """The harmonised arm (R1): sample_requires holds the sample at the
    extended set's complete cases while conditioning stays premarket;
    defaults reproduce the T6 behaviour."""
    from study2lib.acs import load_cell
    with tempfile.TemporaryDirectory() as root:
        _fake_acs(root, missing_share=0.3)
        pre = load_cell(root, "WY", 2022, ["AGEP", "SCHL"], "acs50k")
        ext = load_cell(root, "WY", 2022, ["AGEP", "SCHL", "WKHP", "COW"], "acs50k")
        har = load_cell(root, "WY", 2022, ["AGEP", "SCHL"], "acs50k",
                        sample_requires=["WKHP", "COW"])
        assert pre["n"] > ext["n"]                     # the confound exists
        assert har["n"] == ext["n"]                    # harmonised = extended sample
        assert har["X"].shape[1] == pre["X"].shape[1]  # premarket covariates only
        pre2 = load_cell(root, "WY", 2022, ["AGEP", "SCHL"], "acs50k",
                         sample_requires=None)
        assert pre2["n"] == pre["n"]                   # defaults unchanged
        res_h = run_cell(root, "WY", 2022, kappa_point=0.75,
                         covsets={"premarket_h": ["AGEP", "SCHL"]},
                         sample_requires=["WKHP", "COW"])
        ns = {r["n"] for r in res_h["spec_results"]}
        assert ns == {har["n"]}                        # full path uses the held sample
        assert all(r["boot"] is None for r in res_h["spec_results"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print(f"  pass  {f.__name__}")
    print(f"{len(fns)} tests passed")
