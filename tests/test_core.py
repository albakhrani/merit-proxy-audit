# -*- coding: utf-8 -*-
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from study2lib import (twofold_neumark, eiv_correct, estimate_kappa, mrr,
                       stratified_bootstrap, benjamini_hochberg)
from study2lib.synthetic import make_cell


def test_decomposition_identity():
    d = make_cell(seed=1)
    ga = d["g"] == 1
    r = twofold_neumark(d["x"][ga, None], d["y_star"][ga],
                        d["x"][~ga, None], d["y_star"][~ga])
    assert abs(r["raw"] - (r["explained"] + r["unexplained"])) < 1e-10


def test_kappa_recovery_and_group_difference():
    d = make_cell(n=200000, kappa_a=0.85, kappa_b=0.65, seed=2)
    ka = estimate_kappa(d["x"][d["g"] == 1], d["assessed"][d["g"] == 1])
    kb = estimate_kappa(d["x"][d["g"] == 0], d["assessed"][d["g"] == 0])
    assert abs(ka - 0.85) < 0.02 and abs(kb - 0.65) < 0.02


def test_eiv_reduces_to_ols_at_kappa_one():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(5000, 2)); y = X @ np.array([0.5, -0.2]) + rng.normal(size=5000)
    b1 = eiv_correct(X, y, [1.0, 1.0])
    Xc = np.column_stack([np.ones(5000), X])
    b0, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    assert np.allclose(b1, b0, atol=1e-8)


def test_eiv_recovers_true_coefficient():
    d = make_cell(n=300000, kappa_a=0.7, kappa_b=0.7, beta=1.0, seed=4)
    b_naive = np.polyfit(d["x"], d["y_star"], 1)[0]
    assert b_naive < 0.85                    # attenuated
    b_corr = eiv_correct(d["x"][:, None], d["y_star"], [0.7])
    assert abs(b_corr[1] - 1.0) < 0.03       # recovered


def test_mrr_signs_and_seeded():
    d = make_cell(n=30000, gap_label=-0.05, seed=5)
    X = d["x"][:, None]
    r1 = mrr(X, d["y"], d["g"], seed=7)
    r2 = mrr(X, d["y"], d["g"], seed=7)
    assert r1["R"] == r2["R"]                # deterministic given seed
    assert abs(r1["R"] - (r1["delta_merit"] - r1["delta_label"])) < 1e-12


def test_bootstrap_and_fdr():
    d = make_cell(n=4000, seed=6)
    fn = lambda idx: float(np.mean(d["y"][idx][d["g"][idx] == 1])
                           - np.mean(d["y"][idx][d["g"][idx] == 0]))
    r = stratified_bootstrap(fn, d["g"], n_boot=200, seed=8)
    assert r["lo"] <= r["point"] <= r["hi"]
    p = np.array([0.001, 0.01, 0.02, 0.2, 0.8])
    rej = benjamini_hochberg(p, q=0.05)
    assert rej[0] and not rej[4]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print(f"  pass  {f.__name__}")
    print(f"{len(fns)} tests passed")
