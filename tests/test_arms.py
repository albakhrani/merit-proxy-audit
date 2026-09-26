# -*- coding: utf-8 -*-
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from study2lib.dml import dml_gap
from study2lib.mrr import mrr
from study2lib.learners import oof_predictions, ArmUnavailable
from study2lib.synthetic import make_cell


def test_dml_recovers_gap_under_nonlinearity():
    rng = np.random.default_rng(11)
    n = 40000
    X = rng.normal(size=(n, 2))
    g = (rng.random(n) < 0.5).astype(float)
    theta = -0.04
    y = theta * g + np.sin(1.5 * X[:, 0]) + 0.5 * X[:, 1] ** 2 + rng.normal(0, 0.5, n)
    naive = np.linalg.lstsq(np.column_stack([np.ones(n), g, X]), y, rcond=None)[0][1]
    r = dml_gap(X, y, g, seed=3, nuisance="sklearn_gbm")
    assert abs(r["theta"] - theta) < 3 * r["se"] + 0.005
    assert abs(r["theta"] - theta) <= abs(naive - theta) + 0.01


def test_dml_ridge_fallback_runs():
    rng = np.random.default_rng(12)
    n = 5000
    X = rng.normal(size=(n, 2)); g = (rng.random(n) < 0.5).astype(float)
    y = 0.1 * g + X[:, 0] + rng.normal(0, 0.3, n)
    r = dml_gap(X, y, g, seed=4, nuisance="ridge_poly")
    assert abs(r["theta"] - 0.1) < 0.05


def test_learner_interface_logistic_matches_mrr():
    d = make_cell(n=8000, seed=13)
    X = d["x"][:, None]
    m1 = mrr(X, d["y"], d["g"], seed=5, method="logistic")
    oof = oof_predictions(X, d["y"], np.ones(len(d["y"])), "logistic", seed=5)
    assert np.allclose(m1["oof"], oof)


def test_unavailable_arm_raises_cleanly():
    d = make_cell(n=500, seed=14)
    try:
        import tabpfn  # noqa: F401
        return  # installed here; the skip path is exercised on machines without it
    except ImportError:
        pass
    try:
        mrr(d["x"][:, None], d["y"], d["g"], seed=6, method="tabpfn")
        raise AssertionError("expected ArmUnavailable")
    except ArmUnavailable:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print(f"  pass  {f.__name__}")
    print(f"{len(fns)} tests passed")
