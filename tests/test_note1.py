# -*- coding: utf-8 -*-
"""Supplementary Note 1: group-specific reliabilities under a pooled
reference reduce to one common reliability.

Added 16 September 2026, when the manuscript began to rest on the result.
Three checks on synthetic data with known answers:

  1. the pooled error variance is the weight-average of the within-group
     error variances, so kappa* of Note 1 equation (5) implies exactly the
     error variance the two group reliabilities imply;
  2. feeding kappa* to eiv_correct recovers the true coefficient on the
     mismeasured column, which ordinary least squares attenuates;
  3. kappa* is bounded below by the variance-weighted mean of the group
     reliabilities, and its derivative with respect to the differential
     matches the closed form.

These are the claims the paper makes when it treats the differential
channel as closed. If any of them fails, Supplementary Note 1 is wrong.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from study2lib.eiv import eiv_correct

TOL_EXACT = 1e-10
PASSED = []


def _sample(seed=20261001, nf=60000, nm=50000, sf=1.9, sm=1.1):
    rng = np.random.default_rng(seed)
    Mf = rng.normal(13.0, 2.4, nf); Mm = rng.normal(12.2, 3.1, nm)
    Xf = Mf + rng.normal(0, sf, nf); Xm = Mm + rng.normal(0, sm, nm)
    X = np.concatenate([Xf, Xm]); M = np.concatenate([Mf, Mm])
    g = np.concatenate([np.ones(nf), np.zeros(nm)])
    age = rng.normal(42, 10, nf + nm)
    beta = np.array([0.11, 0.004, -0.18])
    y = beta[0] * M + beta[1] * age + beta[2] * g + rng.normal(0, 0.5, nf + nm)
    pf = nf / (nf + nm)
    return dict(X=X, M=M, g=g, age=age, y=y, beta=beta, pf=pf, pm=1 - pf,
                Vf=Xf.var(), Vm=Xm.var(), mf=Xf.mean(), mm=Xm.mean(),
                kf=Mf.var() / Xf.var(), km=Mm.var() / Xm.var(), Vp=X.var())


def _kstar(d):
    return 1 - (d["pf"] * (1 - d["kf"]) * d["Vf"]
                + d["pm"] * (1 - d["km"]) * d["Vm"]) / d["Vp"]


def test_pooled_variance_identity():
    d = _sample()
    formula = (d["pf"] * d["Vf"] + d["pm"] * d["Vm"]
               + d["pf"] * d["pm"] * (d["mf"] - d["mm"]) ** 2)
    print("    eq(1) pooled variance residual      %.3e" % abs(formula - d["Vp"]))
    assert abs(formula - d["Vp"]) < 1e-9, "Note 1 eq (1) fails"
    ks = _kstar(d)
    implied = d["pf"] * (1 - d["kf"]) * d["Vf"] + d["pm"] * (1 - d["km"]) * d["Vm"]
    print("    eq(5) kappa* equivalence residual   %.3e" % abs((1 - ks) * d["Vp"] - implied))
    assert abs((1 - ks) * d["Vp"] - implied) < TOL_EXACT, "Note 1 eq (5) fails"
    PASSED.append("pooled variance identity and kappa* equivalence")


def test_kstar_recovers_the_truth():
    d = _sample()
    Xg = np.column_stack([d["X"], d["age"], d["g"]])
    w = np.ones(len(d["y"]))
    b_ols = eiv_correct(Xg, d["y"], [1.0, 1.0, 1.0], w)[1]
    b_cor = eiv_correct(Xg, d["y"], [_kstar(d), 1.0, 1.0], w)[1]
    true = d["beta"][0]
    print("    coefficient on the mismeasured column: true %.6f | OLS %.6f (error %.6f)"
          " | corrected at kappa* %.6f (error %.6f)"
          % (true, b_ols, abs(b_ols - true), b_cor, abs(b_cor - true)))
    assert abs(b_ols - true) > 10 * abs(b_cor - true), \
        "correction at kappa* should beat OLS by an order of magnitude"
    assert abs(b_cor - true) < 0.005, "correction at kappa* does not recover the truth"
    PASSED.append("kappa* recovers the true coefficient; OLS attenuates")


def test_bound_and_sensitivity():
    d = _sample()
    W = d["pf"] * d["Vf"] + d["pm"] * d["Vm"]
    kbar_V = (d["pf"] * d["kf"] * d["Vf"] + d["pm"] * d["km"] * d["Vm"]) / W
    assert _kstar(d) >= kbar_V - TOL_EXACT, "Note 1 eq (6) bound violated"
    kbar = (d["kf"] + d["km"]) / 2

    def ks_of_d(delta):
        return 1 - (d["pf"] * (1 - (kbar - delta / 2)) * d["Vf"]
                    + d["pm"] * (1 - (kbar + delta / 2)) * d["Vm"]) / d["Vp"]

    h = 1e-5; d0 = d["km"] - d["kf"]
    fd = (ks_of_d(d0 + h) - ks_of_d(d0 - h)) / (2 * h)
    analytic = -(d["pf"] * d["Vf"] - d["pm"] * d["Vm"]) / (2 * d["Vp"])
    print("    eq(6) bound   kappa* %.6f >= kbar_V %.6f" % (_kstar(d), kbar_V))
    print("    eq(7) sensitivity analytic %.8f vs finite difference %.8f (diff %.3e)"
          % (analytic, fd, abs(fd - analytic)))
    assert abs(fd - analytic) < 1e-8, "Note 1 eq (7) sensitivity fails"
    PASSED.append("bound and sensitivity match their closed forms")


if __name__ == "__main__":
    test_pooled_variance_identity()
    test_kstar_recovers_the_truth()
    test_bound_and_sensitivity()
    for p in PASSED:
        print("  ok:", p)
    print("test_note1: %d checks passed" % len(PASSED))
