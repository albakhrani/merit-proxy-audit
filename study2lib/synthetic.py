# -*- coding: utf-8 -*-
"""Synthetic data with known reliability, for the test suite only.

Never used in any reported estimate. Generates a latent merit construct,
group-specific proxies with chosen reliabilities, and an outcome from latent
merit, so every estimator's target quantity is known exactly.
"""
from __future__ import annotations
import numpy as np


def make_cell(n=20000, kappa_a=0.8, kappa_b=0.8, beta=1.0, gap_label=-0.03,
              share_a=0.4, seed=0):
    rng = np.random.default_rng(seed)
    g = (rng.random(n) < share_a).astype(int)          # 1 = group a
    m = rng.normal(0, 1, n)                            # latent merit
    var_m = 1.0
    nu_sd = np.where(g == 1, np.sqrt(var_m * (1 - kappa_a) / kappa_a),
                             np.sqrt(var_m * (1 - kappa_b) / kappa_b))
    x = m + rng.normal(0, 1, n) * nu_sd                # proxy
    y_star = beta * m + rng.normal(0, 0.5, n) + gap_label * g
    y = (y_star > np.quantile(y_star, 0.5)).astype(float)
    assessed = m + rng.normal(0, 0.05, n)              # near-gold instrument
    return {"g": g, "m": m, "x": x, "y": y, "y_star": y_star, "assessed": assessed}
