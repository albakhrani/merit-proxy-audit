# -*- coding: utf-8 -*-
"""Benjamini-Hochberg false discovery rate control."""
from __future__ import annotations
import numpy as np


def benjamini_hochberg(pvals, q=0.05):
    """Returns a boolean rejection mask at level q (BH 1995)."""
    p = np.asarray(pvals, float)
    m = np.isfinite(p).sum()
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    ranked = p[order]
    thresh = q * (np.arange(1, len(p) + 1)) / m
    passed = ranked <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    reject = np.zeros(len(p), bool)
    reject[order[:k]] = True
    return reject
