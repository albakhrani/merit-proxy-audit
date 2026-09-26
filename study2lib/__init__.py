"""Shared estimation library for Studies 2 and 3.

Every estimator is deterministic given a seed. No function fabricates a
default: reliabilities, weights and thresholds are always explicit arguments.
"""
from .decomposition import twofold_neumark
from .eiv import eiv_correct, estimate_kappa
from .mrr import mrr
from .bootstrap import stratified_bootstrap
from .fdr import benjamini_hochberg
from .dml import dml_gap
from .learners import oof_predictions, ArmUnavailable
