"""TDD tests for secom statistical-rigor helpers:
leakage-free operating-point selection, DeLong AUC test, bootstrap AUC CI."""
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import rigor  # noqa: E402


def test_select_threshold_maximizes_recall_minus_fa_above_floor():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.3, 0.1, 0.4])
    # thr 0.5 -> recall 1.0, FA 0.0 (best); thr 0.1 -> recall 1.0, FA 1.0; 0.85 -> recall 0.5 (below floor)
    thr = rigor.select_threshold(y, p, grid=[0.1, 0.5, 0.85], recall_floor=0.70)
    assert thr == 0.5


def test_select_threshold_falls_back_when_floor_unreachable():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.2, 0.1, 0.05, 0.05])
    # no threshold in grid reaches recall>=0.99 floor except very low; fallback picks max recall
    thr = rigor.select_threshold(y, p, grid=[0.5, 0.9], recall_floor=0.99)
    assert thr in (0.5, 0.9)


def test_delong_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    auc1, auc2, _ = rigor.delong_roc_test(y, p, p)
    assert np.isclose(auc1, roc_auc_score(y, p), atol=1e-9)
    assert np.isclose(auc1, auc2)


def test_delong_identical_predictions_give_p_near_one():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 150)
    p = rng.random(150)
    _, _, pval = rigor.delong_roc_test(y, p, p)
    assert pval > 0.99


def test_delong_better_model_has_small_pvalue():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    noise = rng.normal(0, 0.3, 400)
    p_good = np.clip(y + noise, 0, 1)      # tracks the label well
    p_bad = rng.random(400)                # random
    _, _, pval = rigor.delong_roc_test(y, p_bad, p_good)
    assert 0.0 <= pval <= 1.0 and pval < 0.05


def test_bootstrap_auc_ci_brackets_point_estimate():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    noise = rng.normal(0, 0.4, 300)
    p = np.clip(y + noise, 0, 1)
    point = roc_auc_score(y, p)
    lo, hi = rigor.bootstrap_auc_ci(y, p, n_boot=500, seed=4)
    assert lo <= point <= hi
