"""TDD tests for E2 uncertainty helpers (percentile CI + bootstrap resampling)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import e2_uncertainty as e2  # noqa: E402


def test_ci95_returns_2p5_and_97p5_percentiles():
    lo, hi = e2.ci95(np.arange(0, 101))      # 0..100 inclusive
    assert np.isclose(lo, 2.5) and np.isclose(hi, 97.5)


def test_ci95_brackets_the_mean_for_normal_data():
    rng = np.random.default_rng(0)
    x = rng.normal(10.0, 2.0, 5000)
    lo, hi = e2.ci95(x)
    assert lo < 10.0 < hi


def test_bootstrap_rates_ci_brackets_point_estimate():
    # per-item escape counts; bootstrap CI of the mean rate should bracket the point mean
    counts = np.array([0, 0, 0, 1, 0, 2, 0, 0, 1, 0], dtype=float)
    point = counts.mean()
    lo, hi = e2.bootstrap_mean_ci(counts, n_boot=2000, seed=1)
    assert lo <= point <= hi
