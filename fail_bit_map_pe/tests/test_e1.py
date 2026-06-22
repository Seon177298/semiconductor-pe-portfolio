"""TDD test for E1 synthetic-limit dataset generator (shape/label contract)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import e1_synthetic_limits as e1  # noqa: E402
import fbm_core as c  # noqa: E402


def test_gen_dataset_shapes_and_label_range():
    X, maps, y = e1.gen_dataset(np.random.default_rng(0), 40, sigma_meas=0.02, mix_frac=0.0)
    assert X.shape == (40, len(c.FEATURE_NAMES))
    assert maps.shape == (40, c.GRID // 4, c.GRID // 4)
    assert int(y.min()) >= 0 and int(y.max()) < len(c.LABELS)


def test_overlay_mixing_never_reduces_fail_count():
    # mixing OR-overlays a second pattern onto the same base die -> fails only added
    rng = np.random.default_rng(3)
    m0, lk, _ = c.make_die("ROW", rng)
    det = c.margin_at(m0, lk, c.T_TEST_DEFAULT) < 0.0
    det_mixed = e1.overlay_second_pattern(det, "CLUSTER", np.random.default_rng(4), sigma_meas=0.0)
    assert int(det_mixed.sum()) >= int(det.sum())
