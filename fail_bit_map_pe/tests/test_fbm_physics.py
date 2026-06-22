"""TDD tests for the physical core extensions (Vdd-aware margin) and the
HBM KGD stacking-yield math. Run: `python -m pytest tests/ -q` from the
fail_bit_map_pe/ directory."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fbm_core as c  # noqa: E402
import kgd_stacking as kgd  # noqa: E402


# ---- Vdd-aware physical margin (E3 shmoo needs a supply-voltage axis) ----
def test_margin_at_default_vdd_matches_baseline_formula():
    m0 = np.array([[0.20, 0.10]])
    leak = np.array([[0.005, 0.0]])
    expected = m0 - leak * c.T_SPEC * c.arrhenius(c.T_TEST_DEFAULT)
    got = c.margin_at(m0, leak, c.T_TEST_DEFAULT)
    assert np.allclose(got, expected)


def test_margin_at_higher_vdd_increases_margin_by_gain():
    nom = c.margin_at(0.0, 0.0, c.T_TEST_DEFAULT, Vdd=c.VDD_NOM)
    high = c.margin_at(0.0, 0.0, c.T_TEST_DEFAULT, Vdd=c.VDD_NOM + 0.05)
    low = c.margin_at(0.0, 0.0, c.T_TEST_DEFAULT, Vdd=c.VDD_NOM - 0.05)
    assert low < nom < high
    assert np.allclose(high - nom, c.GAMMA_VDD * 0.05)


def test_margin_at_retention_drops_with_temperature():
    # a leaky cell loses more margin at the hot field corner than at test corner
    m_test = c.margin_at(0.20, 0.006, c.T_TEST_DEFAULT)
    m_field = c.margin_at(0.20, 0.006, c.T_FIELD)
    assert m_field < m_test  # hotter => more Arrhenius-accelerated leak


# ---- HBM KGD stacking yield (E4) ----
def test_stack_good_prob_is_per_die_product():
    assert np.isclose(kgd.stack_good_prob(0.01, 12), 0.99 ** 12)


def test_stack_good_prob_reference_12high_kgd():
    # canonical KGD reference: 0.99 per-die good, 12-high -> ~0.886
    assert abs(kgd.stack_good_prob(0.01, 12) - 0.8864) < 1e-3


def test_stack_good_prob_monotonic_in_height_and_escape():
    assert kgd.stack_good_prob(0.05, 4) > kgd.stack_good_prob(0.05, 12)
    assert kgd.stack_good_prob(0.02, 12) > kgd.stack_good_prob(0.06, 12)


def test_cube_escape_prob_is_complement_of_good():
    assert np.isclose(kgd.cube_escape_prob(0.05, 12), 1 - 0.95 ** 12)


def test_zero_escape_gives_perfect_stack():
    assert kgd.stack_good_prob(0.0, 16) == 1.0


def test_rule_of_three_upper_bound_for_zero_observed():
    # 0 escapes in n dies -> 95% upper bound ~ 3/n (honest non-zero ceiling)
    assert np.isclose(kgd.rule_of_three_upper(1500), 3.0 / 1500)
