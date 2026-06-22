"""TDD tests for E3 shmoo / operating-window shape signatures.

The point of a shmoo is that the SHAPE of the pass/fail window diagnoses the
failure mechanism. We assert the three hypotheses produce distinct signatures:
  - retention-limited : window shrinks as temperature rises (Arrhenius leak)
  - Vdd-limited       : window is temperature-invariant (no leak; a Vdd wall)
  - hard structural   : no operating window at any corner
Run: `python -m pytest tests/ -q` from fail_bit_map_pe/.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shmoo  # noqa: E402


def test_retention_window_shrinks_with_temperature():
    rng = np.random.default_rng(5)
    m0, leak = shmoo.make_hypothesis_die("retention", rng)
    cold = shmoo.window_area_fraction(m0, leak, 25)
    hot = shmoo.window_area_fraction(m0, leak, 85)
    assert hot < cold


def test_vdd_limited_window_is_temperature_invariant():
    rng = np.random.default_rng(5)
    m0, leak = shmoo.make_hypothesis_die("vdd", rng)
    a25 = shmoo.window_area_fraction(m0, leak, 25)
    a85 = shmoo.window_area_fraction(m0, leak, 85)
    assert abs(a25 - a85) < 0.05  # no leak => essentially T-independent


def test_vdd_limited_window_opens_with_higher_vdd():
    # at fixed timing/temperature, raising Vdd should recover the part
    rng = np.random.default_rng(5)
    m0, leak = shmoo.make_hypothesis_die("vdd", rng)
    low = shmoo.die_passes(m0, leak, temp_C=80, timing=1.0, vdd=1.00)
    high = shmoo.die_passes(m0, leak, temp_C=80, timing=1.0, vdd=1.20)
    assert (not low) and high


def test_hard_defect_has_no_operating_window():
    rng = np.random.default_rng(5)
    m0, leak = shmoo.make_hypothesis_die("hard", rng)
    for T in (25, 80, 85):
        assert shmoo.window_area_fraction(m0, leak, T) == 0.0
