"""TDD tests for process-capability helpers: Cp/Cpk and Gage R&R (ANOVA)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import process_capability as pc  # noqa: E402


def test_cp_cpk_unit_capability_when_limits_are_three_sigma():
    vals = np.array([8.0, 9.0, 10.0, 11.0, 12.0])
    sigma = vals.std(ddof=1)
    lsl, usl = vals.mean() - 3 * sigma, vals.mean() + 3 * sigma
    cp, cpk = pc.cp_cpk(vals, lsl, usl)
    assert np.isclose(cp, 1.0, atol=1e-9)
    assert np.isclose(cpk, 1.0, atol=1e-9)


def test_cpk_below_cp_when_process_off_center():
    vals = np.array([8.0, 9.0, 10.0, 11.0, 12.0])
    sigma = vals.std(ddof=1)
    # shift the spec window so the mean is off-center -> Cpk < Cp
    lsl, usl = vals.mean() - 2 * sigma, vals.mean() + 4 * sigma
    cp, cpk = pc.cp_cpk(vals, lsl, usl)
    assert cpk < cp


def test_gage_rr_zero_measurement_noise_gives_zero_grr():
    # value depends only on the part (no operator effect, no repeat noise) -> %GRR == 0
    rows = []
    for part in range(4):
        for op in range(3):
            for trial in range(2):
                rows.append({"part": part, "operator": op, "value": 10.0 * part})
    out = pc.gage_rr_anova(rows)
    assert np.isclose(out["pct_grr"], 0.0, atol=1e-6)


def test_gage_rr_components_nonneg_and_bounded():
    rng = np.random.default_rng(0)
    rows = []
    for part in range(8):
        part_mean = rng.normal(50, 5)            # real part-to-part spread
        for op in range(3):
            bias = rng.normal(0, 0.5)            # operator reproducibility
            for trial in range(3):
                rows.append({"part": part, "operator": op,
                             "value": part_mean + bias + rng.normal(0, 0.4)})
    out = pc.gage_rr_anova(rows)
    assert out["ev"] >= 0 and out["av"] >= 0 and out["grr"] >= 0 and out["pv"] >= 0
    assert 0.0 <= out["pct_grr"] <= 100.0
    assert out["ndc"] >= 0
