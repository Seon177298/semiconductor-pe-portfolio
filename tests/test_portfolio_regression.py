"""Portfolio numeric regression guard.

Locks the headline numbers against silent drift by reading the committed report
artifacts (not by retraining). Deterministic, seed-fixed values are asserted
exactly; stochastic CNN metrics are guarded by range only.

Run: `python -m pytest tests/test_portfolio_regression.py -q` from the repo root.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _read(rel):
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"committed report missing: {rel}")
    return pd.read_csv(path)


# ---- fail_bit_map_pe (seed-fixed, byte-identical reproduction) ----
def test_fbm_guardband_headline():
    df = _read("fail_bit_map_pe/reports/fbm_v2_gb_sweep.csv")
    g0 = df[df["guardband"] == 0.0].iloc[0]
    opt = df[df["guardband"] == 0.05].iloc[0]
    assert abs(g0["escape_dppm"] - 550.13) < 1.0 and int(g0["escape_dies"]) == 86
    assert abs(opt["escape_dppm"] - 89.64) < 1.0 and int(opt["escape_dies"]) == 0
    reduction = (g0["total_cost"] - opt["total_cost"]) / g0["total_cost"]
    assert 0.79 < reduction < 0.81          # the headline "-80%" (seed 7)


def test_fbm_classifier_headline():
    df = _read("fail_bit_map_pe/reports/classifier_comparison.csv").set_index("model")
    assert df.loc["random_forest", "test_accuracy"] == 1.0
    assert abs(df.loc["rule_based", "test_accuracy"] - 0.982) < 0.01
    assert abs(df.loc["cnn", "test_accuracy"] - 0.998) < 0.01


def test_fbm_kgd_12high_cube_escape():
    df = _read("fail_bit_map_pe/reports/kgd_stacking.csv").set_index("stack_height")
    assert abs(df.loc[12, "unscreened_cube_escape"] - 0.5948) < 0.01


# ---- secom_quality_8d (single 70/30 split, seed 42, deterministic) ----
def test_secom_single_split_headline():
    df = _read("secom_quality_8d/reports/threshold_tradeoff.csv")
    row = df[(df["strategy"] == "median_all") & (df["model"] == "random_forest")
             & (np.isclose(df["threshold"], 0.10))].iloc[0]
    assert abs(row["recall_defect"] - 0.774) < 0.005
    assert abs(row["false_alarm_rate"] - 0.248) < 0.005
    assert int(row["missed_defect_count"]) == 7


def test_secom_nested_cv_operating_point_is_leakage_free_lower():
    # the no-leakage nested-CV estimate at threshold 0.10 should sit below the
    # optimistic single-split recall 0.774 (the no-leakage nested-CV result)
    df = _read("secom_quality_8d/reports/rigor_nested_cv.csv").set_index("model")
    rf_recall = df.loc["random_forest", "recall_at_0p10_mean"]
    assert rf_recall < 0.774
