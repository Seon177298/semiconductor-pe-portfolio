"""TDD tests for E5 test-time / throughput model (pure formulas)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import e5_throughput as e5  # noqa: E402


def test_base_only_time_is_n_times_base():
    assert e5.total_test_time_s(10, repair=0, remeasure=0) == 10 * e5.T_BASE


def test_time_increases_with_repair_insertions():
    base = e5.total_test_time_s(1000, repair=0, remeasure=0)
    more = e5.total_test_time_s(1000, repair=500, remeasure=0)
    assert more > base


def test_time_increases_with_remeasurement():
    a = e5.total_test_time_s(1000, repair=100, remeasure=0)
    b = e5.total_test_time_s(1000, repair=100, remeasure=200)
    assert b > a


def test_throughput_is_count_per_hour():
    assert e5.throughput_dph(100, 3600.0) == 100.0
