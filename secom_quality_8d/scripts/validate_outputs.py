"""Validate that run_analysis.py produced the expected core portfolio outputs.

Lightweight contract check (no test framework needed):
  python scripts/validate_outputs.py
Exits non-zero with a clear message if any core artifact is missing or empty,
so the portfolio's "runnable + reproducible" claim stays verifiable.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

REQUIRED = [
    REPORTS / "data_profile.csv",
    REPORTS / "metrics.csv",
    REPORTS / "threshold_tradeoff.csv",
    REPORTS / "sample_scores.csv",
    REPORTS / "top_features.csv",
    REPORTS / "error_cases.csv",
    REPORTS / "8d_report.md",
    REPORTS / "figures" / "threshold_tradeoff.png",
]


def main() -> int:
    missing = [p for p in REQUIRED if not p.exists()]
    empty = [p for p in REQUIRED if p.exists() and p.stat().st_size == 0]
    if missing:
        print("MISSING outputs (run `python scripts/run_analysis.py` first):")
        for p in missing:
            print(f"  - {p.relative_to(ROOT)}")
    if empty:
        print("EMPTY outputs:")
        for p in empty:
            print(f"  - {p.relative_to(ROOT)}")
    if missing or empty:
        return 1
    print(f"OK: {len(REQUIRED)} core outputs present and non-empty.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
