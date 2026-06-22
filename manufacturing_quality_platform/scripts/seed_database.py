from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.db import DEFAULT_DB_PATH  # noqa: E402
from app.seed import DEFAULT_SOURCE_ROOT, seed_database  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the manufacturing quality SQLite database.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--no-reset", action="store_true", help="Keep the existing database file.")
    args = parser.parse_args()

    summary = seed_database(
        db_path=args.db_path,
        source_root=args.source_root,
        reset=not args.no_reset,
    )
    print(
        "Seeded manufacturing quality database: "
        f"{summary['samples']} samples, "
        f"{summary['sensor_readings']} sensor readings, "
        f"{summary['predictions']} predictions, "
        f"{summary['alerts']} alerts, "
        f"{summary['operation_events']} synthetic operation events, "
        f"{summary['digital_thread_rows']} synthetic digital-thread rows"
    )


if __name__ == "__main__":
    main()
