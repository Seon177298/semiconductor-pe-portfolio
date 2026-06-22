from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.db import DEFAULT_DB_PATH, connect  # noqa: E402
from app.main import create_app  # noqa: E402
from app.seed import seed_database  # noqa: E402


def ensure_seeded() -> None:
    if not DEFAULT_DB_PATH.exists():
        seed_database()
        return
    with connect(DEFAULT_DB_PATH) as conn:
        try:
            prediction_count = conn.execute("SELECT COUNT(*) FROM quality_predictions").fetchone()[0]
            digital_thread_count = conn.execute("SELECT COUNT(*) FROM quality_gates").fetchone()[0]
        except Exception:
            prediction_count = 0
            digital_thread_count = 0
    if prediction_count == 0 or digital_thread_count == 0:
        seed_database()


def main() -> None:
    ensure_seeded()
    client = TestClient(create_app(DEFAULT_DB_PATH))
    checks = [
        ("GET /health", client.get("/health")),
        ("GET /quality/metrics", client.get("/quality/metrics", params={"threshold": 0.50})),
        (
            "GET /quality/failure-cases",
            client.get("/quality/failure-cases", params={"threshold": 0.50, "case_type": "false_alarm", "limit": 3}),
        ),
        ("GET /digital-thread/source-map", client.get("/digital-thread/source-map")),
        ("GET /digital-thread/trace", client.get("/digital-thread/trace", params={"sample_id": 1})),
        ("GET /vision/status", client.get("/vision/status")),
    ]
    failed = False
    for label, response in checks:
        ok = response.status_code == 200
        print(f"{label}: {response.status_code}")
        if not ok:
            failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
