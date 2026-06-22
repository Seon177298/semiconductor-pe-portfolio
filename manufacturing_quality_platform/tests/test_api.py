from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.db import connect
from app.main import create_app
from app.seed import seed_database


def build_client(tmp_path: Path) -> TestClient:
    db_path = tmp_path / "manufacturing_quality.db"
    seed_database(db_path=db_path, source_root=Path(__file__).resolve().parents[2] / "secom_quality_8d")
    return TestClient(create_app(db_path=db_path))


def test_seeded_api_exposes_health_lines_readings_predictions_and_alerts(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    assert client.get("/health").json() == {"status": "ok"}

    lines = client.get("/lines").json()
    assert lines[0]["line_code"] == "LINE-SECOM-01"
    assert lines[0]["equipment_count"] >= 1

    readings = client.get("/equipment/EQP-SECOM-001/readings", params={"limit": 5}).json()
    assert len(readings) == 5
    assert readings[0]["sensor_name"].startswith("sensor_")

    predictions = client.get("/quality/predictions", params={"limit": 10}).json()
    assert len(predictions) == 10
    assert {"sample_id", "equipment_id", "defect_score", "true_defect"} <= predictions[0].keys()

    alerts = client.get("/alerts").json()
    assert len(alerts) >= 20
    assert alerts[0]["alert_type"] == "quality_alert"


def test_threshold_recalculation_changes_recall_and_false_alarm_metrics(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    low = client.post("/quality/threshold", json={"threshold": 0.10}).json()
    high = client.post("/quality/threshold", json={"threshold": 0.50}).json()

    assert low["threshold"] == 0.10
    assert high["threshold"] == 0.50
    assert low["policy_id"] != high["policy_id"]
    assert low["recall_defect"] != high["recall_defect"]
    assert low["false_alarm_count"] != high["false_alarm_count"]


def test_threshold_policy_preserves_alert_history_and_filters_active_policy(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    first = client.post("/quality/threshold", json={"threshold": 0.10}).json()
    first_alerts = client.get("/alerts", params={"policy_id": first["policy_id"], "limit": 2000}).json()
    second = client.post("/quality/threshold", json={"threshold": 0.50}).json()
    active_alerts = client.get("/alerts", params={"limit": 2000}).json()
    first_alerts_after = client.get("/alerts", params={"policy_id": first["policy_id"], "limit": 2000}).json()
    policies = client.get("/quality/policies").json()

    assert first["alert_count"] == len(first_alerts)
    assert first_alerts_after == first_alerts
    assert second["alert_count"] == len(active_alerts)
    assert {row["policy_id"] for row in active_alerts} == {second["policy_id"]}
    assert len(policies) >= 3
    assert policies[0]["is_active"] is True
    assert policies[0]["policy_id"] == second["policy_id"]
    assert any(row["policy_id"] == first["policy_id"] and row["is_active"] is False for row in policies)


def test_quality_metrics_previews_threshold_without_mutating_policy_or_alerts(tmp_path: Path) -> None:
    db_path = tmp_path / "manufacturing_quality.db"
    seed_database(db_path=db_path, source_root=Path(__file__).resolve().parents[2] / "secom_quality_8d")
    client = TestClient(create_app(db_path=db_path))

    with connect(db_path) as conn:
        policy_count_before = conn.execute("SELECT COUNT(*) FROM threshold_policies").fetchone()[0]
        alert_count_before = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]

    low = client.get("/quality/metrics", params={"threshold": 0.10}).json()
    high = client.get("/quality/metrics", params={"threshold": 0.50}).json()

    assert low["threshold"] == 0.10
    assert high["threshold"] == 0.50
    assert low["recall_defect"] != high["recall_defect"]
    assert low["false_alarm_count"] != high["false_alarm_count"]
    assert "alert_count" not in low

    with connect(db_path) as conn:
        policy_count_after = conn.execute("SELECT COUNT(*) FROM threshold_policies").fetchone()[0]
        alert_count_after = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]

    assert policy_count_after == policy_count_before
    assert alert_count_after == alert_count_before


def test_failure_cases_reviews_and_operations_context(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    false_alarms = client.get(
        "/quality/failure-cases",
        params={"threshold": 0.50, "case_type": "false_alarm", "limit": 5},
    ).json()
    missed_defects = client.get(
        "/quality/failure-cases",
        params={"threshold": 0.50, "case_type": "missed_defect", "limit": 5},
    ).json()
    events = client.get("/operations/events", params={"limit": 5}).json()

    assert false_alarms
    assert missed_defects
    assert false_alarms[0]["case_type"] == "false_alarm"
    assert missed_defects[0]["case_type"] == "missed_defect"
    assert events[0]["context_type"] in {"production_order", "process_step", "equipment_event"}
    assert events[0]["is_synthetic"] is True

    review = client.post(
        "/quality/reviews",
        json={
            "sample_id": false_alarms[0]["sample_id"],
            "case_type": "false_alarm",
            "threshold": 0.50,
            "review_note": "Check whether missing sensors inflated the score.",
            "root_cause_tag": "needs_real_equipment_context",
            "next_data_needed": "tool maintenance and lot route history",
        },
    ).json()
    reviewed = client.get(
        "/quality/failure-cases",
        params={"threshold": 0.50, "case_type": "false_alarm", "limit": 5},
    ).json()

    assert review["review_id"] > 0
    assert any(row["latest_review_note"] == "Check whether missing sensors inflated the score." for row in reviewed)


def test_digital_thread_source_map_lots_and_trace(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    source_map = client.get("/digital-thread/source-map").json()
    assert source_map["boundary_note"] == "Public case-informed synthetic layer, not internal company internal data."
    assert len(source_map["sources"]) == 2
    assert {"source_name", "public_url", "used_for", "boundary_note"} <= source_map["sources"][0].keys()

    lots = client.get("/digital-thread/lots").json()
    assert lots
    assert {"ebom_count", "mbom_count", "pbom_count", "bop_count"} <= lots[0].keys()
    assert lots[0]["is_synthetic"] is True
    assert lots[0]["ebom_count"] > 0
    assert lots[0]["mbom_count"] > 0
    assert lots[0]["pbom_count"] > 0
    assert lots[0]["bop_count"] > 0

    lot_detail = client.get(f"/digital-thread/lot/{lots[0]['lot_id']}").json()
    assert lot_detail["lot"]["lot_id"] == lots[0]["lot_id"]
    assert {row["bom_type"] for row in lot_detail["bom_items"]} == {"EBOM", "MBOM", "PBOM"}
    assert lot_detail["bop_steps"]
    assert lot_detail["quality_gates"]

    trace = client.get("/digital-thread/trace", params={"sample_id": 1}).json()
    assert trace["prediction"]["sample_id"] == 1
    assert trace["lot"]["lot_id"]
    assert {row["bom_type"] for row in trace["bom_items"]} == {"EBOM", "MBOM", "PBOM"}
    assert trace["bop_steps"]
    assert trace["quality_gate"]["sample_id"] == 1
    assert "active_alert" in trace
    assert "latest_review" in trace
    assert "eight_d_report_candidate" in trace


def test_digital_thread_seed_invariants_and_missing_sample(tmp_path: Path) -> None:
    db_path = tmp_path / "manufacturing_quality.db"
    seed_database(db_path=db_path, source_root=Path(__file__).resolve().parents[2] / "secom_quality_8d")
    client = TestClient(create_app(db_path=db_path))

    assert client.get("/digital-thread/trace", params={"sample_id": 999999}).status_code == 404

    with connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM quality_predictions").fetchone()[0] == 1567
        assert conn.execute("SELECT COUNT(*) FROM process_lots").fetchone()[0] >= 4
        assert conn.execute("SELECT COUNT(*) FROM bom_items").fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM bop_steps").fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM quality_gates").fetchone()[0] == 1567

        for table_name in ("process_lots", "bom_items", "bop_steps", "quality_gates"):
            non_synthetic = conn.execute(
                f"SELECT COUNT(*) FROM {table_name} WHERE is_synthetic != 1"
            ).fetchone()[0]
            assert non_synthetic == 0


def test_ingest_sensor_batch_and_create_8d_report(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    ingest = client.post(
        "/ingest/sensor-batch",
        json={
            "equipment_id": "EQP-SECOM-001",
            "readings": [
                {"sample_id": 900001, "sensor_name": "sensor_000", "sensor_value": 1.23},
                {"sample_id": 900001, "sensor_name": "sensor_001", "sensor_value": None},
            ],
        },
    ).json()
    assert ingest == {"inserted": 2}

    alert_id = client.get("/alerts").json()[0]["alert_id"]
    created = client.post(
        "/reports/8d",
        json={
            "alert_id": alert_id,
            "title": "High defect score follow-up",
            "owner": "portfolio-demo",
        },
    ).json()
    assert created["report_id"] > 0
    report = client.get(f"/reports/8d/{created['report_id']}").json()
    assert report["alert_id"] == alert_id
    assert "public SECOM" in report["containment_action"]


def test_all_synthetic_layer_tables_are_flagged_synthetic(tmp_path: Path) -> None:
    """Boundary guard over the FULL synthetic layer (7 tables).

    test_digital_thread_seed_invariants_and_missing_sample covers four tables;
    this extends the same is_synthetic=1 invariant to production_orders,
    process_steps, and equipment_events so the public-data boundary claim is
    verifiable across every synthetic table, not just the BOM/BOP ones.
    """
    db_path = tmp_path / "manufacturing_quality.db"
    seed_database(db_path=db_path, source_root=Path(__file__).resolve().parents[2] / "secom_quality_8d")

    synthetic_tables = (
        "production_orders",
        "process_steps",
        "equipment_events",
        "process_lots",
        "bom_items",
        "bop_steps",
        "quality_gates",
    )
    with connect(db_path) as conn:
        for table_name in synthetic_tables:
            total = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            non_synthetic = conn.execute(
                f"SELECT COUNT(*) FROM {table_name} WHERE is_synthetic != 1"
            ).fetchone()[0]
            assert total > 0, f"{table_name} should be seeded with synthetic rows"
            assert non_synthetic == 0, f"{table_name} has {non_synthetic} rows not flagged synthetic"


def test_8d_report_includes_review_note_when_available(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    false_alarm = client.get(
        "/quality/failure-cases",
        params={"threshold": 0.50, "case_type": "false_alarm", "limit": 1},
    ).json()[0]
    client.post(
        "/quality/reviews",
        json={
            "sample_id": false_alarm["sample_id"],
            "case_type": "false_alarm",
            "threshold": 0.50,
            "review_note": "Operator review should compare neighboring process context.",
            "root_cause_tag": "follow_up_only",
            "next_data_needed": "process step, lot, and maintenance logs",
        },
    )
    alert = client.get("/alerts", params={"limit": 2000}).json()
    alert_id = next(row["alert_id"] for row in alert if row["sample_id"] == false_alarm["sample_id"])

    created = client.post("/reports/8d", json={"alert_id": alert_id}).json()
    report = client.get(f"/reports/8d/{created['report_id']}").json()

    assert "Operator review should compare neighboring process context." in report["report_markdown"]
    assert "process step, lot, and maintenance logs" in report["report_markdown"]


def test_quality_yield_summary_reports_yield_bins_and_escape_overkill(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    data = client.get("/quality/yield-summary", params={"threshold": 0.5}).json()
    assert 0.0 <= data["yield"] <= 1.0
    assert data["shipped"] + data["rejected"] == data["total"]
    assert data["escape_count"] >= 0 and data["overkill_count"] >= 0
    assert sum(b["count"] for b in data["bins"]) == data["total"]
    # higher defect-score bands should carry a higher (or equal) true-defect rate
    assert data["bins"][-1]["defect_rate"] >= data["bins"][0]["defect_rate"]
