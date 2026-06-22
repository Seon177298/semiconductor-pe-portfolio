from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "manufacturing_quality.db"


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS production_lines (
            line_id INTEGER PRIMARY KEY AUTOINCREMENT,
            line_code TEXT NOT NULL UNIQUE,
            line_name TEXT NOT NULL,
            description TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS equipment (
            equipment_id TEXT PRIMARY KEY,
            line_id INTEGER NOT NULL REFERENCES production_lines(line_id),
            equipment_name TEXT NOT NULL,
            equipment_role TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sensor_readings (
            reading_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER NOT NULL,
            equipment_id TEXT NOT NULL REFERENCES equipment(equipment_id),
            sensor_name TEXT NOT NULL,
            sensor_value REAL,
            reading_ts TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_sensor_readings_equipment
            ON sensor_readings(equipment_id, sample_id);

        CREATE TABLE IF NOT EXISTS quality_predictions (
            sample_id INTEGER PRIMARY KEY,
            equipment_id TEXT NOT NULL REFERENCES equipment(equipment_id),
            model_name TEXT NOT NULL,
            defect_score REAL NOT NULL,
            true_defect INTEGER NOT NULL CHECK (true_defect IN (0, 1))
        );
        CREATE INDEX IF NOT EXISTS idx_quality_predictions_score
            ON quality_predictions(defect_score DESC);

        CREATE TABLE IF NOT EXISTS threshold_policies (
            policy_id INTEGER PRIMARY KEY AUTOINCREMENT,
            threshold REAL NOT NULL,
            precision_defect REAL NOT NULL,
            recall_defect REAL NOT NULL,
            f1_defect REAL NOT NULL,
            false_alarm_rate REAL NOT NULL,
            false_alarm_count INTEGER NOT NULL,
            missed_defect_count INTEGER NOT NULL,
            tp INTEGER NOT NULL,
            tn INTEGER NOT NULL,
            fp INTEGER NOT NULL,
            fn INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS alerts (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            policy_id INTEGER REFERENCES threshold_policies(policy_id),
            sample_id INTEGER NOT NULL,
            equipment_id TEXT NOT NULL REFERENCES equipment(equipment_id),
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            defect_score REAL NOT NULL,
            threshold REAL NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_alerts_status_score
            ON alerts(status, defect_score DESC);
        CREATE INDEX IF NOT EXISTS idx_alerts_policy_status_score
            ON alerts(policy_id, status, defect_score DESC);

        CREATE TABLE IF NOT EXISTS quality_reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER NOT NULL,
            case_type TEXT NOT NULL CHECK (case_type IN ('false_alarm', 'missed_defect')),
            threshold REAL NOT NULL,
            review_note TEXT NOT NULL,
            root_cause_tag TEXT NOT NULL,
            next_data_needed TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_quality_reviews_sample_case
            ON quality_reviews(sample_id, case_type, created_at DESC);

        CREATE TABLE IF NOT EXISTS production_orders (
            order_id TEXT PRIMARY KEY,
            sample_id INTEGER NOT NULL,
            product_family TEXT NOT NULL,
            planned_quantity INTEGER NOT NULL,
            synthetic_note TEXT NOT NULL,
            is_synthetic INTEGER NOT NULL DEFAULT 1 CHECK (is_synthetic IN (0, 1))
        );
        CREATE INDEX IF NOT EXISTS idx_production_orders_sample
            ON production_orders(sample_id);

        CREATE TABLE IF NOT EXISTS process_steps (
            step_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER NOT NULL,
            order_id TEXT NOT NULL REFERENCES production_orders(order_id),
            step_name TEXT NOT NULL,
            step_sequence INTEGER NOT NULL,
            synthetic_note TEXT NOT NULL,
            is_synthetic INTEGER NOT NULL DEFAULT 1 CHECK (is_synthetic IN (0, 1))
        );
        CREATE INDEX IF NOT EXISTS idx_process_steps_sample
            ON process_steps(sample_id);

        CREATE TABLE IF NOT EXISTS equipment_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER NOT NULL,
            equipment_id TEXT NOT NULL REFERENCES equipment(equipment_id),
            event_type TEXT NOT NULL,
            event_detail TEXT NOT NULL,
            event_ts TEXT NOT NULL,
            synthetic_note TEXT NOT NULL,
            is_synthetic INTEGER NOT NULL DEFAULT 1 CHECK (is_synthetic IN (0, 1))
        );
        CREATE INDEX IF NOT EXISTS idx_equipment_events_sample
            ON equipment_events(sample_id);

        CREATE TABLE IF NOT EXISTS process_lots (
            lot_id TEXT PRIMARY KEY,
            device_family TEXT NOT NULL,
            lot_type TEXT NOT NULL,
            source_note TEXT NOT NULL,
            is_synthetic INTEGER NOT NULL DEFAULT 1 CHECK (is_synthetic IN (0, 1))
        );

        CREATE TABLE IF NOT EXISTS bom_items (
            bom_item_id TEXT PRIMARY KEY,
            bom_type TEXT NOT NULL CHECK (bom_type IN ('EBOM', 'MBOM', 'PBOM')),
            lot_id TEXT NOT NULL REFERENCES process_lots(lot_id),
            part_code TEXT NOT NULL,
            part_name TEXT NOT NULL,
            revision TEXT NOT NULL,
            quantity REAL NOT NULL,
            parent_part_code TEXT,
            source_note TEXT NOT NULL,
            is_synthetic INTEGER NOT NULL DEFAULT 1 CHECK (is_synthetic IN (0, 1))
        );
        CREATE INDEX IF NOT EXISTS idx_bom_items_lot_type
            ON bom_items(lot_id, bom_type);

        CREATE TABLE IF NOT EXISTS bop_steps (
            bop_step_id TEXT PRIMARY KEY,
            lot_id TEXT NOT NULL REFERENCES process_lots(lot_id),
            step_sequence INTEGER NOT NULL,
            operation_name TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            expected_minutes INTEGER NOT NULL,
            source_note TEXT NOT NULL,
            is_synthetic INTEGER NOT NULL DEFAULT 1 CHECK (is_synthetic IN (0, 1))
        );
        CREATE INDEX IF NOT EXISTS idx_bop_steps_block_sequence
            ON bop_steps(lot_id, step_sequence);

        CREATE TABLE IF NOT EXISTS quality_gates (
            gate_id TEXT PRIMARY KEY,
            sample_id INTEGER NOT NULL,
            lot_id TEXT NOT NULL REFERENCES process_lots(lot_id),
            bop_step_id TEXT NOT NULL REFERENCES bop_steps(bop_step_id),
            gate_name TEXT NOT NULL,
            check_type TEXT NOT NULL,
            quality_signal TEXT NOT NULL,
            source_note TEXT NOT NULL,
            is_synthetic INTEGER NOT NULL DEFAULT 1 CHECK (is_synthetic IN (0, 1))
        );
        CREATE INDEX IF NOT EXISTS idx_quality_gates_sample
            ON quality_gates(sample_id);
        CREATE INDEX IF NOT EXISTS idx_quality_gates_block
            ON quality_gates(lot_id);

        CREATE TABLE IF NOT EXISTS eight_d_reports (
            report_id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id INTEGER NOT NULL REFERENCES alerts(alert_id),
            title TEXT NOT NULL,
            owner TEXT NOT NULL,
            problem_statement TEXT NOT NULL,
            containment_action TEXT NOT NULL,
            root_cause_hypothesis TEXT NOT NULL,
            corrective_action TEXT NOT NULL,
            verification_plan TEXT NOT NULL,
            lessons_learned TEXT NOT NULL,
            report_markdown TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    _ensure_alert_policy_column(conn)


def _ensure_alert_policy_column(conn: sqlite3.Connection) -> None:
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(alerts)").fetchall()
    }
    if "policy_id" not in columns:
        conn.execute("ALTER TABLE alerts ADD COLUMN policy_id INTEGER REFERENCES threshold_policies(policy_id)")


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]
