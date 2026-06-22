from __future__ import annotations

import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .db import DEFAULT_DB_PATH, connect, initialize_schema


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
LABEL_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"
DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "secom_quality_8d"
SELECTED_SENSOR_COUNT = 40
DEFAULT_THRESHOLD = 0.50
DIGITAL_THREAD_SOURCE_NOTE = (
    "Actual internal company data was not used. This is a public-case-informed synthetic "
    "process-traceability layer over UCI SECOM samples, modeled on a generic semiconductor "
    "design -> mask/process -> wafer-test flow (EBOM/MBOM/PBOM/BOP). Secondary to the PE "
    "yield / bin / escape-overkill dashboard."
)


def download_if_missing(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)


def resolve_raw_paths(source_root: Path) -> tuple[Path, Path]:
    source_data = source_root / "data" / "raw" / "secom.data"
    source_labels = source_root / "data" / "raw" / "secom_labels.data"
    local_raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    data_path = local_raw / "secom.data"
    label_path = local_raw / "secom_labels.data"

    if source_data.exists() and source_labels.exists():
        return source_data, source_labels

    download_if_missing(DATA_URL, data_path)
    download_if_missing(LABEL_URL, label_path)
    return data_path, label_path


def load_secom(source_root: Path = DEFAULT_SOURCE_ROOT) -> tuple[pd.DataFrame, pd.Series]:
    data_path, label_path = resolve_raw_paths(source_root)
    x = pd.read_csv(data_path, sep=r"\s+", header=None, na_values="NaN")
    labels = pd.read_csv(label_path, sep=r"\s+", header=None, names=["label", "timestamp"])
    x.columns = [f"sensor_{idx:03d}" for idx in range(x.shape[1])]
    y = (labels["label"] == 1).astype(int)
    return x, y


def train_scores(x: pd.DataFrame, y: pd.Series) -> list[float]:
    selected = x.iloc[:, :SELECTED_SENSOR_COUNT]
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=500,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(selected, y)
    return [float(score) for score in model.predict_proba(selected)[:, 1]]


def calculate_metrics_from_rows(rows: list[dict[str, float | int]], threshold: float) -> dict[str, float | int]:
    tp = tn = fp = fn = 0
    for row in rows:
        actual = int(row["true_defect"])
        predicted = float(row["defect_score"]) >= threshold
        if actual == 1 and predicted:
            tp += 1
        elif actual == 0 and not predicted:
            tn += 1
        elif actual == 0 and predicted:
            fp += 1
        elif actual == 1 and not predicted:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    false_alarm_rate = fp / (fp + tn) if fp + tn else 0.0
    return {
        "threshold": threshold,
        "precision_defect": precision,
        "recall_defect": recall,
        "f1_defect": f1,
        "false_alarm_rate": false_alarm_rate,
        "false_alarm_count": fp,
        "missed_defect_count": fn,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def refresh_alerts(conn, threshold: float, policy_id: int) -> int:
    predictions = conn.execute(
        """
        SELECT sample_id, equipment_id, defect_score
        FROM quality_predictions
        WHERE defect_score >= ?
        ORDER BY defect_score DESC
        """,
        (threshold,),
    ).fetchall()
    conn.executemany(
        """
        INSERT INTO alerts (
            policy_id, sample_id, equipment_id, alert_type, severity, defect_score, threshold, status
        )
        VALUES (?, ?, ?, 'quality_alert', ?, ?, ?, 'open')
        """,
        [
            (
                policy_id,
                int(row["sample_id"]),
                row["equipment_id"],
                "high" if float(row["defect_score"]) >= 0.75 else "medium",
                float(row["defect_score"]),
                threshold,
            )
            for row in predictions
        ],
    )
    return len(predictions)


def seed_synthetic_operations_context(conn) -> int:
    predictions = conn.execute(
        """
        SELECT sample_id, equipment_id, defect_score, true_defect
        FROM quality_predictions
        ORDER BY sample_id
        """
    ).fetchall()
    order_rows = []
    step_rows = []
    event_rows = []
    start_ts = datetime(2026, 5, 11, 9, 0, 0)

    for idx, row in enumerate(predictions, start=1):
        sample_id = int(row["sample_id"])
        equipment_id = row["equipment_id"]
        order_id = f"PO-SECOM-{((sample_id - 1) // 50) + 1:04d}"
        product_family = "public-secom-sensor-lot"
        synthetic_note = "Synthetic context for portfolio workflow design; not internal company or field data."
        order_rows.append(
            (
                order_id,
                sample_id,
                product_family,
                50,
                synthetic_note,
                1,
            )
        )
        step_rows.append(
            (
                sample_id,
                order_id,
                f"anonymous_step_{(sample_id % 4) + 1}",
                (sample_id % 4) + 1,
                synthetic_note,
                1,
            )
        )
        event_type = "defect_score_review" if float(row["defect_score"]) >= DEFAULT_THRESHOLD else "routine_sensor_capture"
        event_rows.append(
            (
                sample_id,
                equipment_id,
                event_type,
                f"sample={sample_id}; true_defect={int(row['true_defect'])}; score={float(row['defect_score']):.3f}",
                (start_ts + timedelta(minutes=idx)).isoformat(),
                synthetic_note,
                1,
            )
        )

    conn.executemany(
        """
        INSERT OR IGNORE INTO production_orders (
            order_id, sample_id, product_family, planned_quantity, synthetic_note, is_synthetic
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        order_rows,
    )
    conn.executemany(
        """
        INSERT INTO process_steps (
            sample_id, order_id, step_name, step_sequence, synthetic_note, is_synthetic
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        step_rows,
    )
    conn.executemany(
        """
        INSERT INTO equipment_events (
            sample_id, equipment_id, event_type, event_detail, event_ts, synthetic_note, is_synthetic
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        event_rows,
    )
    return len(event_rows)


def seed_synthetic_digital_thread(conn) -> dict[str, int]:
    predictions = conn.execute(
        """
        SELECT sample_id, defect_score, true_defect
        FROM quality_predictions
        ORDER BY sample_id
        """
    ).fetchall()
    lots = [
        ("LOT-LOGIC-001", "SECOM-DT-DEVICE-A", "logic_wafer_lot", DIGITAL_THREAD_SOURCE_NOTE, 1),
        ("LOT-LOGIC-002", "SECOM-DT-DEVICE-A", "logic_wafer_lot", DIGITAL_THREAD_SOURCE_NOTE, 1),
        ("LOT-DRAM-003", "SECOM-DT-DEVICE-B", "dram_wafer_lot", DIGITAL_THREAD_SOURCE_NOTE, 1),
        ("LOT-DRAM-004", "SECOM-DT-DEVICE-B", "dram_wafer_lot", DIGITAL_THREAD_SOURCE_NOTE, 1),
        ("LOT-NAND-005", "SECOM-DT-DEVICE-C", "nand_wafer_lot", DIGITAL_THREAD_SOURCE_NOTE, 1),
        ("LOT-NAND-006", "SECOM-DT-DEVICE-C", "nand_wafer_lot", DIGITAL_THREAD_SOURCE_NOTE, 1),
    ]
    conn.executemany(
        """
        INSERT INTO process_lots (
            lot_id, device_family, lot_type, source_note, is_synthetic
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        lots,
    )

    # design -> mask/process -> assembly BOM (generic PLM layers, semiconductor-themed, synthetic)
    bom_templates = {
        "EBOM": [
            ("NETLIST", "Design netlist / IP block set", 1.0, None),
            ("STDCELL", "Standard-cell library package", 2.0, "NETLIST"),
            ("IO", "IO / PHY block package", 1.0, "NETLIST"),
        ],
        "MBOM": [
            ("MASKSET", "Mask set (reticles)", 1.0, None),
            ("RECIPE", "Process recipe / traveler kit", 6.0, "MASKSET"),
            ("PROC-CONS", "Process consumable package", 12.0, "MASKSET"),
        ],
        "PBOM": [
            ("SUBSTRATE", "Package substrate package", 1.0, None),
            ("TESTPROG", "Wafer/final test program kit", 1.0, "SUBSTRATE"),
        ],
    }
    bom_rows = []
    for lot_id, _, lot_type, _, _ in lots:
        lot_prefix = lot_id.replace("LOT-", "")
        for bom_type, templates in bom_templates.items():
            for idx, (part_suffix, part_name, quantity, parent_suffix) in enumerate(templates, start=1):
                part_code = f"{lot_prefix}-{bom_type}-{part_suffix}"
                parent_part_code = f"{lot_prefix}-{bom_type}-{parent_suffix}" if parent_suffix else None
                bom_rows.append(
                    (
                        f"BOM-{lot_prefix}-{bom_type}-{idx:02d}",
                        bom_type,
                        lot_id,
                        part_code,
                        f"{lot_type} {part_name}",
                        "R1",
                        quantity,
                        parent_part_code,
                        DIGITAL_THREAD_SOURCE_NOTE,
                        1,
                    )
                )
    conn.executemany(
        """
        INSERT INTO bom_items (
            bom_item_id, bom_type, lot_id, part_code, part_name, revision,
            quantity, parent_part_code, source_note, is_synthetic
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        bom_rows,
    )

    # process-of-record (BOP): a simplified wafer-fab + test flow
    bop_templates = [
        ("Photolithography patterning", "litho_cell", 60),
        ("Etch / deposition module", "etch_depo_chamber", 180),
        ("CMP planarization", "cmp_tool", 60),
        ("Wafer probe test gate", "probe_station", 90),
    ]
    bop_rows = []
    bop_ids_by_lot: dict[str, list[str]] = {}
    for lot_id, *_ in lots:
        lot_prefix = lot_id.replace("LOT-", "")
        bop_ids_by_lot[lot_id] = []
        for sequence, (operation_name, resource_type, expected_minutes) in enumerate(bop_templates, start=1):
            bop_step_id = f"BOP-{lot_prefix}-{sequence:02d}"
            bop_ids_by_lot[lot_id].append(bop_step_id)
            bop_rows.append(
                (
                    bop_step_id,
                    lot_id,
                    sequence,
                    operation_name,
                    resource_type,
                    expected_minutes,
                    DIGITAL_THREAD_SOURCE_NOTE,
                    1,
                )
            )
    conn.executemany(
        """
        INSERT INTO bop_steps (
            bop_step_id, lot_id, step_sequence, operation_name, resource_type,
            expected_minutes, source_note, is_synthetic
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        bop_rows,
    )

    gate_rows = []
    for row in predictions:
        sample_id = int(row["sample_id"])
        lot_id = lots[(sample_id - 1) % len(lots)][0]
        bop_step_id = bop_ids_by_lot[lot_id][(sample_id - 1) % len(bop_templates)]
        score = float(row["defect_score"])
        true_defect = int(row["true_defect"])
        if score >= DEFAULT_THRESHOLD:
            quality_signal = "model_defect_candidate"
        elif true_defect == 1:
            quality_signal = "missed_defect_review_candidate"
        else:
            quality_signal = "routine_pass_monitoring"
        gate_rows.append(
            (
                f"QG-SECOM-{sample_id:04d}",
                sample_id,
                lot_id,
                bop_step_id,
                "SECOM-linked synthetic wafer-test gate",
                "anonymous_sensor_model_review",
                quality_signal,
                DIGITAL_THREAD_SOURCE_NOTE,
                1,
            )
        )
    conn.executemany(
        """
        INSERT INTO quality_gates (
            gate_id, sample_id, lot_id, bop_step_id, gate_name, check_type,
            quality_signal, source_note, is_synthetic
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        gate_rows,
    )
    return {
        "lots": len(lots),
        "bom_items": len(bom_rows),
        "bop_steps": len(bop_rows),
        "quality_gates": len(gate_rows),
    }


def seed_database(
    db_path: Path | str = DEFAULT_DB_PATH,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    reset: bool = True,
) -> dict[str, int]:
    db_path = Path(db_path)
    if reset and db_path.exists():
        db_path.unlink()

    x, y = load_secom(source_root)
    scores = train_scores(x, y)

    conn = connect(db_path)
    initialize_schema(conn)
    conn.execute(
        """
        INSERT INTO production_lines (line_code, line_name, description)
        VALUES (
            'LINE-SECOM-01',
            'Public SECOM virtual line',
            'Virtual line metadata for public SECOM semiconductor sensor samples.'
        )
        """
    )
    line_id = int(conn.execute("SELECT line_id FROM production_lines WHERE line_code = 'LINE-SECOM-01'").fetchone()["line_id"])
    equipment_rows = [
        ("EQP-SECOM-001", line_id, "SECOM virtual tool A", "sensor aggregation"),
        ("EQP-SECOM-002", line_id, "SECOM virtual tool B", "sensor aggregation"),
        ("EQP-SECOM-003", line_id, "SECOM virtual tool C", "sensor aggregation"),
    ]
    conn.executemany(
        "INSERT INTO equipment (equipment_id, line_id, equipment_name, equipment_role) VALUES (?, ?, ?, ?)",
        equipment_rows,
    )

    equipment_ids = [row[0] for row in equipment_rows]
    prediction_rows = []
    reading_rows = []
    start_ts = datetime(2026, 5, 11, 9, 0, 0)
    selected_columns = list(x.columns[:SELECTED_SENSOR_COUNT])

    for sample_idx, (_, row) in enumerate(x[selected_columns].iterrows(), start=1):
        equipment_id = equipment_ids[(sample_idx - 1) % len(equipment_ids)]
        prediction_rows.append(
            (
                sample_idx,
                equipment_id,
                "logistic_regression_first_40_sensors",
                scores[sample_idx - 1],
                int(y.iloc[sample_idx - 1]),
            )
        )
        reading_ts = (start_ts + timedelta(minutes=sample_idx)).isoformat()
        for sensor_name in selected_columns:
            value = row[sensor_name]
            reading_rows.append(
                (
                    sample_idx,
                    equipment_id,
                    sensor_name,
                    None if pd.isna(value) else float(value),
                    reading_ts,
                )
            )

    conn.executemany(
        """
        INSERT INTO quality_predictions (sample_id, equipment_id, model_name, defect_score, true_defect)
        VALUES (?, ?, ?, ?, ?)
        """,
        prediction_rows,
    )
    conn.executemany(
        """
        INSERT INTO sensor_readings (sample_id, equipment_id, sensor_name, sensor_value, reading_ts)
        VALUES (?, ?, ?, ?, ?)
        """,
        reading_rows,
    )

    metric_input = [
        {"defect_score": score, "true_defect": int(label)}
        for score, label in zip(scores, y.tolist())
    ]
    metrics = calculate_metrics_from_rows(metric_input, DEFAULT_THRESHOLD)
    cursor = conn.execute(
        """
        INSERT INTO threshold_policies (
            threshold, precision_defect, recall_defect, f1_defect, false_alarm_rate,
            false_alarm_count, missed_defect_count, tp, tn, fp, fn
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            metrics["threshold"],
            metrics["precision_defect"],
            metrics["recall_defect"],
            metrics["f1_defect"],
            metrics["false_alarm_rate"],
            metrics["false_alarm_count"],
            metrics["missed_defect_count"],
            metrics["tp"],
            metrics["tn"],
            metrics["fp"],
            metrics["fn"],
        ),
    )
    policy_id = int(cursor.lastrowid)
    alert_count = refresh_alerts(conn, DEFAULT_THRESHOLD, policy_id)
    operation_events = seed_synthetic_operations_context(conn)
    digital_thread = seed_synthetic_digital_thread(conn)
    conn.commit()
    conn.close()
    return {
        "samples": len(x),
        "sensor_readings": len(reading_rows),
        "predictions": len(prediction_rows),
        "alerts": alert_count,
        "operation_events": operation_events,
        "digital_thread_rows": sum(digital_thread.values()),
    }
