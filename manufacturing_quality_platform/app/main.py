from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .db import DEFAULT_DB_PATH, connect, initialize_schema, rows_to_dicts
from .seed import calculate_metrics_from_rows, refresh_alerts


class SensorReadingIn(BaseModel):
    sample_id: int
    sensor_name: str
    sensor_value: Optional[float] = None


class SensorBatchIn(BaseModel):
    equipment_id: str
    readings: List[SensorReadingIn] = Field(min_length=1)


class ThresholdIn(BaseModel):
    threshold: float = Field(ge=0.0, le=1.0)


class EightDReportIn(BaseModel):
    alert_id: int
    title: str = "SECOM defect-score follow-up"
    owner: str = "portfolio-demo"


class QualityReviewIn(BaseModel):
    sample_id: int
    case_type: Literal["false_alarm", "missed_defect"]
    threshold: float = Field(ge=0.0, le=1.0)
    review_note: str = Field(min_length=1, max_length=1000)
    root_cause_tag: str = Field(min_length=1, max_length=120)
    next_data_needed: str = Field(min_length=1, max_length=500)


DIGITAL_THREAD_BOUNDARY_NOTE = "Public case-informed synthetic layer, not internal company internal data."
DIGITAL_THREAD_FIXED_STATEMENT = (
    "실제 특정 회사 데이터가 아니라, 공개 SECOM 품질 데이터 위에 PE 관점의 "
    "yield/bin/escape-overkill 대시보드를 구현하고, 보조로 generic 반도체 "
    "design->mask/process->wafer-test process-traceability(EBOM/MBOM/PBOM/BOP) "
    "synthetic 계층을 붙였다."
)
DIGITAL_THREAD_SOURCES = [
    {
        "source_name": "Generic semiconductor design-to-test flow (public concept)",
        "public_url": "https://en.wikipedia.org/wiki/Semiconductor_device_fabrication",
        "used_for": "design -> mask/process -> wafer-test traceability vocabulary (EBOM/MBOM/PBOM/BOP) for the secondary synthetic layer.",
        "boundary_note": "Public concept only. No company fab, MES, PLM, or ERP data is used.",
    },
    {
        "source_name": "PLM digital-thread concept (public descriptions)",
        "public_url": "https://en.wikipedia.org/wiki/Digital_thread",
        "used_for": "EBOM/MBOM/PBOM/BOP design-to-production traceability structure for the synthetic layer.",
        "boundary_note": "Structure only. This project does not claim access to any commercial PLM platform.",
    },
]


def calculate_metrics(conn, threshold: float) -> dict[str, Any]:
    rows = rows_to_dicts(
        conn.execute(
            "SELECT defect_score, true_defect FROM quality_predictions"
        ).fetchall()
    )
    return calculate_metrics_from_rows(rows, threshold)


def calculate_yield_summary(conn, threshold: float) -> dict[str, Any]:
    """PE headline view: yield, score-band bins, and escape/overkill at a threshold.
    score >= threshold => reject(fail bin); a shipped true-defect is an escape, a
    rejected good die is overkill. Public SECOM, illustrative — not production yield."""
    rows = rows_to_dicts(
        conn.execute("SELECT defect_score, true_defect FROM quality_predictions").fetchall()
    )
    total = len(rows)
    shipped = rejected = escape = overkill = 0
    bands = [(0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.01)]
    bins = [{"bin": f"{lo:.2f}-{min(hi, 1.0):.2f}", "count": 0, "true_defect_count": 0} for lo, hi in bands]
    for r in rows:
        score = float(r["defect_score"])
        defect = int(r["true_defect"])
        if score < threshold:
            shipped += 1
            if defect == 1:
                escape += 1            # shipped a true defect -> escape (missed)
        else:
            rejected += 1
            if defect == 0:
                overkill += 1          # rejected a good die -> overkill (false alarm)
        for i, (lo, hi) in enumerate(bands):
            if lo <= score < hi:
                bins[i]["count"] += 1
                bins[i]["true_defect_count"] += defect
                break
    for b in bins:
        b["defect_rate"] = round(b["true_defect_count"] / b["count"], 4) if b["count"] else 0.0
    return {
        "threshold": threshold,
        "total": total,
        "yield": round(shipped / total, 4) if total else 0.0,
        "shipped": shipped,
        "rejected": rejected,
        "escape_count": escape,
        "overkill_count": overkill,
        "escape_rate": round(escape / total, 4) if total else 0.0,
        "overkill_rate": round(overkill / total, 4) if total else 0.0,
        "bins": bins,
        "boundary_note": (
            "Yield / bin / escape-overkill over public UCI SECOM (first-40-sensor logistic "
            "regression). Illustrative operating-point behavior, not production yield."
        ),
    }


def normalize_synthetic_flags(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        if "is_synthetic" in row:
            row["is_synthetic"] = bool(row["is_synthetic"])
    return rows


def create_app(db_path: Path | str = DEFAULT_DB_PATH) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        with connect(app.state.db_path) as conn:
            initialize_schema(conn)
        yield

    app = FastAPI(
        title="Manufacturing Operations Intelligence Platform",
        description=(
            "Public SECOM manufacturing sensor data demo with SQLite, API, "
            "threshold policies, alerts, and 8D report generation."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.db_path = Path(db_path)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ingest/sensor-batch")
    def ingest_sensor_batch(payload: SensorBatchIn) -> dict[str, int]:
        with connect(app.state.db_path) as conn:
            equipment = conn.execute(
                "SELECT equipment_id FROM equipment WHERE equipment_id = ?",
                (payload.equipment_id,),
            ).fetchone()
            if equipment is None:
                raise HTTPException(status_code=404, detail="equipment not found")
            conn.executemany(
                """
                INSERT INTO sensor_readings (
                    sample_id, equipment_id, sensor_name, sensor_value, reading_ts
                )
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                [
                    (
                        reading.sample_id,
                        payload.equipment_id,
                        reading.sensor_name,
                        reading.sensor_value,
                    )
                    for reading in payload.readings
                ],
            )
            conn.commit()
        return {"inserted": len(payload.readings)}

    @app.get("/lines")
    def lines() -> list[dict[str, Any]]:
        with connect(app.state.db_path) as conn:
            return rows_to_dicts(
                conn.execute(
                    """
                    SELECT
                        pl.line_id,
                        pl.line_code,
                        pl.line_name,
                        pl.description,
                        COUNT(e.equipment_id) AS equipment_count
                    FROM production_lines pl
                    LEFT JOIN equipment e ON e.line_id = pl.line_id
                    GROUP BY pl.line_id
                    ORDER BY pl.line_code
                    """
                ).fetchall()
            )

    @app.get("/equipment/{equipment_id}/readings")
    def equipment_readings(
        equipment_id: str,
        limit: int = Query(default=100, ge=1, le=1000),
        sample_id: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT sample_id, equipment_id, sensor_name, sensor_value, reading_ts
            FROM sensor_readings
            WHERE equipment_id = ?
        """
        params: list[Any] = [equipment_id]
        if sample_id is not None:
            query += " AND sample_id = ?"
            params.append(sample_id)
        query += " ORDER BY sample_id, sensor_name LIMIT ?"
        params.append(limit)
        with connect(app.state.db_path) as conn:
            rows = rows_to_dicts(conn.execute(query, params).fetchall())
        if not rows:
            raise HTTPException(status_code=404, detail="readings not found")
        return rows

    @app.get("/quality/predictions")
    def predictions(
        limit: int = Query(default=100, ge=1, le=2000),
        offset: int = Query(default=0, ge=0),
        equipment_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT sample_id, equipment_id, model_name, defect_score, true_defect
            FROM quality_predictions
        """
        params: list[Any] = []
        if equipment_id is not None:
            query += " WHERE equipment_id = ?"
            params.append(equipment_id)
        query += " ORDER BY defect_score DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with connect(app.state.db_path) as conn:
            return rows_to_dicts(conn.execute(query, params).fetchall())

    @app.get("/quality/metrics")
    def quality_metrics(
        threshold: float = Query(default=0.50, ge=0.0, le=1.0),
    ) -> dict[str, Any]:
        with connect(app.state.db_path) as conn:
            return calculate_metrics(conn, threshold)

    @app.get("/quality/yield-summary")
    def quality_yield_summary(
        threshold: float = Query(default=0.50, ge=0.0, le=1.0),
    ) -> dict[str, Any]:
        with connect(app.state.db_path) as conn:
            return calculate_yield_summary(conn, threshold)

    @app.post("/quality/threshold")
    def threshold_policy(payload: ThresholdIn) -> dict[str, Any]:
        with connect(app.state.db_path) as conn:
            metrics = calculate_metrics(conn, payload.threshold)
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
            alert_count = refresh_alerts(conn, payload.threshold, policy_id)
            conn.commit()
        metrics["policy_id"] = policy_id
        metrics["alert_count"] = alert_count
        return metrics

    @app.get("/quality/policies")
    def quality_policies(limit: int = Query(default=20, ge=1, le=200)) -> list[dict[str, Any]]:
        with connect(app.state.db_path) as conn:
            active_policy_id = conn.execute(
                "SELECT MAX(policy_id) AS policy_id FROM threshold_policies"
            ).fetchone()["policy_id"]
            policies = rows_to_dicts(
                conn.execute(
                    """
                    SELECT
                        policy_id,
                        threshold,
                        precision_defect,
                        recall_defect,
                        f1_defect,
                        false_alarm_rate,
                        false_alarm_count,
                        missed_defect_count,
                        tp,
                        tn,
                        fp,
                        fn,
                        created_at
                    FROM threshold_policies
                    ORDER BY policy_id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            )
        for policy in policies:
            policy["is_active"] = policy["policy_id"] == active_policy_id
        return policies

    @app.get("/alerts")
    def alerts(
        limit: int = Query(default=100, ge=1, le=2000),
        status: Optional[str] = "open",
        policy_id: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT alert_id, policy_id, sample_id, equipment_id, alert_type, severity,
                   defect_score, threshold, status, created_at
            FROM alerts
        """
        params: list[Any] = []
        filters = []
        with connect(app.state.db_path) as conn:
            active_policy_id = conn.execute(
                "SELECT MAX(policy_id) AS policy_id FROM threshold_policies"
            ).fetchone()["policy_id"]
        selected_policy_id = policy_id if policy_id is not None else active_policy_id
        if selected_policy_id is not None:
            filters.append("policy_id = ?")
            params.append(selected_policy_id)
        if status is not None:
            filters.append("status = ?")
            params.append(status)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY defect_score DESC LIMIT ?"
        params.append(limit)
        with connect(app.state.db_path) as conn:
            return rows_to_dicts(conn.execute(query, params).fetchall())

    @app.get("/quality/failure-cases")
    def failure_cases(
        threshold: float = Query(default=0.50, ge=0.0, le=1.0),
        case_type: Literal["false_alarm", "missed_defect"] = "false_alarm",
        limit: int = Query(default=100, ge=1, le=2000),
    ) -> list[dict[str, Any]]:
        if case_type == "false_alarm":
            predicate = "qp.defect_score >= ? AND qp.true_defect = 0"
            order_clause = "qp.defect_score DESC"
        else:
            predicate = "qp.defect_score < ? AND qp.true_defect = 1"
            order_clause = "qp.defect_score DESC"
        with connect(app.state.db_path) as conn:
            rows = rows_to_dicts(
                conn.execute(
                    f"""
                    SELECT
                        qp.sample_id,
                        qp.equipment_id,
                        qp.model_name,
                        qp.defect_score,
                        qp.true_defect,
                        ? AS threshold,
                        ? AS case_type,
                        qr.review_note AS latest_review_note,
                        qr.root_cause_tag AS latest_root_cause_tag,
                        qr.next_data_needed AS latest_next_data_needed,
                        qr.created_at AS latest_reviewed_at
                    FROM quality_predictions qp
                    LEFT JOIN quality_reviews qr
                        ON qr.review_id = (
                            SELECT review_id
                            FROM quality_reviews
                            WHERE sample_id = qp.sample_id
                              AND case_type = ?
                            ORDER BY created_at DESC, review_id DESC
                            LIMIT 1
                        )
                    WHERE {predicate}
                    ORDER BY {order_clause}
                    LIMIT ?
                    """,
                    (threshold, case_type, case_type, threshold, limit),
                ).fetchall()
            )
        return rows

    @app.post("/quality/reviews")
    def create_quality_review(payload: QualityReviewIn) -> dict[str, Any]:
        with connect(app.state.db_path) as conn:
            prediction = conn.execute(
                "SELECT sample_id FROM quality_predictions WHERE sample_id = ?",
                (payload.sample_id,),
            ).fetchone()
            if prediction is None:
                raise HTTPException(status_code=404, detail="sample not found")
            cursor = conn.execute(
                """
                INSERT INTO quality_reviews (
                    sample_id, case_type, threshold, review_note, root_cause_tag, next_data_needed
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.sample_id,
                    payload.case_type,
                    payload.threshold,
                    payload.review_note,
                    payload.root_cause_tag,
                    payload.next_data_needed,
                ),
            )
            conn.commit()
            review_id = int(cursor.lastrowid)
        return {
            "review_id": review_id,
            "sample_id": payload.sample_id,
            "case_type": payload.case_type,
            "note_type": "follow_up_review_note",
        }

    @app.get("/operations/events")
    def operations_events(
        sample_id: Optional[int] = None,
        limit: int = Query(default=100, ge=1, le=2000),
    ) -> list[dict[str, Any]]:
        params: list[Any] = [limit]
        where = ""
        if sample_id is not None:
            where = "WHERE sample_id = ?"
            params = [sample_id, limit]
        with connect(app.state.db_path) as conn:
            orders = rows_to_dicts(
                conn.execute(
                    f"""
                    SELECT
                        'production_order' AS context_type,
                        sample_id,
                        NULL AS equipment_id,
                        order_id AS context_id,
                        product_family AS context_value,
                        synthetic_note,
                        is_synthetic,
                        NULL AS event_ts
                    FROM production_orders
                    {where}
                    ORDER BY sample_id
                    LIMIT ?
                    """,
                    tuple(params),
                ).fetchall()
            )
            steps = rows_to_dicts(
                conn.execute(
                    f"""
                    SELECT
                        'process_step' AS context_type,
                        sample_id,
                        NULL AS equipment_id,
                        CAST(step_id AS TEXT) AS context_id,
                        step_name AS context_value,
                        synthetic_note,
                        is_synthetic,
                        NULL AS event_ts
                    FROM process_steps
                    {where}
                    ORDER BY sample_id
                    LIMIT ?
                    """,
                    tuple(params),
                ).fetchall()
            )
            events = rows_to_dicts(
                conn.execute(
                    f"""
                    SELECT
                        'equipment_event' AS context_type,
                        sample_id,
                        equipment_id,
                        CAST(event_id AS TEXT) AS context_id,
                        event_type || ': ' || event_detail AS context_value,
                        synthetic_note,
                        is_synthetic,
                        event_ts
                    FROM equipment_events
                    {where}
                    ORDER BY sample_id
                    LIMIT ?
                    """,
                    tuple(params),
                ).fetchall()
            )
        combined: list[dict[str, Any]] = []
        max_rows = max(len(orders), len(steps), len(events))
        for index in range(max_rows):
            for group in (orders, steps, events):
                if index < len(group):
                    combined.append(group[index])
                if len(combined) >= limit:
                    break
            if len(combined) >= limit:
                break
        for row in combined:
            row["is_synthetic"] = bool(row["is_synthetic"])
        return combined

    @app.get("/digital-thread/source-map")
    def digital_thread_source_map() -> dict[str, Any]:
        return {
            "boundary_note": DIGITAL_THREAD_BOUNDARY_NOTE,
            "fixed_statement": DIGITAL_THREAD_FIXED_STATEMENT,
            "sources": DIGITAL_THREAD_SOURCES,
        }

    @app.get("/digital-thread/lots")
    def digital_thread_lots() -> list[dict[str, Any]]:
        with connect(app.state.db_path) as conn:
            rows = rows_to_dicts(
                conn.execute(
                    """
                    SELECT
                        sb.lot_id,
                        sb.device_family,
                        sb.lot_type,
                        sb.source_note,
                        sb.is_synthetic,
                        COALESCE(SUM(CASE WHEN bi.bom_type = 'EBOM' THEN 1 ELSE 0 END), 0) AS ebom_count,
                        COALESCE(SUM(CASE WHEN bi.bom_type = 'MBOM' THEN 1 ELSE 0 END), 0) AS mbom_count,
                        COALESCE(SUM(CASE WHEN bi.bom_type = 'PBOM' THEN 1 ELSE 0 END), 0) AS pbom_count,
                        (
                            SELECT COUNT(*)
                            FROM bop_steps bs
                            WHERE bs.lot_id = sb.lot_id
                        ) AS bop_count,
                        (
                            SELECT COUNT(*)
                            FROM quality_gates qg
                            WHERE qg.lot_id = sb.lot_id
                        ) AS quality_gate_count
                    FROM process_lots sb
                    LEFT JOIN bom_items bi ON bi.lot_id = sb.lot_id
                    GROUP BY sb.lot_id
                    ORDER BY sb.lot_id
                    """
                ).fetchall()
            )
        return normalize_synthetic_flags(rows)

    @app.get("/digital-thread/lot/{lot_id}")
    def digital_thread_lot(lot_id: str) -> dict[str, Any]:
        with connect(app.state.db_path) as conn:
            lot = conn.execute(
                "SELECT * FROM process_lots WHERE lot_id = ?",
                (lot_id,),
            ).fetchone()
            if lot is None:
                raise HTTPException(status_code=404, detail="lot not found")
            bom_items = rows_to_dicts(
                conn.execute(
                    """
                    SELECT *
                    FROM bom_items
                    WHERE lot_id = ?
                    ORDER BY bom_type, bom_item_id
                    """,
                    (lot_id,),
                ).fetchall()
            )
            bop_steps = rows_to_dicts(
                conn.execute(
                    """
                    SELECT *
                    FROM bop_steps
                    WHERE lot_id = ?
                    ORDER BY step_sequence
                    """,
                    (lot_id,),
                ).fetchall()
            )
            quality_gates = rows_to_dicts(
                conn.execute(
                    """
                    SELECT *
                    FROM quality_gates
                    WHERE lot_id = ?
                    ORDER BY sample_id
                    """,
                    (lot_id,),
                ).fetchall()
            )
        return {
            "boundary_note": DIGITAL_THREAD_BOUNDARY_NOTE,
            "lot": normalize_synthetic_flags([dict(lot)])[0],
            "bom_items": normalize_synthetic_flags(bom_items),
            "bop_steps": normalize_synthetic_flags(bop_steps),
            "quality_gates": normalize_synthetic_flags(quality_gates),
        }

    @app.get("/digital-thread/trace")
    def digital_thread_trace(sample_id: int = Query(..., ge=1)) -> dict[str, Any]:
        with connect(app.state.db_path) as conn:
            prediction = conn.execute(
                """
                SELECT sample_id, equipment_id, model_name, defect_score, true_defect
                FROM quality_predictions
                WHERE sample_id = ?
                """,
                (sample_id,),
            ).fetchone()
            if prediction is None:
                raise HTTPException(status_code=404, detail="sample not found")
            quality_gate = conn.execute(
                """
                SELECT *
                FROM quality_gates
                WHERE sample_id = ?
                """,
                (sample_id,),
            ).fetchone()
            if quality_gate is None:
                raise HTTPException(status_code=404, detail="digital thread link not found")
            lot_id = quality_gate["lot_id"]
            lot = conn.execute(
                "SELECT * FROM process_lots WHERE lot_id = ?",
                (lot_id,),
            ).fetchone()
            bom_items = rows_to_dicts(
                conn.execute(
                    """
                    SELECT *
                    FROM bom_items
                    WHERE lot_id = ?
                    ORDER BY bom_type, bom_item_id
                    """,
                    (lot_id,),
                ).fetchall()
            )
            bop_steps = rows_to_dicts(
                conn.execute(
                    """
                    SELECT *
                    FROM bop_steps
                    WHERE lot_id = ?
                    ORDER BY step_sequence
                    """,
                    (lot_id,),
                ).fetchall()
            )
            active_policy_id = conn.execute(
                "SELECT MAX(policy_id) AS policy_id FROM threshold_policies"
            ).fetchone()["policy_id"]
            active_alert = None
            if active_policy_id is not None:
                active_alert = conn.execute(
                    """
                    SELECT alert_id, policy_id, sample_id, equipment_id, alert_type,
                           severity, defect_score, threshold, status, created_at
                    FROM alerts
                    WHERE sample_id = ?
                      AND policy_id = ?
                      AND status = 'open'
                    ORDER BY defect_score DESC, alert_id DESC
                    LIMIT 1
                    """,
                    (sample_id, active_policy_id),
                ).fetchone()
            latest_review = conn.execute(
                """
                SELECT review_id, sample_id, case_type, threshold, review_note,
                       root_cause_tag, next_data_needed, created_at
                FROM quality_reviews
                WHERE sample_id = ?
                ORDER BY created_at DESC, review_id DESC
                LIMIT 1
                """,
                (sample_id,),
            ).fetchone()

        active_alert_dict = dict(active_alert) if active_alert is not None else None
        latest_review_dict = dict(latest_review) if latest_review is not None else None
        return {
            "boundary_note": DIGITAL_THREAD_BOUNDARY_NOTE,
            "sample_id": sample_id,
            "lot": normalize_synthetic_flags([dict(lot)])[0],
            "bom_items": normalize_synthetic_flags(bom_items),
            "bop_steps": normalize_synthetic_flags(bop_steps),
            "quality_gate": normalize_synthetic_flags([dict(quality_gate)])[0],
            "prediction": dict(prediction),
            "active_alert": active_alert_dict,
            "latest_review": latest_review_dict,
            "eight_d_report_candidate": {
                "available": active_alert_dict is not None,
                "reason": (
                    "An active alert can be used with POST /reports/8d."
                    if active_alert_dict is not None
                    else "No active alert for the current threshold policy."
                ),
            },
        }

    @app.post("/reports/8d")
    def create_8d_report(payload: EightDReportIn) -> dict[str, Any]:
        with connect(app.state.db_path) as conn:
            alert = conn.execute(
                "SELECT * FROM alerts WHERE alert_id = ?",
                (payload.alert_id,),
            ).fetchone()
            if alert is None:
                raise HTTPException(status_code=404, detail="alert not found")
            review_rows = rows_to_dicts(
                conn.execute(
                    """
                    SELECT case_type, threshold, review_note, root_cause_tag, next_data_needed
                    FROM quality_reviews
                    WHERE sample_id = ?
                    ORDER BY created_at DESC, review_id DESC
                    LIMIT 3
                    """,
                    (alert["sample_id"],),
                ).fetchall()
            )
            review_markdown = "\n".join(
                [
                    (
                        f"- {row['case_type']} at threshold {row['threshold']:.2f}: "
                        f"{row['review_note']} "
                        f"(tag: {row['root_cause_tag']}; next data: {row['next_data_needed']})"
                    )
                    for row in review_rows
                ]
            ) or "- No follow-up review note has been recorded yet."

            problem_statement = (
                f"Public SECOM sample {alert['sample_id']} on {alert['equipment_id']} "
                f"exceeded the active defect threshold {alert['threshold']:.2f} "
                f"with score {alert['defect_score']:.3f}."
            )
            containment_action = (
                "Hold this public SECOM sample as a demo follow-up case, compare nearby "
                "anonymous sensors, and do not claim a real line stop or confirmed field cause."
            )
            root_cause_hypothesis = (
                "Anonymous sensor pattern drift or missing-value sensitivity. SECOM does not "
                "provide process step names, equipment history, operator data, or verified root cause labels."
            )
            corrective_action = (
                "Tune threshold policy, review false alarms and missed defects together, and require "
                "real MES/PLM/equipment context before operational deployment."
            )
            verification_plan = (
                "Track recall, false alarm count, missed defect count, and failure-case review notes "
                "after every threshold change."
            )
            lessons_learned = (
                "A useful digital manufacturing artifact connects model output to DB, API, alert, "
                "dashboard, and report flow without overstating public-data evidence."
            )
            report_markdown = "\n\n".join(
                [
                    f"# 8D Report: {payload.title}",
                    f"- Owner: {payload.owner}",
                    f"- Alert ID: {payload.alert_id}",
                    "## D2 Problem",
                    problem_statement,
                    "## D3 Containment",
                    containment_action,
                    "## D4 Root Cause Hypothesis",
                    root_cause_hypothesis,
                    "## Failure Review Notes",
                    review_markdown,
                    "## D5-D6 Corrective Action",
                    corrective_action,
                    "## D7 Verification",
                    verification_plan,
                    "## D8 Lessons Learned",
                    lessons_learned,
                ]
            )
            cursor = conn.execute(
                """
                INSERT INTO eight_d_reports (
                    alert_id, title, owner, problem_statement, containment_action,
                    root_cause_hypothesis, corrective_action, verification_plan,
                    lessons_learned, report_markdown
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.alert_id,
                    payload.title,
                    payload.owner,
                    problem_statement,
                    containment_action,
                    root_cause_hypothesis,
                    corrective_action,
                    verification_plan,
                    lessons_learned,
                    report_markdown,
                ),
            )
            conn.commit()
            report_id = int(cursor.lastrowid)
        return {"report_id": report_id, "alert_id": payload.alert_id}

    @app.get("/reports/8d/{report_id}")
    def get_8d_report(report_id: int) -> dict[str, Any]:
        with connect(app.state.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM eight_d_reports WHERE report_id = ?",
                (report_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="report not found")
        return dict(row)

    @app.get("/vision/status")
    def vision_status() -> dict[str, Any]:
        bottle_root = Path(__file__).resolve().parents[1] / "data" / "mvtec" / "bottle"
        installed = bottle_root.exists()
        image_count = len(list(bottle_root.rglob("*.png"))) + len(list(bottle_root.rglob("*.jpg"))) if installed else 0
        if not installed:
            return {
                "installed": False,
                "message": "MVTec data not installed",
                "dataset": "MVTec AD bottle",
                "license_note": "MVTec AD is for non-commercial research and educational use.",
            }
        return {
            "installed": True,
            "message": "MVTec bottle data detected",
            "dataset": "MVTec AD bottle",
            "image_count": image_count,
            "model_note": "Optional vision scoring hook; keep SECOM platform usable without image data.",
            "license_note": "MVTec AD is for non-commercial research and educational use.",
        }

    return app


app = create_app()
