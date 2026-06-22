from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


def api_get(path: str, **params: Any) -> Any:
    response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> Any:
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def show_api_error(error: Exception) -> None:
    st.error(f"API request failed: {error}")
    st.caption(f"API_BASE_URL={API_BASE_URL}")


def main() -> None:
    st.set_page_config(
        page_title="Manufacturing Operations Intelligence Platform",
        page_icon="MOI",
        layout="wide",
    )
    st.title("Semiconductor PE Quality Platform — Yield / Bin / Escape-Overkill")
    st.caption("Public SECOM sensor data -> SQLite -> FastAPI -> yield/bin/escape-overkill dashboard -> alert -> 8D report")

    try:
        health = api_get("/health")
    except Exception as exc:
        show_api_error(exc)
        st.stop()

    st.sidebar.success(f"API status: {health['status']}")
    threshold = st.sidebar.slider("Defect threshold", 0.0, 1.0, 0.50, 0.05)
    threshold_metrics = api_get("/quality/metrics", threshold=threshold)
    policies = api_get("/quality/policies")
    applied_policy = None
    if st.sidebar.button("Apply threshold policy"):
        applied_policy = api_post("/quality/threshold", {"threshold": threshold})
        policies = api_get("/quality/policies")

    overview_tab, sensor_tab, quality_tab, failure_tab, digital_thread_tab, vision_tab, report_tab = st.tabs(
        [
            "Overview",
            "Sensor Data",
            "Quality Model",
            "Failure Cases",
            "Digital Thread",
            "Vision Inspection",
            "8D Report",
        ]
    )

    with overview_tab:
        lines = api_get("/lines")
        alerts = api_get("/alerts", limit=200)
        predictions = api_get("/quality/predictions", limit=2000)
        operations = api_get("/operations/events", limit=12)

        cols = st.columns(5)
        cols[0].metric("Lines", len(lines))
        cols[1].metric("Predictions", len(predictions))
        cols[2].metric("Open alerts", len(alerts))
        cols[3].metric("Recall", f"{threshold_metrics['recall_defect']:.3f}")
        cols[4].metric("False alarms", threshold_metrics["false_alarm_count"])
        if applied_policy is not None:
            st.success(f"Applied threshold {applied_policy['threshold']:.2f}; refreshed {applied_policy['alert_count']} alerts.")

        st.subheader("Production Lines")
        st.dataframe(lines, use_container_width=True, hide_index=True)

        st.subheader("Recent Alerts")
        st.dataframe(alerts[:20], use_container_width=True, hide_index=True)

        st.subheader("Synthetic operations context")
        st.caption(
            "Production order, process step, and equipment event rows are synthetic design handles, not field records."
        )
        st.dataframe(operations, use_container_width=True, hide_index=True)

    with sensor_tab:
        st.subheader("SECOM Sensor Readings")
        equipment_id = st.selectbox(
            "Equipment",
            ["EQP-SECOM-001", "EQP-SECOM-002", "EQP-SECOM-003"],
        )
        limit = st.slider("Rows", 10, 500, 100, 10)
        readings = api_get(f"/equipment/{equipment_id}/readings", limit=limit)
        st.dataframe(readings, use_container_width=True, hide_index=True)
        st.caption(
            "SECOM feature names are anonymous. `sensor_000` style names are inspection handles, not real equipment causes."
        )

    with quality_tab:
        yield_summary = api_get("/quality/yield-summary", threshold=threshold)
        st.subheader("Yield / Bin / Escape-Overkill (PE headline)")
        ycols = st.columns(5)
        ycols[0].metric("Yield (ship rate)", f"{yield_summary['yield']:.3f}")
        ycols[1].metric("Shipped", yield_summary["shipped"])
        ycols[2].metric("Rejected", yield_summary["rejected"])
        ycols[3].metric("Escape (missed)", yield_summary["escape_count"])
        ycols[4].metric("Overkill (false reject)", yield_summary["overkill_count"])
        st.caption(
            "Bin = defect-score band; defect_rate shows where true defects concentrate. "
            + yield_summary["boundary_note"]
        )
        st.dataframe(yield_summary["bins"], use_container_width=True, hide_index=True)

        st.subheader("Threshold Metrics")
        cols = st.columns(6)
        cols[0].metric("Precision", f"{threshold_metrics['precision_defect']:.3f}")
        cols[1].metric("Recall", f"{threshold_metrics['recall_defect']:.3f}")
        cols[2].metric("F1", f"{threshold_metrics['f1_defect']:.3f}")
        cols[3].metric("False alarm rate", f"{threshold_metrics['false_alarm_rate']:.3f}")
        cols[4].metric("Missed defects", threshold_metrics["missed_defect_count"])
        cols[5].metric("Predicted defects", threshold_metrics["tp"] + threshold_metrics["fp"])

        confusion = [
            {
                "actual": "pass",
                "predicted_pass": threshold_metrics["tn"],
                "predicted_defect": threshold_metrics["fp"],
            },
            {
                "actual": "defect",
                "predicted_pass": threshold_metrics["fn"],
                "predicted_defect": threshold_metrics["tp"],
            },
        ]
        st.dataframe(confusion, use_container_width=True, hide_index=True)

        predictions = api_get("/quality/predictions", limit=100)
        st.subheader("Top Defect Scores")
        st.dataframe(predictions, use_container_width=True, hide_index=True)

        st.subheader("Policy History")
        active_policy = next((row for row in policies if row["is_active"]), None)
        previous_policy = next((row for row in policies if not row["is_active"]), None)
        policy_cols = st.columns(2)
        if active_policy:
            policy_cols[0].metric("Active policy", f"#{active_policy['policy_id']} / {active_policy['threshold']:.2f}")
            policy_cols[0].metric("Active recall", f"{active_policy['recall_defect']:.3f}")
            policy_cols[0].metric("Active false alarms", active_policy["false_alarm_count"])
        if previous_policy:
            policy_cols[1].metric("Previous policy", f"#{previous_policy['policy_id']} / {previous_policy['threshold']:.2f}")
            policy_cols[1].metric("Previous recall", f"{previous_policy['recall_defect']:.3f}")
            policy_cols[1].metric("Previous false alarms", previous_policy["false_alarm_count"])
        st.dataframe(policies, use_container_width=True, hide_index=True)

    with failure_tab:
        st.subheader("False Alarm / Missed Defect Review")
        false_alarms = api_get("/quality/failure-cases", threshold=threshold, case_type="false_alarm", limit=50)
        missed_defects = api_get("/quality/failure-cases", threshold=threshold, case_type="missed_defect", limit=50)
        st.caption("Use any sample_id from these rows in the Digital Thread tab to inspect the synthetic trace.")
        left, right = st.columns(2)
        left.write("False alarms")
        left.dataframe(false_alarms, use_container_width=True, hide_index=True)
        right.write("Missed defects")
        right.dataframe(missed_defects, use_container_width=True, hide_index=True)

        selected_cases = false_alarms + missed_defects
        if selected_cases:
            options = {
                f"{row['case_type']} / sample {row['sample_id']} / score {row['defect_score']:.3f}": row
                for row in selected_cases[:100]
            }
            selected_case = options[st.selectbox("Review case", list(options.keys()))]
            review_note = st.text_area("Review note", selected_case.get("latest_review_note") or "")
            root_cause_tag = st.text_input("Root cause tag", selected_case.get("latest_root_cause_tag") or "follow_up_only")
            next_data_needed = st.text_input(
                "Next data needed",
                selected_case.get("latest_next_data_needed") or "process step, equipment history, label review",
            )
            if st.button("Save follow-up review"):
                api_post(
                    "/quality/reviews",
                    {
                        "sample_id": selected_case["sample_id"],
                        "case_type": selected_case["case_type"],
                        "threshold": threshold,
                        "review_note": review_note,
                        "root_cause_tag": root_cause_tag,
                        "next_data_needed": next_data_needed,
                    },
                )
                st.success("Saved follow-up review note.")

    with digital_thread_tab:
        st.subheader("Digital Thread")
        st.warning("Public case-informed synthetic layer, not internal company internal data.")
        source_map = api_get("/digital-thread/source-map")
        st.caption(source_map["fixed_statement"])
        with st.expander("Source map"):
            st.dataframe(source_map["sources"], use_container_width=True, hide_index=True)

        lots = api_get("/digital-thread/lots")
        lot_options = {
            f"{row['lot_id']} / {row['lot_type']} / gates {row['quality_gate_count']}": row["lot_id"]
            for row in lots
        }
        selected_lot_label = st.selectbox("Synthetic lot", list(lot_options.keys()))
        selected_lot_id = lot_options[selected_lot_label]
        lot_detail = api_get(f"/digital-thread/lot/{selected_lot_id}")

        st.subheader("EBOM / MBOM / PBOM")
        bom_cols = st.columns(3)
        for col, bom_type in zip(bom_cols, ["EBOM", "MBOM", "PBOM"]):
            col.write(bom_type)
            bom_rows = [row for row in lot_detail["bom_items"] if row["bom_type"] == bom_type]
            col.dataframe(bom_rows, use_container_width=True, hide_index=True)

        st.subheader("BOP Steps")
        st.dataframe(lot_detail["bop_steps"], use_container_width=True, hide_index=True)

        st.subheader("Linked Quality Gates")
        st.dataframe(lot_detail["quality_gates"][:50], use_container_width=True, hide_index=True)

        st.subheader("Failure Case Trace")
        trace_sample_id = st.number_input("SECOM sample_id", min_value=1, max_value=1567, value=1, step=1)
        trace = api_get("/digital-thread/trace", sample_id=int(trace_sample_id))
        flow_rows = [
            {"stage": "sample", "value": trace["sample_id"]},
            {"stage": "lot", "value": trace["lot"]["lot_id"]},
            {"stage": "BOP", "value": trace["quality_gate"]["bop_step_id"]},
            {"stage": "quality gate", "value": trace["quality_gate"]["quality_signal"]},
            {"stage": "prediction", "value": f"{trace['prediction']['defect_score']:.3f}"},
            {
                "stage": "alert",
                "value": trace["active_alert"]["alert_id"] if trace["active_alert"] else "no active alert",
            },
            {
                "stage": "review note",
                "value": trace["latest_review"]["review_note"] if trace["latest_review"] else "not reviewed",
            },
        ]
        st.dataframe(flow_rows, use_container_width=True, hide_index=True)
        trace_cols = st.columns(2)
        trace_cols[0].write("Prediction")
        trace_cols[0].json(trace["prediction"])
        trace_cols[1].write("8D report candidate")
        trace_cols[1].json(trace["eight_d_report_candidate"])

    with vision_tab:
        st.subheader("MVTec AD Bottle")
        status = api_get("/vision/status")
        if not status["installed"]:
            st.warning(status["message"])
            st.caption(status["license_note"])
        else:
            st.success(status["message"])
            st.json(status)

    with report_tab:
        st.subheader("8D Report")
        alerts = api_get("/alerts", limit=100)
        if not alerts:
            st.info("No alerts for the current threshold.")
        else:
            alert_options = {f"Alert {row['alert_id']} / sample {row['sample_id']}": row["alert_id"] for row in alerts}
            selected_label = st.selectbox("Alert", list(alert_options.keys()))
            owner = st.text_input("Owner", "portfolio-demo")
            title = st.text_input("Title", "SECOM defect-score follow-up")
            if st.button("Create 8D Report"):
                created = api_post(
                    "/reports/8d",
                    {
                        "alert_id": alert_options[selected_label],
                        "title": title,
                        "owner": owner,
                    },
                )
                st.session_state["last_report_id"] = created["report_id"]

            report_id = st.session_state.get("last_report_id")
            if report_id:
                report = api_get(f"/reports/8d/{report_id}")
                st.markdown(report["report_markdown"])


if __name__ == "__main__":
    main()
