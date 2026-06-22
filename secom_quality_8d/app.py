from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports"


@st.cache_data
def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORT_DIR / name)


def read_text(name: str) -> str:
    return (REPORT_DIR / name).read_text(encoding="utf-8")


def metric_row(score_df: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    pred = (score_df["defect_score"] >= threshold).astype(int)
    y = score_df["true_defect"].astype(int)

    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    false_alarm_rate = fp / (fp + tn) if fp + tn else 0.0

    return {
        "precision_defect": precision,
        "recall_defect": recall,
        "f1_defect": f1,
        "false_alarm_rate": false_alarm_rate,
        "missed_defect_count": fn,
        "false_alarm_count": fp,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def format_rate(value: float) -> str:
    return f"{value:.3f}"


def cases_for_threshold(score_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    cases = score_df.copy()
    cases["threshold"] = threshold
    cases["predicted_defect"] = (cases["defect_score"] >= threshold).astype(int)
    cases["error_type"] = "correct"
    cases.loc[
        (cases["true_defect"] == 0) & (cases["predicted_defect"] == 1),
        "error_type",
    ] = "false_alarm"
    cases.loc[
        (cases["true_defect"] == 1) & (cases["predicted_defect"] == 0),
        "error_type",
    ] = "missed_defect"
    return cases[cases["error_type"] != "correct"].sort_values("defect_score", ascending=False)


def main() -> None:
    st.set_page_config(
        page_title="Manufacturing Quality Intelligence Demo",
        page_icon="MQI",
        layout="wide",
    )
    st.title("Manufacturing Quality Intelligence Demo")
    st.caption("SECOM public manufacturing sensor data -> metrics -> failure cases -> 8D report")

    required = [
        "data_profile.csv",
        "metrics.csv",
        "threshold_tradeoff.csv",
        "sample_scores.csv",
        "top_features.csv",
        "top_missing_features.csv",
        "error_cases.csv",
        "8d_report.md",
    ]
    missing = [name for name in required if not (REPORT_DIR / name).exists()]
    if missing:
        st.error("Run `python scripts/run_analysis.py` first. Missing: " + ", ".join(missing))
        st.stop()

    profile = read_csv("data_profile.csv")
    metrics = read_csv("metrics.csv")
    threshold_df = read_csv("threshold_tradeoff.csv")
    scores = read_csv("sample_scores.csv")
    top_features = read_csv("top_features.csv")
    top_missing = read_csv("top_missing_features.csv")

    strategy_options = sorted(scores["strategy"].unique())
    model_options = sorted(scores["model"].unique())

    with st.sidebar:
        st.header("Operating Point")
        strategy = st.selectbox("Preprocessing strategy", strategy_options)
        model = st.selectbox("Model", model_options)
        threshold = st.slider("Defect threshold", min_value=0.05, max_value=0.90, value=0.10, step=0.05)
        st.caption("Lower threshold usually raises recall and false alarms together.")

    selected_scores = scores[(scores["strategy"] == strategy) & (scores["model"] == model)].copy()
    selected_metrics = metric_row(selected_scores, threshold)

    dataset_tab, metrics_tab, failure_tab, report_tab = st.tabs(
        ["Dataset", "Metrics", "Failure Cases", "8D Report"]
    )

    with dataset_tab:
        st.subheader("Dataset Profile")
        cols = st.columns(4)
        profile_map = dict(zip(profile["metric"], profile["value"]))
        cols[0].metric("Samples", f"{int(float(profile_map['samples'])):,}")
        cols[1].metric("Features", f"{int(float(profile_map['features'])):,}")
        cols[2].metric("Defects", f"{int(float(profile_map['defect_count'])):,}")
        cols[3].metric("Defect rate", f"{float(profile_map['defect_rate']):.2%}")

        st.markdown(
            "This demo uses the public UCI SECOM dataset. It is not field data, "
            "and anonymous features are treated as inspection candidates rather than confirmed causes."
        )
        left, right = st.columns(2)
        with left:
            st.write("Top missing features")
            st.dataframe(top_missing.head(15), use_container_width=True, hide_index=False)
        with right:
            st.write("Baseline metric table at threshold 0.50")
            st.dataframe(metrics, use_container_width=True, hide_index=True)

    with metrics_tab:
        st.subheader("Threshold Trade-off")
        cols = st.columns(6)
        cols[0].metric("Precision", format_rate(selected_metrics["precision_defect"]))
        cols[1].metric("Recall", format_rate(selected_metrics["recall_defect"]))
        cols[2].metric("F1", format_rate(selected_metrics["f1_defect"]))
        cols[3].metric("False alarm", format_rate(selected_metrics["false_alarm_rate"]))
        cols[4].metric("Missed defects", str(selected_metrics["missed_defect_count"]))
        cols[5].metric("False alarms", str(selected_metrics["false_alarm_count"]))

        confusion = pd.DataFrame(
            [
                {"actual": "pass", "predicted_pass": selected_metrics["tn"], "predicted_defect": selected_metrics["fp"]},
                {"actual": "defect", "predicted_pass": selected_metrics["fn"], "predicted_defect": selected_metrics["tp"]},
            ]
        )
        st.write("Confusion matrix")
        st.dataframe(confusion, use_container_width=True, hide_index=True)

        curve = threshold_df[(threshold_df["strategy"] == strategy) & (threshold_df["model"] == model)][
            ["threshold", "precision_defect", "recall_defect", "false_alarm_rate", "f1_defect"]
        ].set_index("threshold")
        st.line_chart(curve)

        st.write("Top feature candidates")
        st.dataframe(
            top_features[(top_features["strategy"] == strategy) & (top_features["model"] == model)].head(10),
            use_container_width=True,
            hide_index=True,
        )

    with failure_tab:
        st.subheader("False Positive / False Negative Cases")
        errors = cases_for_threshold(selected_scores, threshold)
        false_alarm = errors[errors["error_type"] == "false_alarm"].head(20)
        missed_defect = errors[errors["error_type"] == "missed_defect"].sort_values("defect_score", ascending=False).head(20)

        left, right = st.columns(2)
        with left:
            st.write("False alarm: pass sample predicted as defect")
            st.dataframe(false_alarm, use_container_width=True, hide_index=True)
        with right:
            st.write("Missed defect: defect sample predicted as pass")
            st.dataframe(missed_defect, use_container_width=True, hide_index=True)

        st.caption(
            "Use these rows as a failure-case checklist. In real data, each row should be joined with process step, equipment, operator, and label-review history."
        )

    with report_tab:
        st.subheader("8D Report")
        st.markdown(read_text("8d_report.md"))


if __name__ == "__main__":
    main()
