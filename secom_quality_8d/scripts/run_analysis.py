from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
LABEL_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"


def download_if_missing(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    data_path = RAW_DIR / "secom.data"
    label_path = RAW_DIR / "secom_labels.data"
    download_if_missing(DATA_URL, data_path)
    download_if_missing(LABEL_URL, label_path)

    x = pd.read_csv(data_path, sep=r"\s+", header=None, na_values="NaN")
    labels = pd.read_csv(label_path, sep=r"\s+", header=None, names=["label", "timestamp"])
    x.columns = [f"f_{idx:03d}" for idx in range(x.shape[1])]

    # UCI SECOM labels use -1 for pass and 1 for fail. Use 1 as the defect class.
    y = (labels["label"] == 1).astype(int)
    return x, y


def metric_row(strategy: str, model_name: str, y_true: pd.Series, proba: pd.Series, threshold: float) -> dict[str, float | int | str]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:
        auc = float("nan")

    return {
        "strategy": strategy,
        "model": model_name,
        "threshold": threshold,
        "precision_defect": precision_score(y_true, pred, zero_division=0),
        "recall_defect": recall_score(y_true, pred, zero_division=0),
        "f1_defect": f1_score(y_true, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "roc_auc": auc,
        "false_alarm_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "missed_defect_count": int(fn),
        "false_alarm_count": int(fp),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def build_dataset_variants(x: pd.DataFrame) -> dict[str, pd.DataFrame]:
    missing_ratio = x.isna().mean()
    return {
        "median_all": x.copy(),
        "drop_high_missing": x.loc[:, missing_ratio <= 0.50].copy(),
    }


def build_models() -> dict[str, Pipeline | RandomForestClassifier]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3000,
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def extract_top_features(model: Pipeline, columns: list[str], strategy: str, model_name: str) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        score_name = "importance"
    elif hasattr(estimator, "coef_"):
        values = abs(estimator.coef_[0])
        score_name = "abs_coefficient"
    else:
        return pd.DataFrame()

    rows = [
        {
            "strategy": strategy,
            "model": model_name,
            "feature": feature,
            "score_type": score_name,
            "score": float(score),
            "interpretation_limit": "anonymous feature; treat as follow-up inspection candidate, not confirmed root cause",
        }
        for feature, score in zip(columns, values)
    ]
    return pd.DataFrame(rows).sort_values("score", ascending=False).head(20)


def write_data_profile(x: pd.DataFrame, y: pd.Series) -> None:
    missing_ratio = x.isna().mean()
    rows = [
        {"metric": "samples", "value": len(x)},
        {"metric": "features", "value": x.shape[1]},
        {"metric": "pass_count", "value": int((y == 0).sum())},
        {"metric": "defect_count", "value": int((y == 1).sum())},
        {"metric": "defect_rate", "value": float(y.mean())},
        {"metric": "features_with_missing", "value": int((missing_ratio > 0).sum())},
        {"metric": "max_feature_missing_ratio", "value": float(missing_ratio.max())},
        {"metric": "median_feature_missing_ratio", "value": float(missing_ratio.median())},
    ]
    pd.DataFrame(rows).to_csv(REPORT_DIR / "data_profile.csv", index=False)

    missing_ratio.sort_values(ascending=False).head(30).rename("missing_ratio").to_csv(
        REPORT_DIR / "top_missing_features.csv", index_label="feature"
    )


def choose_operating_point(threshold_df: pd.DataFrame) -> pd.Series:
    candidates = threshold_df[threshold_df["recall_defect"] >= 0.70].copy()
    if candidates.empty:
        candidates = threshold_df.copy()
    candidates["sort_key"] = candidates["recall_defect"] - candidates["false_alarm_rate"]
    return candidates.sort_values(["sort_key", "f1_defect"], ascending=False).iloc[0]


def build_error_cases(score_df: pd.DataFrame, operating_point: pd.Series) -> pd.DataFrame:
    threshold = float(operating_point["threshold"])
    case_df = score_df[
        (score_df["strategy"] == operating_point["strategy"])
        & (score_df["model"] == operating_point["model"])
    ].copy()
    case_df["threshold"] = threshold
    case_df["predicted_defect"] = (case_df["defect_score"] >= threshold).astype(int)
    case_df["error_type"] = "correct"
    case_df.loc[
        (case_df["true_defect"] == 0) & (case_df["predicted_defect"] == 1),
        "error_type",
    ] = "false_alarm"
    case_df.loc[
        (case_df["true_defect"] == 1) & (case_df["predicted_defect"] == 0),
        "error_type",
    ] = "missed_defect"
    case_df["inspection_note"] = ""
    case_df.loc[
        case_df["error_type"] == "false_alarm",
        "inspection_note",
    ] = "정상 sample이 defect로 분류된 case. 재검사 비용과 alarm fatigue 관점에서 점검한다."
    case_df.loc[
        case_df["error_type"] == "missed_defect",
        "inspection_note",
    ] = "defect sample을 놓친 case. 고객 품질 리스크와 추가 sensor/label 검증 관점에서 점검한다."

    errors = case_df[case_df["error_type"] != "correct"].copy()
    false_alarm = errors[errors["error_type"] == "false_alarm"].sort_values("defect_score", ascending=False).head(30)
    missed_defect = errors[errors["error_type"] == "missed_defect"].sort_values("defect_score", ascending=False).head(30)
    return pd.concat([false_alarm, missed_defect], ignore_index=True)


def write_threshold_plot(threshold_df: pd.DataFrame) -> None:
    best_model = threshold_df.groupby("model")["f1_defect"].max().sort_values(ascending=False).index[0]
    plot_df = threshold_df[threshold_df["model"] == best_model]

    plt.figure(figsize=(9, 5))
    for strategy, group in plot_df.groupby("strategy"):
        plt.plot(group["threshold"], group["recall_defect"], marker="o", label=f"{strategy} recall")
        plt.plot(group["threshold"], group["false_alarm_rate"], marker="x", linestyle="--", label=f"{strategy} false alarm")
    plt.xlabel("Decision threshold")
    plt.ylabel("Rate")
    plt.title(f"SECOM defect recall vs false alarm trade-off ({best_model})")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "threshold_tradeoff.png", dpi=160)
    plt.close()


def write_8d_report(profile: pd.DataFrame, metrics: pd.DataFrame, operating_point: pd.Series) -> None:
    defect_rate = float(profile.loc[profile["metric"] == "defect_rate", "value"].iloc[0])
    defect_count = int(float(profile.loc[profile["metric"] == "defect_count", "value"].iloc[0]))
    pass_count = int(float(profile.loc[profile["metric"] == "pass_count", "value"].iloc[0]))

    content = f"""# 8D Report: SECOM 공정 센서 pass/fail 분석

기준일: 2026-05-09

## D1. Team / Scope

- 목적: 공개 제조 공정 sensor data에서 pass/fail 예측 결과를 품질 문제 해결 문법으로 정리한다.
- 범위: UCI SECOM dataset. 실제 또는 특정 제조사의 현장 데이터가 아니다.
- 분석 역할: 데이터 결측/불균형 확인, baseline 모델 비교, threshold trade-off 분석, 원인 후보 제한 해석.

## D2. Problem Description

- 전체 sample: {pass_count + defect_count}
- pass sample: {pass_count}
- defect sample: {defect_count}
- defect rate: {defect_rate:.3%}
- 품질 관점 문제: defect class가 희소하므로 단순 accuracy는 높은데 defect를 놓치는 모델이 나올 수 있다.

## D3. Interim Containment

- 운영 threshold를 고정 0.50으로만 두지 않고 defect recall과 false alarm rate를 함께 보며 조정한다.
- defect score가 높은 sample은 재검사 또는 추가 sensor 확인 대상으로 분리한다.
- false alarm 비용이 큰 경우, threshold별 alarm volume을 별도 모니터링한다.

## D4. Root Cause Candidates

확정 원인이 아니라 후속 점검 후보로만 기록한다.

- 결측치가 많은 feature가 일부 존재해 imputation 전략에 따라 판단 경계가 달라질 수 있다.
- defect class 비중이 낮아 class imbalance가 recall 저하를 만든다.
- feature 이름이 익명화되어 있어 상위 중요 feature를 실제 설비·공정 원인으로 단정할 수 없다.
- label noise 또는 공정 조건 drift 가능성은 원본 데이터만으로 확인할 수 없다.

## D5. Corrective Actions Selected

- 결측 처리 2개 전략 비교: 모든 feature 유지 후 median imputation, 결측률 50% 초과 feature 제거 후 median imputation.
- baseline 2개 비교: logistic regression, random forest.
- threshold sweep으로 missed defect와 false alarm trade-off를 표로 남긴다.
- feature importance는 원인 확정이 아니라 추가 점검 우선순위 후보로만 사용한다.

## D6. Validation

선택 operating point:

- strategy: {operating_point['strategy']}
- model: {operating_point['model']}
- threshold: {operating_point['threshold']:.2f}
- defect recall: {operating_point['recall_defect']:.3f}
- false alarm rate: {operating_point['false_alarm_rate']:.3f}
- missed defect count: {int(operating_point['missed_defect_count'])}
- false alarm count: {int(operating_point['false_alarm_count'])}

전체 결과는 `reports/metrics.csv`, threshold별 결과는 `reports/threshold_tradeoff.csv`를 기준으로 확인한다.

## D7. Prevent Recurrence

- 운영 시 accuracy가 아니라 defect recall, false alarm rate, missed defect count를 함께 monitoring한다.
- feature별 결측률 drift를 정기 점검한다.
- 고위험 score sample과 missed defect case를 failure gallery로 축적한다.
- 실제 현장 데이터에서는 feature 이름, 설비 조건, 공정 step, 작업 이력과 연결해 원인 검증을 별도로 수행한다.

## D8. Closure / Note

이 산출물은 실제 현장 개선 완료 사례가 아니라 공개 제조 데이터를 활용한 분석 프로젝트다. 제조 데이터 품질 판단에서 미검출과 오경보를 분리해 보고, 모델 결과를 8D report 형식으로 정리한 것이다.
"""
    (REPORT_DIR / "8d_report.md").write_text(content, encoding="utf-8")


def write_notebook_placeholders() -> None:
    notebook_dir = ROOT / "notebooks"
    notebook_dir.mkdir(parents=True, exist_ok=True)
    (notebook_dir / "README.md").write_text(
        """# Notebooks

The reproducible source of truth is `scripts/run_analysis.py`.

Suggested notebook split:

- `01_eda_secom.ipynb`: load raw files, inspect label imbalance, missing ratios, and feature distributions.
- `02_baseline_models.ipynb`: compare imputation strategies, logistic regression, random forest, threshold sweep, and error cases.

The reports generated by the script are the portfolio artifacts to attach first.
""",
        encoding="utf-8",
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    x, y = load_data()
    write_data_profile(x, y)

    metrics: list[dict[str, float | int | str]] = []
    threshold_rows: list[dict[str, float | int | str]] = []
    top_feature_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []

    for strategy, x_variant in build_dataset_variants(x).items():
        x_train, x_test, y_train, y_test = train_test_split(
            x_variant,
            y,
            test_size=0.30,
            stratify=y,
            random_state=42,
        )
        for model_name, model in build_models().items():
            model.fit(x_train, y_train)
            proba = pd.Series(model.predict_proba(x_test)[:, 1], index=x_test.index, name="defect_score")
            score_frames.append(
                pd.DataFrame(
                    {
                        "sample_index": x_test.index,
                        "strategy": strategy,
                        "model": model_name,
                        "true_defect": y_test.values,
                        "defect_score": proba.values,
                    }
                ).sort_values("defect_score", ascending=False)
            )

            metrics.append(metric_row(strategy, model_name, y_test, proba, 0.50))
            for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]:
                threshold_rows.append(metric_row(strategy, model_name, y_test, proba, threshold))

            top_feature_frames.append(extract_top_features(model, list(x_variant.columns), strategy, model_name))

    profile_df = pd.read_csv(REPORT_DIR / "data_profile.csv")
    metric_df = pd.DataFrame(metrics)
    threshold_df = pd.DataFrame(threshold_rows)
    top_feature_df = pd.concat(top_feature_frames, ignore_index=True)
    score_df = pd.concat(score_frames, ignore_index=True)

    metric_df.to_csv(REPORT_DIR / "metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    threshold_df.to_csv(REPORT_DIR / "threshold_tradeoff.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    top_feature_df.to_csv(REPORT_DIR / "top_features.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    score_df.to_csv(REPORT_DIR / "sample_scores.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    write_threshold_plot(threshold_df)
    operating_point = choose_operating_point(threshold_df)
    error_df = build_error_cases(score_df, operating_point)
    error_df.to_csv(REPORT_DIR / "error_cases.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    write_8d_report(profile_df, metric_df, operating_point)
    write_notebook_placeholders()

    print("Wrote reports:")
    for path in sorted(REPORT_DIR.glob("*")):
        if path.is_file():
            print(f"- {path.relative_to(ROOT)}")
    print(f"- {FIGURE_DIR.relative_to(ROOT)}/threshold_tradeoff.png")


if __name__ == "__main__":
    main()
