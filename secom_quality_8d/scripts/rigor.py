"""rigor.py — secom 통계적 엄밀화.

단일 70/30 split + test-set 에서 operating point 를 고르던 방식의 **낙관 편향(operating-point
leakage)** 을 제거하고, 모델 비교에 유의성을 붙인다:

  1) repeated nested stratified CV
       - outer: RepeatedStratifiedKFold (재현적 일반화 추정)
       - operating point(threshold)는 각 outer-train 안의 validation 에서만 선택 → test 누수 제거
       - outer-test 에서 그 threshold 로 recall/false-alarm/missed + AUC → 비편향 추정 + 95% CI
  2) GBM(HistGradientBoosting) baseline 추가 (RF/LogReg 와 비교)
  3) DeLong 검정(paired AUC)으로 모델 간 AUC 차이 유의성 + per-model bootstrap AUC 95% CI

공개 UCI SECOM 데이터. feature 익명화 — 실제 공정/설비 원인으로 단정하지 않는다.
재현: `python scripts/rigor.py` (data/raw 없으면 run_analysis 와 동일 경로로 다운로드).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from run_analysis import build_models, load_data

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

SEED = 42
THRESHOLD_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]
RECALL_FLOOR = 0.70
N_SPLITS = 5
N_REPEATS = 3
N_BOOT = 2000


# ---------------------------------------------------------------------------
# leakage-free operating-point selection (pure)
# ---------------------------------------------------------------------------
def _recall_fa(y_true, pred):
    y_true = np.asarray(y_true)
    pred = np.asarray(pred)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fa = fp / (fp + tn) if (fp + tn) else 0.0
    return recall, fa, fn, fp


def select_threshold(y_true, proba, grid=THRESHOLD_GRID, recall_floor=RECALL_FLOOR):
    """Pick the threshold maximizing (recall - false_alarm) among those with
    recall >= floor; fall back to max-recall if the floor is unreachable.
    Chosen on validation data only — never on the test fold."""
    proba = np.asarray(proba)
    best = None
    fallback = None
    for thr in grid:
        recall, fa, _, _ = _recall_fa(y_true, (proba >= thr).astype(int))
        score = recall - fa
        if fallback is None or recall > fallback[1]:
            fallback = (thr, recall)
        if recall >= recall_floor and (best is None or score > best[1]):
            best = (thr, score)
    return float(best[0]) if best is not None else float(fallback[0])


def op_metrics(y_true, proba, threshold):
    pred = (np.asarray(proba) >= threshold).astype(int)
    recall, fa, fn, fp = _recall_fa(y_true, pred)
    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:
        auc = float("nan")
    return {"threshold": threshold, "recall": recall, "false_alarm": fa,
            "missed": int(fn), "false_alarm_count": int(fp), "auc": auc}


# ---------------------------------------------------------------------------
# DeLong paired-AUC test (Sun & Xu fast algorithm)
# ---------------------------------------------------------------------------
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, proba_a, proba_b):
    """Two-sided p-value for H0: AUC_a == AUC_b on the same samples. Returns (auc_a, auc_b, p)."""
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true)               # positives (label 1) first
    label_1_count = int((y_true == 1).sum())
    preds = np.vstack((np.asarray(proba_a)[order], np.asarray(proba_b)[order]))
    aucs, cov = _fast_delong(preds, label_1_count)
    el = np.array([[1.0, -1.0]])
    var = float(np.atleast_2d(el @ cov @ el.T)[0, 0])
    z = abs(aucs[0] - aucs[1]) / (np.sqrt(var) + 1e-300)
    p = 2 * (1 - stats.norm.cdf(z))
    return float(aucs[0]), float(aucs[1]), float(p)


def bootstrap_auc_ci(y_true, proba, n_boot=N_BOOT, seed=0):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], proba[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ---------------------------------------------------------------------------
# models (RF + LogReg from run_analysis, + GBM baseline)
# ---------------------------------------------------------------------------
def build_models_with_gbm():
    models = build_models()
    models["hist_gbm"] = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.06, max_iter=300, l2_regularization=1.0,
        class_weight="balanced", random_state=SEED)   # native NaN handling
    return models


# ---------------------------------------------------------------------------
# nested CV with leakage-free operating-point selection
# ---------------------------------------------------------------------------
def nested_cv(x, y, model, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED, fixed_threshold=0.10):
    """Repeated nested CV. Per fold: select the operating point on a TRAIN-only
    validation split (no test leakage), evaluate on the test fold. Also evaluate
    at a FIXED threshold (default 0.10) on the same fold for an apples-to-apples
    comparison with the single-split headline."""
    outer = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = []
    for tr, te in outer.split(x, y):
        xtr, xte = x.iloc[tr], x.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        # inner validation split of TRAIN only -> select threshold (no test leakage)
        xfit, xval, yfit, yval = train_test_split(
            xtr, ytr, test_size=0.25, stratify=ytr, random_state=seed)
        inner = clone(model).fit(xfit, yfit)
        thr = select_threshold(yval, inner.predict_proba(xval)[:, 1])
        # refit on full outer-train, evaluate at both the selected and fixed thresholds
        proba_te = clone(model).fit(xtr, ytr).predict_proba(xte)[:, 1]
        rec = op_metrics(yte, proba_te, thr)
        fx = op_metrics(yte, proba_te, fixed_threshold)
        rec["recall_fixed"] = fx["recall"]
        rec["false_alarm_fixed"] = fx["false_alarm"]
        rec["missed_fixed"] = fx["missed"]
        rows.append(rec)
    return pd.DataFrame(rows)


def _ci(s):
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def build():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    x, y = load_data()   # median_all == full feature set (canonical strategy)

    # ---- 1+2) nested CV per model (incl. GBM) ----
    models = build_models_with_gbm()
    cv_rows = []
    for name, model in models.items():
        df = nested_cv(x, y, model)
        rl, rh = _ci(df["recall"])
        fl, fh = _ci(df["false_alarm"])
        al, ah = _ci(df["auc"])
        rfl, rfh = _ci(df["recall_fixed"])
        ffl, ffh = _ci(df["false_alarm_fixed"])
        cv_rows.append({
            "model": name,
            "recall_mean": round(df["recall"].mean(), 4),
            "recall_ci_lo": round(rl, 4), "recall_ci_hi": round(rh, 4),
            "false_alarm_mean": round(df["false_alarm"].mean(), 4),
            "false_alarm_ci_lo": round(fl, 4), "false_alarm_ci_hi": round(fh, 4),
            "auc_mean": round(df["auc"].mean(), 4),
            "auc_ci_lo": round(al, 4), "auc_ci_hi": round(ah, 4),
            "missed_mean": round(df["missed"].mean(), 2),
            "median_threshold": float(np.median(df["threshold"])),
            "recall_at_0p10_mean": round(df["recall_fixed"].mean(), 4),
            "recall_at_0p10_ci_lo": round(rfl, 4), "recall_at_0p10_ci_hi": round(rfh, 4),
            "false_alarm_at_0p10_mean": round(df["false_alarm_fixed"].mean(), 4),
            "false_alarm_at_0p10_ci_lo": round(ffl, 4), "false_alarm_at_0p10_ci_hi": round(ffh, 4),
            "n_folds": len(df),
        })
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(REPORT_DIR / "rigor_nested_cv.csv", index=False)

    # ---- 3) significance on a single canonical held-out split (paired preds) ----
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.30, stratify=y, random_state=SEED)
    probas = {}
    sig_rows = []
    for name, model in build_models_with_gbm().items():
        m = clone(model).fit(xtr, ytr)
        p = m.predict_proba(xte)[:, 1]
        probas[name] = p
        lo, hi = bootstrap_auc_ci(yte, p, seed=SEED)
        sig_rows.append({"model": name, "auc": round(roc_auc_score(yte, p), 4),
                         "auc_boot_ci_lo": round(lo, 4), "auc_boot_ci_hi": round(hi, 4)})
    sig = pd.DataFrame(sig_rows)

    pairs = [("random_forest", "logistic_regression"),
             ("random_forest", "hist_gbm"),
             ("hist_gbm", "logistic_regression")]
    delong_rows = []
    for a, b in pairs:
        auc_a, auc_b, pval = delong_roc_test(yte, probas[a], probas[b])
        delong_rows.append({"model_a": a, "model_b": b, "auc_a": round(auc_a, 4),
                            "auc_b": round(auc_b, 4), "delong_p": round(pval, 4)})
    delong = pd.DataFrame(delong_rows)
    sig.to_csv(REPORT_DIR / "rigor_auc_bootstrap.csv", index=False)
    delong.to_csv(REPORT_DIR / "rigor_delong.csv", index=False)

    save_figure(cv, sig)
    write_summary(cv, sig, delong)

    print("=== rigor (nested CV + GBM + DeLong) done ===")
    print(cv[["model", "recall_mean", "recall_ci_lo", "recall_ci_hi",
              "false_alarm_mean", "auc_mean", "median_threshold"]].to_string(index=False))
    print("\nDeLong paired AUC:\n", delong.to_string(index=False))


def save_figure(cv, sig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    yloc = np.arange(len(cv))
    ax.errorbar(cv["recall_mean"], yloc - 0.12,
                xerr=[cv["recall_mean"] - cv["recall_ci_lo"], cv["recall_ci_hi"] - cv["recall_mean"]],
                fmt="o", color="#1e88e5", capsize=5, label="defect recall")
    ax.errorbar(cv["false_alarm_mean"], yloc + 0.12,
                xerr=[cv["false_alarm_mean"] - cv["false_alarm_ci_lo"], cv["false_alarm_ci_hi"] - cv["false_alarm_mean"]],
                fmt="s", color="#e53935", capsize=5, label="false-alarm rate")
    ax.axvline(0.774, color="#1e88e5", ls=":", alpha=0.6, label="single-split recall 0.774")
    ax.axvline(0.248, color="#e53935", ls=":", alpha=0.6, label="single-split FA 0.248")
    ax.set_yticks(yloc)
    ax.set_yticklabels(cv["model"])
    ax.set_xlabel("rate")
    ax.set_title("Nested-CV operating point (95% CI) vs single-split headline")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)

    yloc2 = np.arange(len(sig))
    ax2.errorbar(sig["auc"], yloc2,
                 xerr=[sig["auc"] - sig["auc_boot_ci_lo"], sig["auc_boot_ci_hi"] - sig["auc"]],
                 fmt="o", color="#43a047", capsize=5)
    ax2.set_yticks(yloc2)
    ax2.set_yticklabels(sig["model"])
    ax2.set_xlabel("ROC-AUC (held-out, bootstrap 95% CI)")
    ax2.set_title("Model AUC with bootstrap CI")
    ax2.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "rigor_nested_cv.png", dpi=150)
    plt.close(fig)


def write_summary(cv, sig, delong):
    rf = cv[cv["model"] == "random_forest"].iloc[0]
    cv_tbl = "| model | recall (95% CI) | false-alarm (95% CI) | AUC (95% CI) | missed(평균) | median thr |\n"
    cv_tbl += "|---|---|---|---|---|---|\n"
    for r in cv.itertuples():
        cv_tbl += (f"| {r.model} | {r.recall_mean} [{r.recall_ci_lo}, {r.recall_ci_hi}] | "
                   f"{r.false_alarm_mean} [{r.false_alarm_ci_lo}, {r.false_alarm_ci_hi}] | "
                   f"{r.auc_mean} [{r.auc_ci_lo}, {r.auc_ci_hi}] | {r.missed_mean} | {r.median_threshold} |\n")
    dl_tbl = "| 비교 | AUC a | AUC b | DeLong p |\n|---|---|---|---|\n"
    for r in delong.itertuples():
        dl_tbl += f"| {r.model_a} vs {r.model_b} | {r.auc_a} | {r.auc_b} | {r.delong_p} |\n"

    md = f"""# secom 통계적 엄밀화 (nested CV + GBM + DeLong)

> 공개 UCI SECOM. feature 익명화 → 실제 공정/설비 원인 단정 금지. 재현: `python scripts/rigor.py`.

## 1. operating-point 누수 제거 (단일 split → repeated nested CV)

기존 주요 수치(median_all·RF·threshold 0.10, 단일 70/30 split)은 **threshold 를 test set 에서 골라** 낙관 편향이 있었다.
여기서는 threshold 를 각 outer-train 의 validation 에서만 고르고(누수 제거), {int(rf.n_folds)} outer fold 의 test 에서 평가한다.

{cv_tbl}

**apples-to-apples (고정 threshold 0.10, RF) — 주요 수치 직접 검증:**
- 단일 split 주요 수치: recall **0.774** / false-alarm **0.248** / missed **7**.
- nested-CV(@0.10, 누수 없음): recall **{rf.recall_at_0p10_mean} [{rf.recall_at_0p10_ci_lo}, {rf.recall_at_0p10_ci_hi}]**,
  false-alarm **{rf.false_alarm_at_0p10_mean} [{rf.false_alarm_at_0p10_ci_lo}, {rf.false_alarm_at_0p10_ci_hi}]**.
- → 주요 수치 recall 0.774 가 nested-CV @0.10 CI 에 {'포함됨 → 단일 split 값이 대표적' if rf.recall_at_0p10_ci_lo <= 0.774 <= rf.recall_at_0p10_ci_hi else '미포함 → 단일 split 이 낙관적'}; FA 0.248 은 {'포함됨' if rf.false_alarm_at_0p10_ci_lo <= 0.248 <= rf.false_alarm_at_0p10_ci_hi else '미포함'}.

**자동 operating-point 선택의 불안정성:** validation 에서 recall≥0.70 floor 로 고르면 threshold 가 median {rf.median_threshold} 로 낮아지고
recall **{rf.recall_mean} [{rf.recall_ci_lo}, {rf.recall_ci_hi}]** / FA **{rf.false_alarm_mean} [{rf.false_alarm_ci_lo}, {rf.false_alarm_ci_hi}]** 로 산포가 크다(소수 defect → 작은 fold 에서 OP 불안정).
→ **operating point 는 고정 상수가 아니라 비용·표본에 민감**하므로 CI 와 함께 보고하는 것이 정직하다.

## 2. GBM baseline 비교

GBM(HistGradientBoosting, native NaN)을 RF/LogReg 와 같은 nested-CV 로 비교(위 표). AUC·recall 기준 상대 위치로 모델 선택 근거 제시.

## 3. AUC 유의성 (held-out, paired DeLong + bootstrap)

{dl_tbl}

- per-model bootstrap AUC 95% CI: `reports/rigor_auc_bootstrap.csv`.
- DeLong p 가 0.05 미만이면 두 모델 AUC 차이가 통계적으로 유의(같은 test 표본의 상관 AUC 비교).

## 금지선

- CI/p 는 **공개 데이터의 통계 변동**이며 실측 성능이 아니다. SECOM feature 익명화로 공정 원인 단정 금지.
- (figure: `figures/rigor_nested_cv.png`, raw: `rigor_nested_cv.csv`/`rigor_delong.csv`/`rigor_auc_bootstrap.csv`.)
"""
    (REPORT_DIR / "rigor_summary.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    build()
