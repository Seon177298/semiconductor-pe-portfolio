"""
Escape / Overkill 비용 환산 레이어 (cost-based operating point selection).

목적 (Product Engineering 관점):
- escape(미검, 불량 출하)와 overkill(과검, 양품 폐기)을 '통계 지표'가 아니라 '비용'으로 환산한다.
- 네 가지 비용 요소를 명시적 가정으로 둔다:
    (1) DPPM  : 출하 후 field quality 패널티 (escape bit DPPM 기준)
    (2) 폐기비용(scrap / yield loss): 양품을 과검으로 버리는 비용
    (3) 재테스트 TAT : fail 판정된 die/unit 의 추가 test insertion 비용
    (4) yield loss   : 과검으로 잃은 양품 비율 (= 폐기비용에 반영)
- operating point(FBM=guardband, SECOM=threshold)를 '통계 최적'이 아니라 '비용 최소점'으로 재선정한다.
- 비용 비대칭(escape >> overkill)에 따라 최적점이 어떻게 이동하는지 sensitivity 로 보여준다.

비용 단위는 임의의 상대 단위(USD-equivalent)이며 **시연용 가정값**이다.
   FBM 데이터는 합성, SECOM 은 공개(UCI) 익명 데이터다. 실제 실제 원가가 아니다.

입력:
- reports/fbm_gb_sweep.csv                 (synth_fail_bit_map.py 산출)
- reports/secom_threshold_tradeoff_frozen.csv (secom_quality_8d run_analysis.py 에서 frozen)

산출:
- reports/fbm_cost_curve.csv
- reports/secom_cost_curve.csv
- reports/cost_sensitivity.csv
- reports/cost_model_summary.md
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports"

# --------------------------------------------------------------------------------------
# Cost assumptions (illustrative, documented). Relative USD-equivalent units.
# --------------------------------------------------------------------------------------
# FBM (memory die) — die-level events + bit-level DPPM
FBM_COST_ESCAPE_DIE = 80.0    # shipping a true-bad die (field return / customer line-down risk)
FBM_COST_SCRAP_DIE = 6.0      # good die wrongly scrapped (lost die value = yield loss)
FBM_COST_RETEST_DIE = 1.5     # extra test insertion for each die flagged to repair/scrap (TAT)
FBM_COST_PER_DPPM = 0.5       # field quality penalty per outgoing escape DPPM

# SECOM (per production unit) — escape = missed defect, overkill = false alarm
SECOM_COST_ESCAPE_UNIT = 50.0   # a defective unit shipped to customer
SECOM_COST_OVERKILL_UNIT = 3.0  # a good unit false-alarmed -> retest / inspection (TAT)

# Cost ratios for the sensitivity sweep (escape : overkill)
RATIO_GRID = [(3, 1), (10, 1), (30, 1), (1, 1), (1, 3), (1, 10)]


# --------------------------------------------------------------------------------------
# FBM cost layer
# --------------------------------------------------------------------------------------
def fbm_cost():
    sw = pd.read_csv(REPORT / "fbm_gb_sweep.csv")
    sw = sw.copy()
    sw["cost_escape_field"] = sw["escape_dies"] * FBM_COST_ESCAPE_DIE
    sw["cost_dppm_quality"] = sw["escape_dppm"] * FBM_COST_PER_DPPM
    sw["cost_scrap_yield"] = sw["overkill_dies"] * FBM_COST_SCRAP_DIE
    sw["cost_retest_tat"] = sw["scrapped_dies"] * FBM_COST_RETEST_DIE
    sw["total_cost"] = (sw["cost_escape_field"] + sw["cost_dppm_quality"]
                        + sw["cost_scrap_yield"] + sw["cost_retest_tat"])
    # statistical optimum: minimise misclassified dies (escape + overkill), equal weight
    sw["misclassified_dies"] = sw["escape_dies"] + sw["overkill_dies"]
    stat_gb = sw.loc[sw["misclassified_dies"].idxmin(), "guardband"]
    cost_gb = sw.loc[sw["total_cost"].idxmin(), "guardband"]
    sw.to_csv(REPORT / "fbm_cost_curve.csv", index=False)
    return sw, float(stat_gb), float(cost_gb)


def fbm_sensitivity(sw_base: pd.DataFrame):
    """cost-optimal guardband under different escape:overkill cost ratios."""
    rows = []
    for e, o in RATIO_GRID:
        # die-level only, normalised: escape weight e, overkill weight o
        cost = sw_base["escape_dies"] * e + sw_base["overkill_dies"] * o
        gb = sw_base.loc[cost.idxmin(), "guardband"]
        rows.append({"domain": "FBM", "escape_weight": e, "overkill_weight": o,
                     "cost_opt_operating_point": gb})
    return rows


# --------------------------------------------------------------------------------------
# SECOM cost layer
# --------------------------------------------------------------------------------------
def secom_cost():
    df = pd.read_csv(REPORT / "secom_threshold_tradeoff_frozen.csv")
    # canonical model/strategy used as the single source of truth
    sub = df[(df["strategy"] == "median_all") & (df["model"] == "random_forest")].copy()
    sub = sub.sort_values("threshold")
    sub["escape_units"] = sub["missed_defect_count"]      # fn
    sub["overkill_units"] = sub["false_alarm_count"]      # fp
    sub["cost_escape"] = sub["escape_units"] * SECOM_COST_ESCAPE_UNIT
    sub["cost_overkill_tat"] = sub["overkill_units"] * SECOM_COST_OVERKILL_UNIT
    sub["total_cost"] = sub["cost_escape"] + sub["cost_overkill_tat"]
    cost_thr = sub.loc[sub["total_cost"].idxmin(), "threshold"]
    sub.to_csv(REPORT / "secom_cost_curve.csv", index=False)
    # statistical pick = original run_analysis policy: recall>=0.70 then min false alarm
    cand = sub[sub["recall_defect"] >= 0.70]
    stat_thr = cand.sort_values(["false_alarm_rate", "threshold"]).iloc[0]["threshold"] if len(cand) else 0.10
    return sub, float(stat_thr), float(cost_thr)


def secom_sensitivity(sub: pd.DataFrame):
    rows = []
    for e, o in RATIO_GRID:
        cost = sub["escape_units"] * e + sub["overkill_units"] * o
        thr = sub.loc[cost.idxmin(), "threshold"]
        rows.append({"domain": "SECOM", "escape_weight": e, "overkill_weight": o,
                     "cost_opt_operating_point": thr})
    return rows


def row_at(df, col, val):
    return df[df[col] == val].iloc[0]


def main():
    sw, fbm_stat_gb, fbm_cost_gb = fbm_cost()
    sub, secom_stat_thr, secom_cost_thr = secom_cost()

    sens = fbm_sensitivity(sw) + secom_sensitivity(sub)
    pd.DataFrame(sens).to_csv(REPORT / "cost_sensitivity.csv", index=False)

    f_stat = row_at(sw, "guardband", fbm_stat_gb)
    f_cost = row_at(sw, "guardband", fbm_cost_gb)
    s_stat = row_at(sub, "threshold", secom_stat_thr)
    s_cost = row_at(sub, "threshold", secom_cost_thr)

    def fbm_line(r):
        return (f"gb={r['guardband']:.2f} | escape {int(r['escape_dies'])} dies / "
                f"{r['escape_dppm']:.1f} DPPM, overkill {int(r['overkill_dies'])} dies, "
                f"yield {r['yield']:.3f}, total cost **{r['total_cost']:.1f}**")

    def secom_line(r):
        return (f"threshold={r['threshold']:.2f} | recall {r['recall_defect']:.3f}, "
                f"missed(escape) {int(r['escape_units'])}, false-alarm(overkill) {int(r['overkill_units'])}, "
                f"total cost **{r['total_cost']:.1f}**")

    sens_df = pd.DataFrame(sens)
    fbm_sens_tbl = "\n".join(
        f"| {r.escape_weight}:{r.overkill_weight} | {r.cost_opt_operating_point:.2f} |"
        for r in sens_df[sens_df.domain == "FBM"].itertuples())
    secom_sens_tbl = "\n".join(
        f"| {r.escape_weight}:{r.overkill_weight} | {r.cost_opt_operating_point:.2f} |"
        for r in sens_df[sens_df.domain == "SECOM"].itertuples())

    md = f"""# Escape/Overkill 비용 환산 — operating point 재선정

> 비용 단위는 임의 상대 단위(USD-equivalent), **시연용 가정값**이다.
> FBM=합성 데이터, SECOM=공개(UCI) 익명 데이터. 실제 실제 원가가 아니다.

## 비용 가정 (명시)

**FBM (memory die)**
- escape(불량 출하) field 비용 = {FBM_COST_ESCAPE_DIE}/die
- 폐기/yield-loss(양품 과검 폐기) = {FBM_COST_SCRAP_DIE}/die
- 재테스트 TAT(fail 판정 die 추가 test insertion) = {FBM_COST_RETEST_DIE}/die
- DPPM field-quality 패널티 = {FBM_COST_PER_DPPM}/DPPM (escape bit DPPM 기준)

**SECOM (production unit)**
- escape(불량 출하) = {SECOM_COST_ESCAPE_UNIT}/unit
- overkill(오경보→재검사 TAT) = {SECOM_COST_OVERKILL_UNIT}/unit

## 1. FBM — 통계 최적 vs 비용 최소

- **통계 최적**(escape+overkill die 최소): {fbm_line(f_stat)}
- **비용 최소**: {fbm_line(f_cost)}

escape die 의 field 비용({FBM_COST_ESCAPE_DIE})이 과검 폐기({FBM_COST_SCRAP_DIE})·재테스트({FBM_COST_RETEST_DIE})보다 크기 때문에,
비용 최소점은 단순 오분류 최소점과 다를 수 있다(불량 미검을 더 줄이는 쪽으로 이동).
전 구간: `reports/fbm_cost_curve.csv`.

### FBM cost-optimal guardband sensitivity (escape:overkill 비용비)

| escape:overkill | cost-optimal guardband |
|---|---|
{fbm_sens_tbl}

## 2. SECOM — 통계 최적 vs 비용 최소 (median_all / random_forest)

- **통계 최적**(recall≥0.70 후 false alarm 최소 = 원래 운영점): {secom_line(s_stat)}
- **비용 최소**: {secom_line(s_cost)}

전 구간: `reports/secom_cost_curve.csv`.

### SECOM cost-optimal threshold sensitivity (escape:overkill 비용비)

| escape:overkill | cost-optimal threshold |
|---|---|
{secom_sens_tbl}

## 핵심 메시지

- operating point(guardband/threshold)는 **통계 지표가 아니라 비용 비대칭으로 결정**된다.
- escape 가 비쌀수록 미검을 줄이는 쪽(낮은 threshold / 높은 guardband)으로,
  overkill 이 비쌀수록 과검을 줄이는 쪽으로 최적점이 이동한다 (위 sensitivity 표).
- 따라서 PE 의 test/guardband 정책은 "정확도 최대화"가 아니라
  "escape·overkill·재테스트 TAT·yield-loss 의 총비용 최소화"로 설명해야 한다.
"""
    (REPORT / "cost_model_summary.md").write_text(md, encoding="utf-8")

    print("FBM   statistical-opt guardband:", fbm_stat_gb, "| cost-min guardband:", fbm_cost_gb)
    print("SECOM statistical-opt threshold:", secom_stat_thr, "| cost-min threshold:", secom_cost_thr)
    print("FBM cost-min line :", fbm_line(f_cost))
    print("SECOM cost-min line:", secom_line(s_cost))


if __name__ == "__main__":
    main()
