# SECOM Canonical Operating Point (단일 출처)

> 공개 데이터: UCI SECOM (익명 반도체 공정 sensor pass/fail). 실제 실측 데이터가 아니다.
> 재현: `secom_quality_8d/scripts/run_analysis.py` (sklearn 1.6.1, random_state=42, test_size=0.30, stratified).

이 저장소가 인용하는 **단 하나의 SECOM 운영점**:

| 항목 | 값 |
|---|---|
| strategy | median_all (전체 feature 유지 + median imputation) |
| model | random_forest |
| threshold | 0.10 |
| defect recall (미검 반대) | **0.774** |
| false alarm rate (과검) | **0.248** |
| missed defect count (escape) | **7** |
| false alarm count (overkill) | 109 |
| confusion (tp/fp/fn/tn) | 24 / 109 / 7 / 331 |
| test set | 471 samples (defect 31) |

선택 규칙(통계): defect recall ≥ 0.70 인 후보 중 false alarm rate 최소 → median_all/RF/0.10.
이 운영점에 escape/overkill **비용 환산**을 얹은 결과는 `cost_model_summary.md` 참조
(비용 비대칭에 따라 cost-optimal threshold 가 0.10↔0.30 으로 이동).

**제외:** 다른 split(drop_high_missing)·별도 탐색 run 의 0.839 / 0.266 / missed 5 값은
설정이 다르므로 이 주요 수치에서 함께 쓰지 않는다.

frozen 입력: `secom_metrics_frozen.csv`, `secom_threshold_tradeoff_frozen.csv`
(위 run_analysis.py 산출물을 그대로 복사, 본 폴더 비용 레이어의 입력으로 고정).
