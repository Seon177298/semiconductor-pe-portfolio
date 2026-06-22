# secom 통계적 엄밀화 (nested CV + GBM + DeLong)

> 공개 UCI SECOM. feature 익명화 → 실제 공정/설비 원인 단정 금지. 재현: `python scripts/rigor.py`.

## 1. operating-point 누수 제거 (단일 split → repeated nested CV)

기존 주요 수치(median_all·RF·threshold 0.10, 단일 70/30 split)은 **threshold 를 test set 에서 골라** 낙관 편향이 있었다.
여기서는 threshold 를 각 outer-train 의 validation 에서만 고르고(누수 제거), 15 outer fold 의 test 에서 평가한다.

| model | recall (95% CI) | false-alarm (95% CI) | AUC (95% CI) | missed(평균) | median thr |
|---|---|---|---|---|---|
| logistic_regression | 0.471 [0.3024, 0.65] | 0.2775 [0.2423, 0.3234] | 0.6667 [0.5541, 0.7362] | 11.0 | 0.05 |
| random_forest | 0.8814 [0.5571, 1.0] | 0.6798 [0.2995, 0.8532] | 0.7442 [0.6712, 0.8355] | 2.47 | 0.05 |
| hist_gbm | 0.5133 [0.35, 0.6392] | 0.234 [0.204, 0.2738] | 0.7045 [0.6348, 0.7736] | 10.13 | 0.05 |


**apples-to-apples (고정 threshold 0.10, RF) — 주요 수치 직접 검증:**
- 단일 split 주요 수치: recall **0.774** / false-alarm **0.248** / missed **7**.
- nested-CV(@0.10, 누수 없음): recall **0.6667 [0.5238, 0.7929]**,
  false-alarm **0.3003 [0.249, 0.3448]**.
- → 주요 수치 recall 0.774 가 nested-CV @0.10 CI 에 포함됨 → 단일 split 값이 대표적; FA 0.248 은 미포함.

**자동 operating-point 선택의 불안정성:** validation 에서 recall≥0.70 floor 로 고르면 threshold 가 median 0.05 로 낮아지고
recall **0.8814 [0.5571, 1.0]** / FA **0.6798 [0.2995, 0.8532]** 로 산포가 크다(소수 defect → 작은 fold 에서 OP 불안정).
→ **operating point 는 고정 상수가 아니라 비용·표본에 민감**하므로 CI 와 함께 보고하는 것이 정직하다.

## 2. GBM baseline 비교

GBM(HistGradientBoosting, native NaN)을 RF/LogReg 와 같은 nested-CV 로 비교(위 표). AUC·recall 기준 상대 위치로 모델 선택 근거 제시.

## 3. AUC 유의성 (held-out, paired DeLong + bootstrap)

| 비교 | AUC a | AUC b | DeLong p |
|---|---|---|---|
| random_forest vs logistic_regression | 0.813 | 0.7204 | 0.1209 |
| random_forest vs hist_gbm | 0.813 | 0.7411 | 0.0283 |
| hist_gbm vs logistic_regression | 0.7411 | 0.7204 | 0.7251 |


- per-model bootstrap AUC 95% CI: `reports/rigor_auc_bootstrap.csv`.
- DeLong p 가 0.05 미만이면 두 모델 AUC 차이가 통계적으로 유의(같은 test 표본의 상관 AUC 비교).

## 금지선

- CI/p 는 **공개 데이터의 통계 변동**이며 실측 성능이 아니다. SECOM feature 익명화로 공정 원인 단정 금지.
- (figure: `figures/rigor_nested_cv.png`, raw: `rigor_nested_cv.csv`/`rigor_delong.csv`/`rigor_auc_bootstrap.csv`.)
