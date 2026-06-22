# E2 — operating point·escape/overkill 의 불확실성 (95% CI)

> **합성 데이터.** single-seed point estimate 가 "운 좋은 한 점"이 아님을 보이기 위해
> (1) 다중 seed(20개, population variability) + (2) bootstrap(2000 resamples, sampling)
> 두 불확실성을 분리해 95% CI 를 붙였다.

## 1. cost-optimal guardband 의 안정성 (다중 seed)

20개 seed 의 cost-opt guardband 분포: **0.05V×20**.
→ operating point 가 한 seed 의 우연이 아니라 **재현적으로 같은 영역**에 형성됨을 보인다.

## 2. 주요 지표의 다중 seed 95% CI

| 지표 | median | 95% CI (다중 seed) |
|---|---|---|
| escape_dppm @ cost-opt | 82.195 | [73.091, 85.923] |
| overkill_dppm @ cost-opt | 459.39 | [454.645, 463.632] |
| escape_dies @ gb0 | 74.0 | [61.475, 83.575] |
| cost reduction % | 76.873 | [73.686, 79.412] |
| cost-opt guardband (V) | 0.05 | [0.05, 0.05] |

## 3. cost-opt 에서의 bootstrap 95% CI (canonical seed=7, 2000 resamples)

| 지표 | point | bootstrap 95% CI |
|---|---|---|
| escape DPPM @cost-opt | 89.6 | [81.0, 98.2] |
| overkill DPPM @cost-opt | 461.6 | [451.2, 471.9] |
| escape dies @cost-opt | 0 | [0.0, 0.0] |
| overkill dies @cost-opt | 13 | [7.0, 20.0] |

- 대표 point(seed=7, run_fbm): escape **89.6 DPPM**, 미검 die **0**. 위 bootstrap CI 가 그 point 를 둘러싼다.
- bit-level escape DPPM 은 소수의 누설 셀에 좌우되어 CI 가 상대적으로 넓다 → **재측정/노이즈 저감의 가치**(robustness 와 일관).

## 4. 정직성 플래그 — 대표 seed 는 분포의 우호적 가장자리에 있다

주요 수치는 canonical seed=7 값이다: escape@cost-opt **89.6 DPPM**,
미검 die@gb0 **86**, 비용절감 **80.3%**.
20개 seed 분포와 비교하면 seed7 는 escape·절감 모두 **상단(favorable edge)** 에 위치한다:

| 지표 | seed7(대표) | 다중seed median | 95% CI | 위치 |
|---|---|---|---|---|
| escape@cost-opt (DPPM) | 89.6 | 82.2 | [73.1, 85.9] | 상단↑ |
| 미검 die@gb0 | 86 | 74 | [61.5, 83.6] | 상단↑ |
| 비용절감 (%) | 80.3 | 76.9 | [73.7, 79.4] | 상단↑ |


## 산출물

- `reports/e2_uncertainty.csv`(다중 seed CI), `reports/e2_multiseed_runs.csv`(seed별 raw),
  `figures/e2_uncertainty.png`(operating-point 안정성 + bootstrap CI).

## 금지선

- CI 는 **합성 생성·재표집의 통계 변동**일 뿐 실제 실측 변동이 아니다. 절대 DPPM 은 illustrative.
