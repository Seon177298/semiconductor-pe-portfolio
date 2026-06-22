# Robustness 민감도 분석 (C)

> 합성 데이터, 비용 illustrative, seed=21, dies/point=400. 곡선: `figures/robustness_sensitivity.png`.

## 1. 측정노이즈 σ → cost-optimal guardband
측정 불확실성이 커질수록 같은 guardband 의 overkill 이 급증해, escape 를 줄이는 능력이 떨어진다
(escape DPPM@opt 이 σ와 함께 상승). 즉 노이즈는 escape/overkill 프런티어 전체를 악화시키며,
guardband 만으로는 한계가 있어 **계측 반복(재측정)·노이즈 저감**이 선행돼야 한다.

| sigma_meas | cost_opt_guardband | escape_dppm_at_opt | overkill_dppm_at_opt | escape_dies_at_opt | min_cost |
|---|---|---|---|---|---|
| 0.01 | 0.05 | 51.12 | 182.5 | 0.0 | 468.1 |
| 0.02 | 0.05 | 83.62 | 460.21 | 0.0 | 490.3 |
| 0.03 | 0.04 | 174.56 | 734.1 | 1.0 | 623.3 |
| 0.04 | 0.03 | 303.34 | 1370.54 | 1.0 | 857.2 |
| 0.05 | 0.02 | 501.4 | 2483.83 | 0.0 | 1006.7 |
| 0.06 | 0.01 | 827.48 | 4189.15 | 1.0 | 1393.7 |

## 2. defect rate → cost-optimal guardband
불량률이 오르면 escape 위험이 커져 guardband 가 (대체로) 강화되는 경향.

| defect_scale | true_bad_rate | cost_opt_guardband | escape_dppm_at_opt | overkill_dppm_at_opt | min_cost |
|---|---|---|---|---|---|
| 0.5 | 0.242 | 0.05 | 38.3 | 424.04 | 431.6 |
| 1.0 | 0.265 | 0.05 | 62.41 | 446.62 | 537.2 |
| 1.5 | 0.428 | 0.06 | 68.51 | 998.38 | 574.3 |
| 2.0 | 0.455 | 0.05 | 163.88 | 503.85 | 539.4 |
| 3.0 | 0.492 | 0.06 | 127.26 | 1168.06 | 539.1 |

## 3. escape:overkill 비용비 → cost-optimal guardband
operating point 는 통계가 아니라 **비용 비대칭**이 결정한다.

| escape_weight | overkill_weight | cost_opt_guardband | escape_dppm_at_opt | overkill_dppm_at_opt |
|---|---|---|---|---|
| 30.0 | 1.0 | 0.06 | 43.33 | 955.35 |
| 10.0 | 1.0 | 0.05 | 83.62 | 460.21 |
| 3.0 | 1.0 | 0.04 | 137.18 | 209.96 |
| 1.0 | 1.0 | 0.03 | 207.67 | 92.32 |
| 1.0 | 3.0 | 0.02 | 298.92 | 37.69 |
| 1.0 | 10.0 | 0.01 | 394.29 | 16.48 |

## 4. test 온도 corner → escape / guardband  (물리 모델 B 의 의미)
test corner 가 field(85°C)에서 멀어질수록(cooler) Arrhenius 갭이 커져 **retention escape floor 가 상승**하고
같은 비용을 맞추려면 guardband 를 키워야 한다. 즉 **hot-corner(또는 가속 retention) 테스트**가
guardband 보다 근본적인 escape 저감 레버임을 보여준다.

| test_corner_C | arrhenius_gap_to_field | escape_dppm_gb0 | cost_opt_guardband | escape_dppm_at_opt | overkill_dppm_at_opt | min_cost |
|---|---|---|---|---|---|---|
| 60.0 | 4.3 | 1630.1 | 0.09 | 1266.02 | 6848.14 | 2385.0 |
| 65.0 | 3.16 | 1609.95 | 0.09 | 932.46 | 6850.59 | 1660.2 |
| 70.0 | 2.34 | 1480.87 | 0.08 | 612.79 | 3590.55 | 1522.4 |
| 75.0 | 1.75 | 1083.22 | 0.07 | 268.55 | 1810.0 | 834.8 |
| 80.0 | 1.32 | 498.35 | 0.05 | 83.62 | 460.21 | 490.3 |
| 85.0 | 1.0 | 78.58 | 0.01 | 37.84 | 100.1 | 467.4 |

## 핵심 메시지
- guardband·operating point 는 고정 상수가 아니라 **측정노이즈·불량률·비용비·테스트 온도 코너**의 함수다.
- 특히 retention escape 는 guardband 만으로 한계가 있고, **test 온도를 field 에 가깝게** 가져가는 것이 escape floor 자체를 낮춘다(4번). 이는 물리 margin 모델(Arrhenius)에서 자연히 도출된다.
