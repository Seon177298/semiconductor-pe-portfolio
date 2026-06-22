# Escape/Overkill 비용 환산 — operating point 재선정

> 비용 단위는 임의 상대 단위(USD-equivalent), **시연용 가정값**이다.
> FBM=합성 데이터, SECOM=공개(UCI) 익명 데이터. 실제 실제 원가가 아니다.

## 비용 가정 (명시)

**FBM (memory die)**
- escape(불량 출하) field 비용 = 80.0/die
- 폐기/yield-loss(양품 과검 폐기) = 6.0/die
- 재테스트 TAT(fail 판정 die 추가 test insertion) = 1.5/die
- DPPM field-quality 패널티 = 0.5/DPPM (escape bit DPPM 기준)

**SECOM (production unit)**
- escape(불량 출하) = 50.0/unit
- overkill(오경보→재검사 TAT) = 3.0/unit

## 1. FBM — 통계 최적 vs 비용 최소

- **통계 최적**(escape+overkill die 최소): gb=0.00 | escape 5 dies / 431.9 DPPM, overkill 2 dies, yield 0.663, total cost **779.5**
- **비용 최소**: gb=0.06 | escape 0 dies / 54.5 DPPM, overkill 15 dies, yield 0.603, total cost **295.8**

escape die 의 field 비용(80.0)이 과검 폐기(6.0)·재테스트(1.5)보다 크기 때문에,
비용 최소점은 단순 오분류 최소점과 다를 수 있다(불량 미검을 더 줄이는 쪽으로 이동).
전 구간: `reports/fbm_cost_curve.csv`.

### FBM cost-optimal guardband sensitivity (escape:overkill 비용비)

| escape:overkill | cost-optimal guardband |
|---|---|
| 3:1 | 0.02 |
| 10:1 | 0.02 |
| 30:1 | 0.02 |
| 1:1 | 0.00 |
| 1:3 | 0.00 |
| 1:10 | -0.02 |

## 2. SECOM — 통계 최적 vs 비용 최소 (median_all / random_forest)

- **통계 최적**(recall≥0.70 후 false alarm 최소 = 원래 운영점): threshold=0.10 | recall 0.774, missed(escape) 7, false-alarm(overkill) 109, total cost **677.0**
- **비용 최소**: threshold=0.10 | recall 0.774, missed(escape) 7, false-alarm(overkill) 109, total cost **677.0**

전 구간: `reports/secom_cost_curve.csv`.

### SECOM cost-optimal threshold sensitivity (escape:overkill 비용비)

| escape:overkill | cost-optimal threshold |
|---|---|
| 3:1 | 0.15 |
| 10:1 | 0.10 |
| 30:1 | 0.10 |
| 1:1 | 0.30 |
| 1:3 | 0.30 |
| 1:10 | 0.30 |

## 핵심 메시지

- operating point(guardband/threshold)는 **통계 지표가 아니라 비용 비대칭으로 결정**된다.
- escape 가 비쌀수록 미검을 줄이는 쪽(낮은 threshold / 높은 guardband)으로,
  overkill 이 비쌀수록 과검을 줄이는 쪽으로 최적점이 이동한다 (위 sensitivity 표).
- 따라서 PE 의 test/guardband 정책은 "정확도 최대화"가 아니라
  "escape·overkill·재테스트 TAT·yield-loss 의 총비용 최소화"로 설명해야 한다.
