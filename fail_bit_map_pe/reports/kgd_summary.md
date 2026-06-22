# E4 — HBM KGD stacking yield: die escape를 stack 곱셈으로 환산

> **합성 데이터.** die-escape 확률은 `reports/fbm_v2_gb_sweep.csv`(guardband sweep)에서,
> stack 환산은 (1−p_escape)^K. 합성 population 의 불량률(400/1500)이 높아 **절대 DPPM 은 illustrative**이며,
> 핵심은 (1) 곱셈 증폭 메커니즘과 (2) KGD 선별의 상대 임팩트다.

## 1. die test(KGD 선별)가 stack yield 를 지배하는 이유

HBM 큐브는 core die 12장을 TSV 로 적층한다. 적층 후 한 장이라도 true-bad 면 큐브 전체가
손실(또는 final stack test fail) → die 단위 escape p 는 stack 높이 K 에서 **1−(1−p)^K 로 증폭**된다.
die test 에서 새는 한 장이 die 한 개가 아니라 **완성 큐브 한 개**(core die 12장 + buffer + TSV
assembly + stack test)를 버리게 만든다 — KGD(Known-Good-Die) 선별이 HBM 수율의 핵심인 이유.

## 2. 우리 guardband 선별을 KGD escape 로 환산 (출하 die 기준)

stack 에 들어가는 것은 die test 를 통과해 **출하(ship+repair)** 되는 die 다. 그중 truly-bad = escape.

| KGD 선별 | guardband | escape die / 출하 die | per-die escape |
|---|---|---|---|
| **선별 약함** | 0.00 | 86 / 1186 | **7.25%** |
| **cost-opt 선별** | 0.05 | 0 / 1087 | **0.00%** (point) / ≤ 0.28% (95% UB, rule-of-three) |

cost-opt guardband 은 이 샘플에서 escape die=0 → 0/1087 의 정직한 상한은 rule-of-three 로 ≈3/1087=0.28%.

## 3. stack 높이별 큐브 yield (전/후 KGD)

| K | unscreened all-good | unscreened cube-escape | KGD all-good (95% worst) | ref 0.99^K |
|---|---|---|---|---|
| 4 | 0.740 | 0.260 | 0.989 | 0.961 |
| 8 | 0.548 | 0.452 | 0.978 | 0.923 |
| 12 | 0.405 | 0.595 | 0.967 | 0.886 |
| 16 | 0.300 | 0.700 | 0.957 | 0.852 |

- **12-high 주요 수치:** 약한 선별이면 큐브의 **59%가 latent bad die 를 품는다**
  (all-good 41%). cost-opt KGD 선별이면 all-good **≥97% (95% 최악)**,
  큐브-escape **≤3.3%**. 참고선 0.99 good/die → 0.99^12=0.886.
- 즉 die 단위로는 작아 보이는 escape 차이가 **12장 적층에서 큐브 손실로 곱셈 증폭**되며, die test guardband
  한 칸이 finished-cube 수율을 좌우한다. (figure: `figures/kgd_stack_yield.png`, 표: `reports/kgd_stacking.csv`.)

## 금지선

- 절대 escape/DPPM 은 합성 가정값. per-die intrinsic TSV/stacking yield 는 별도 항으로 분리(여기선 KGD 선별만 격리).
- "die test 의 escape 가 stack 에서 곱셈 증폭된다"는 **메커니즘과 상대 임팩트**를 보인 것이지 실측 수율 예측이 아니다.
