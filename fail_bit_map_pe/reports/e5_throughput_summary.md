# E5 — test time / throughput: guardband·재측정의 처리량 영향

> **합성 데이터.** 시간 상수 illustrative (base 8.0s/die, repair +3.0s, 재측정 +2.5s,
> rescue 70%). 입력은 guardband sweep 의 disposition 카운트(`reports/fbm_v2_gb_sweep.csv`).

## 1. guardband ↔ good-die throughput ↔ escape

| guardband | escape DPPM | yield | avg test time(s) | good-die throughput(/h) |
|---|---|---|---|---|
| 0.00 | 550.13 | 0.791 | 9.502 | 299.6 |
| **0.05 (cost-opt)** | 89.64 | 0.725 | 9.306 | 280.3 |

- cost-opt guardband(0.05)은 gb0 대비 good-die throughput **-6.4%** 로 거의 손해 없이 escape 를 **-83.7%**
  줄인다 → guardband 의 throughput 비용은 작고 품질 이득은 크다.
- guardband 를 더 키우면(>0.05) yield 가 무너져(scrap 급증) good-die throughput 이 급락한다(전 구간 `e5_throughput.csv`).

## 2. 재측정(re-measure) 정책의 trade — cost-opt 기준

reject(scrap) pool 을 한 번 더 측정해 noise 로 인한 overkill 을 구제하면:
- @cost-opt: 재측정 413 die, overkill 9 구제, yield 0.725→0.731,
  throughput **-6.1%**. → cost-opt 에선 overkill 이 이미 낮아 **재측정 이득이 throughput 비용에 못 미친다**.
- 재측정은 **overkill 이 큰 영역(느슨한 screen·노이즈 큰 tester)에서만** 수지가 맞는다 → robustness(노이즈↑→재측정/노이즈저감) 와 일관.

(figure: `figures/e5_throughput.png`.)

## 금지선

- 시간/throughput 절대값은 합성 가정. 결론은 **상대 trade**(guardband 의 throughput 비용은 작고, 재측정은 overkill 큰
  영역에서만 유효)이지 실측 처리량 예측이 아니다.
