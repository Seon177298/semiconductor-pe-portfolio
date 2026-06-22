# FBM v2 — 물리 margin 모델 + repair disposition 요약

> **합성 데이터.** 실제 fab/계측 데이터 아님. seed=7, dies=1500.
> 물리 모델: read margin(t,T)=margin0 − leak·t·A(T), Arrhenius Ea=0.6eV,
> test corner 80°C vs field worst 85°C
> (A_test=38.0, A_field=50.0, 갭 1.3×).
> 측정노이즈 1σ=0.02V. 재현: `python run_fbm.py`. 불확실성(95% CI): `reports/e2_uncertainty_summary.md`.

## guardband 의 물리적 의미

guardband 는 단순 임의 마진이 아니라 **(1) 측정 불확실성(σ=0.02V) + (2) test(80°C)↔field(85°C) 온도/retention 코너 갭**을 흡수한다.
retention-weak(누설) 셀은 benign test corner 에서는 통과하지만 hot field corner 에서 fail → guardband 로 screen.
구조성 fail(row/col/cluster/edge)은 hard fail 이라 코너 무관하게 잡힌다.

## guardband 전/후 (escape=미검, overkill=과검)

| | guardband(V) | escape DPPM | overkill DPPM | escape dies | overkill dies | ship/repair/scrap | yield |
|---|---|---|---|---|---|---|---|
| **전 (no guardband)** | 0.0 | 550.13 | 5.7 | 86.0 | 0.0 | 435.0/751.0/314.0 | 0.791 |
| **후 (cost-opt)** | 0.05 | 89.64 | 461.63 | 0.0 | 13.0 | 434.0/653.0/413.0 | 0.725 |

- 통계 최적 guardband(오분류 die 최소) = 0.05 ; **비용 최소 guardband = 0.05** (`fbm_core.fbm_total_cost`).
- 전 구간: `reports/fbm_v2_gb_sweep.csv`. disposition stack: `figures/fbm_v2_disposition.png`.

## redundancy repair (D)

spare row 4 / col 4 / single-bit budget 150.
disposition 3-way: **ship**(무결/budget 내) / **repair**(spare 로 복구 후 출하) / **scrap**(redundancy 초과·cluster·edge).
yield = (ship+repair)/total. escape die = field-worst 기준 scrap 대상인데 출하된 die.
