# E3 — 2D Shmoo / operating window: 형상이 메커니즘을 진단한다

> **합성 데이터.** 같은 물리 코어(margin = margin0 + γ(Vdd−Vdd_nom) − leak·t·A(T),
> Ea=0.6eV, γ=0.5/V, Vdd_nom=1.1V)로 세 FA 가설 die 를 만들어
> Vdd∈[1.0,1.2]V × timing∈[0.5,2.5] × [25, 80, 85]°C 를 sweep.
> 각 corner pass = die 가 구제 가능(disposition ≠ scrap). seed=5.

## window 면적(=pass 비율)과 형상 차이

| FA 가설 | 25°C | 80°C | 85°C | shmoo 형상 | 다음 FA 액션 |
|---|---|---|---|---|---|
| retention-limited (leak) | 1.00 | 0.14 | 0.06 | 수평 경계, 온도↑/timing↑ 에서 창 축소 | 가속(hot) retention 재측정·refresh 강화·온도 guardband |
| Vdd/sensing-limited (parametric) | 0.27 | 0.27 | 0.27 | 수직 Vdd 벽, 온도/timing 무관 | Vdd/sense trim·스크리닝, 전압 마진 조정 |
| hard structural (cluster) | 0.00 | 0.00 | 0.00 | 창 없음(전 corner fail) | 파라메트릭 튜닝 무의미 → 물리 PFA/공정 이슈 |


- **retention-limited:** 저온에선 거의 전 영역 pass, 고온에서 창이 줄어든다(온도 의존). 경계는 주로
  **timing(retention) 축에 수평**이고 Vdd 로 약하게 기운다 → 누설/retention 한계의 지문.
- **Vdd-limited:** 창 면적이 온도에 거의 불변(누설 없음), 경계가 **Vdd 축에 수직인 벽** → 파라메트릭
  sensing/drive 마진 한계. Vdd 만 올리면 회복.
- **hard structural:** 어느 corner 에서도 창이 없다 → cluster 가 Vdd/timing/온도와 무관하게 hard fail.
  **파라메트릭 튜닝으로 못 고침**을 즉시 보여주는 신호.

## 신호처리·FA 연결

shmoo 는 (Vdd, timing) 평면의 2D pass/fail 장(field)이고, 진단 정보는 **경계의 방향(수평/수직)·온도에
따른 이동**에 있다 — 경계검출+민감도(∂window/∂T, ∂Vdd_min/∂t) 문제로 볼 수 있다. 형상 하나로 다음
액션이 갈린다: retention→가속/재측정, Vdd→트리밍/스크리닝, hard→물리 PFA. (figure:
`figures/shmoo_operating_window.png`, 표: `reports/shmoo_window.csv`.)

## 금지선

- 합성 die. γ·Vdd_nom·leak 분포는 illustrative 가정. 절대 전압/timing 값은 실제 스펙이 아니다.
- 형상→메커니즘 매핑은 **분석 사고의 구조화**이지 실제 원인 확정이 아니다(실측·PFA 없음).
