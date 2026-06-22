# 공정능력(Cpk) · 측정시스템분석(Gage R&R)

> **정직성:** Cpk 의 spec limit 은 UCI SECOM 이 제공하지 않아 **pass-class 분포 ±3σ 로 illustrative 유도**
> (익명화 데이터 → 실제 규격/공정 단정 금지). Gage R&R 은 SECOM 에 MSA 구조가 없어 **합성 MSA 연구**.
> 둘 다 **방법(품질공학 toolkit) 시연**이지 실측 성능 보고가 아니다.

## 1. Cp/Cpk — quality-relevant SECOM sensor (방법 시연)

pass/defect 표준화 평균차가 큰 sensor 6개에 대해, pass-class ±3σ 를 규격으로 두고 전체 분포의 공정능력을 계산.

| sensor | Cp | Cpk | class gap(σ) | LSL~USL (illustrative) |
|---|---|---|---|---|
| f_059 | 0.977 | 0.963 | 0.624 | -25.3759~30.5028 |
| f_103 | 0.977 | 0.964 | 0.607 | -0.0189~-0.0009 |
| f_510 | 0.949 | 0.937 | 0.528 | -52.8688~161.75 |
| f_348 | 0.872 | 0.86 | 0.519 | -0.0067~0.0553 |
| f_431 | 0.875 | 0.864 | 0.487 | -74.3161~116.6986 |
| f_434 | 0.856 | 0.846 | 0.452 | -73.9123~101.3508 |


- Cpk < 1.0 인 sensor 는 (illustrative 규격 기준) 전체 분포가 pass 중심에서 벗어나 능력이 낮음 → defect 와 연관된
  공정 산포 후보. (class gap(σ) 가 클수록 defect 와의 분리가 큼.)
- **해석:** Cpk = min(Cpu, Cpl). 규격은 illustrative 이므로 절대 수치가 아니라 **상대 비교·방법**으로 읽는다.

## 2. Gage R&R (synthetic MSA) — 측정시스템이 부품 변동을 구분하는가

parts×operators×trials 합성 MSA 를 crossed ANOVA 로 분해:

| 성분 | 표준편차 |
|---|---|
| EV (repeatability) | 0.334 |
| AV (reproducibility, operator) | 0.174 |
| interaction | 0.195 |
| **GRR (EV⊕AV⊕int)** | **0.424** |
| PV (part) | 4.687 |
| TV (total) | 4.706 |

- **%GRR = 9.0%** → AIAG 기준 **양호(<10%)**. **ndc = 15.6** (≥5 권장: 측정계가 부품 등급을 구분 가능).
- %GRR 은 측정변동(GRR)이 전체변동(TV)에서 차지하는 비율 — 클수록 측정계가 부품 차이를 못 가린다.

## 금지선

- Cpk 규격·Gage R&R 데이터는 illustrative/synthetic. 결론은 **방법과 상대 비교**이지 실제 공정능력·MSA 결과가 아니다.
- (figure: `figures/process_capability.png`.)
