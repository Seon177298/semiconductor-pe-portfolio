# Synthetic Fail Bit Map (FBM) — guardband 전/후 요약

> **합성 데이터.** 실제 fab / 계측 데이터가 아니며, 수치는 방법론 시연용이다.
> 셀 어레이 128x128 = 16,384 cells/die, dies=300,
> 측정 노이즈 1σ=0.05 (margin 단위), seed=42. 재현: `python synth_fail_bit_map.py`.

## 1. Bin 정의

| bin | type | 설명 | default disposition |
|---|---|---|---|
| BIN1 | PASS | 정상 (fail 없음 또는 redundancy 내 복구 가능) | ship |
| BIN2 | SINGLE_BIT | 산발 single-bit fail (spare 로 복구 가능) | ship-after-repair |
| BIN3 | ROW_FAIL | wordline/row 라인 fail (spare row 로 복구 시도) | repair-limited |
| BIN4 | COLUMN_FAIL | bitline/column 라인 fail (spare column 로 복구 시도) | repair-limited |
| BIN5 | CLUSTER_FAIL | 국소 cluster fail (particle 등, 복구 불가) | scrap |
| BIN6 | EDGE_FAIL | edge ring fail (공정 edge 효과, 복구 불가) | scrap |

## 2. Fail 유형 자동 분류 성능 (rule-based)

- detected bin vs true bin 일치율(@ guardband=0): **0.880** (264/300 dies)
  - 구조성 fail(BIN3 row/BIN4 col/BIN5 cluster/BIN6 edge)은 전부 정확히 분류됨. 불일치는 거의
    BIN1(PASS)↔BIN2(single-bit) 경계로, 측정 노이즈 speckle 때문이며 둘 다 'ship' 등급이라
    출하 판정에는 영향이 없다.
- **ship/scrap 출하 판정 정확도**: guardband 0 → **0.977**, guardband 0.06 → **0.950**
- 혼동행렬(@guardband=0): `reports/fbm_classification_confusion.csv`

## 3. Guardband 설정 전/후 (escape=미검, overkill=과검)

| | guardband | escape bits | escape DPPM | overkill bits | overkill DPPM | escape dies | overkill dies | yield |
|---|---|---|---|---|---|---|---|---|
| **전 (no guardband)** | 0.0 | 2123 | 431.93 | 1158 | 235.6 | 5 | 2 | 0.663 |
| **후 (guardband)** | 0.06 | 268 | 54.52 | 3680 | 748.7 | 0 | 15 | 0.603 |

해석: guardband 를 0→0.06 로 올리면 bit escape 가 431.93→54.52 DPPM 으로 줄지만(미검 감소),
overkill 은 235.6→748.7 DPPM 으로 늘어난다(과검 증가). 전수 sweep 은 `reports/fbm_gb_sweep.csv`.
어디서 멈출지는 통계가 아니라 **비용**으로 결정한다 → `cost_model.py` (escape/overkill 비용 환산).

## 4. 한계 / 금지선

- 합성 margin 모델은 실제 cell Vt/read-margin 물리를 단순화한 것이다.
- redundancy(spare row 4, col 4, single-bit budget 150)·분류 임계값은 시연용 가정이다.
- 실제 FBM 은 ECC, repair allocation, bin map, tester 조건과 연결해 해석해야 한다.
