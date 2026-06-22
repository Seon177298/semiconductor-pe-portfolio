# Fail Bit Map — guardband, escape/overkill 비용, repair, FA

합성 fail bit map 위에서 Product Engineering의 핵심 의사결정을 물리·비용 논리로 구현한 프로젝트다.
fail bit map의 패턴 분류, bin, redundancy repair, escape/overkill 비용 환산(DPPM·폐기·재테스트·yield loss),
guardband 선정, FA(불량분석) 측정 설계를 한 흐름으로 다룬다. 단순 Gaussian 모델의 v1을 물리 margin 모델·학습기반 분류·robustness·shmoo·HBM stacking으로 확장한 것이 v2다. 비용 단위는 임의 상대 단위(USD-equivalent)의 가정값이다.

## 물리 margin 모델

cell read-margin을 `margin(t,T,Vdd) = margin0 + γ(Vdd−Vdd_nom) − leak·t·A(T)`로 모델링한다. A(T)는 Arrhenius 온도 가속(Ea=0.6eV)이다. test corner(80°C)에서 측정하고 field worst(85°C)를 ground truth로 두면, retention이 약한 셀은 test에서 통과해도 field에서 떨어질 수 있다. 이것이 escape(미검 출하)의 물리적 원인이고, guardband는 측정 노이즈와 test↔field 코너 갭을 흡수하는 양이다.

## 주요 결과

각 결과를 그대로 두지 않고, 의심스러운 지점은 추가 실험으로 다시 확인했다 — 분류 정확도 1.000은 E1(난이도 하락 곡선)으로, 대표 seed 수치는 E2(다중 seed CI)로 검증했다.

### 1. fail-pattern 분류 (held-out, 비순환 평가)

라벨을 데이터 생성 latent(주입 패턴)로 두어, rule 분류기 출력으로 채점하던 circularity를 제거했다.

| model | held-out accuracy | macro F1 | 입력 |
| --- | --- | --- | --- |
| rule-based | 0.982 | 0.986 | fail map 고정 규칙 |
| RandomForest | 1.000 | 1.000 | engineered feature 15개 |
| CNN | 0.998 | 0.996 | 32×32 pooled map |

RandomForest의 1.000은 leakage가 아니라 합성 6-class가 feature 공간에서 깨끗이 분리되기 때문이다(train/test 분리). feature가 생성 패턴과 같은 축이라 over-separable한 면이 있어 점수 자체에 의미를 두지 않는다. 실제 wafer의 혼재 패턴에서는 내려간다(E1 참고).

### 2. guardband 전/후 (test 80°C vs field 85°C, cost-opt gb=0.05)

| | guardband | escape DPPM | overkill DPPM | 미검 die | 과검 die | yield | total cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 전 | 0.00 | 550 | 5.7 | 86 | 0 | 0.791 | 8753 |
| 후 | 0.05 | 90 | 462 | 0 | 13 | 0.725 | 1722 |

guardband가 노이즈와 코너 갭을 흡수해 bit escape 550→90 DPPM, 미검 die 86→0, 총비용 약 −80%. disposition(ship/repair/scrap)은 435/751/314 → 434/653/413. 이 주요 수치는 대표 seed의 값이고, 전형값은 E2의 CI로 본다.

### 3. robustness 4축

operating point는 상수가 아니라 노이즈·불량률·비용비·온도의 함수다.

- escape:overkill 비용비 → guardband: 30:1 → 0.06, 1:1 → 0.03, 1:10 → 0.01
- 측정 노이즈 σ↑ → 같은 guardband의 overkill 급증, escape@opt 51 → 827 DPPM
- test 온도 corner: 80°C → escape 약 84 DPPM vs 60°C → 약 1266 DPPM. test corner가 field에서 멀수록 retention escape floor가 오른다. 이 모델 가정에서는 hot-corner retention 테스트가 guardband보다 더 근본적인 escape 저감 수단이다.

### 4. shmoo operating window (E3)

물리 코어에 Vdd 축을 더해 Vdd×timing×온도를 sweep하고, FA 가설별 pass/fail 창의 형상으로 메커니즘을 구분한다.

| FA 가설 | window 25/80/85°C | 형상 |
| --- | --- | --- |
| retention-limited | 1.00 / 0.14 / 0.06 | 수평, 온도 따라 축소 |
| Vdd-limited | 0.27 / 0.27 / 0.27 | 수직 벽, 온도 불변 |
| hard structural | 0 / 0 / 0 | 창 없음 |

여기 timing 축은 tester clock이 아니라 정규화 retention/refresh 시간이다. 진단 대상 die는 메커니즘을 직접 주입한 것이라, 메커니즘→형상 매핑을 보인 forward 시뮬레이션이다.

### 5. HBM KGD stacking (E4)

큐브는 core die를 적층하므로 한 장만 bad여도 큐브가 손실된다. die escape p는 stack 높이 K에서 `1−(1−p)^K`로 증폭된다. 출하 die 기준 무가드밴드 escape 7.25%(86/1186)를 cost-opt 선별로 0%(rule-of-three 95% 상한 ≤0.28%)까지 낮추면, 12-high 큐브 all-good이 0.41에서 약 0.97(95% 최악)로 올라간다. 이 값은 die escape 단일 요인만 고려한 것이고 TSV/assembly yield는 분리했다.

### 6. 불확실성·한계·throughput (E2/E1/E5)

- E2: 다중 seed(20) + bootstrap로 operating point에 95% CI를 붙였다. cost-opt guardband는 20 seed 전부 0.05. 주요 수치 −80%·escape 89.6 DPPM·미검 86은 대표 seed(seed7)의 값으로 분포의 상단이며, 전형값은 절감 약 −77%(CI −74~−79%), escape 약 82 DPPM(CI 73~86)이다.
- E1: 난이도를 올리면 정확도가 떨어진다. RF 1.000 → 0.936(노이즈), → 0.861(혼재), → 0.57(도메인 시프트). CNN도 1.000 → 0.889. 1.000이 합성 분리성 때문임을 곡선으로 보인 것이다.
- E5: cost-opt guardband는 good-die throughput −6%로 escape −84%를 얻는다. reject pool 재측정은 overkill이 큰 구간에서만 수지가 맞는다.

### 7. SECOM operating point (공개 데이터)

median_all / random_forest / threshold 0.10 기준 recall 0.774 / false alarm 0.248 / missed 7 (`reports/secom_canonical_operating_point.md`).

## 재현

```bash
python run_fbm.py            # 물리 margin + repair disposition + guardband sweep
python train_classifier.py   # RF+CNN vs rule, held-out
python robustness.py         # 노이즈/defect/비용비/온도 corner 민감도
python shmoo.py              # E3 2D operating window
python kgd_stacking.py       # E4 HBM stacking yield
python e2_uncertainty.py     # E2 다중 seed + bootstrap 95% CI
python e1_synthetic_limits.py# E1 정확도 하락 곡선
python e5_throughput.py      # E5 test-time/throughput
python -m pytest tests/ -q   # 단위테스트 22개
python synth_fail_bit_map.py && python cost_model.py   # v1 baseline (비교용)
```

환경: python3.9, sklearn 1.6.1, torch 2.8.0, scipy 1.13.1, pandas 2.3.3.

## 산출물

| 파일 | 내용 |
| --- | --- |
| `fbm_core.py` | 물리 margin 모델 + repair disposition + feature + rule 분류 |
| `run_fbm.py` | 데이터셋 + guardband sweep + disposition/yield (`reports/fbm_v2_*`) |
| `train_classifier.py` | RF+CNN vs rule, held-out (`reports/classifier_*`) |
| `robustness.py` | 4축 민감도 (`reports/robustness_*`) |
| `shmoo.py` | E3 Vdd×timing×온도 operating window (`reports/shmoo_*`) |
| `kgd_stacking.py` | E4 HBM stack yield (`reports/kgd_*`) |
| `e2_uncertainty.py` | E2 다중 seed + bootstrap 95% CI (`reports/e2_*`) |
| `e1_synthetic_limits.py` | E1 정확도 하락 곡선 (`reports/e1_*`) |
| `e5_throughput.py` | E5 test-time/throughput (`reports/e5_*`) |
| `tests/` | 단위테스트 22개 (pytest) |
| `synth_fail_bit_map.py`, `cost_model.py` | v1 baseline (비교용) |
| `FA_walkthrough.md` | FA 워크스루(column fail 사례, 측정 설계와 멈춤 기준) |
| `reports/secom_canonical_operating_point.md` | SECOM operating point |
