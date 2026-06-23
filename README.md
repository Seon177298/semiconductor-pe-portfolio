# 반도체 품질·불량분석 포트폴리오

공개 데이터와 합성 데이터로 Product Engineering(PE) 관점의 데이터 의사결정을 재현한 4개 프로젝트 모음이다.
test 기준선을 어디에 둘지, escape(불량 유출)와 overkill(양품 폐기)의 trade-off를 어떻게 잡을지, 불량이 났을 때 무엇을 먼저 확인할지를 공통 주제로 한다.

## 데이터 경계

이 저장소의 모든 분석은 공개 데이터(UCI SECOM, Kaggle WM-811K, 선택적으로 MVTec AD)와 합성 데이터만 사용한다.
특정 회사·FAB·라인의 데이터, 실측치, 비공개 데이터는 포함하지 않는다. 결과는 원인 확정이 아니라 추가 점검 후보로 해석한다.

## 프로젝트

### 1. fail_bit_map_pe — 물리 margin 모델 기반 guardband·escape/overkill·repair·FA

합성 fail bit map 위에서 cell read-margin을 Vt/retention/Arrhenius로 모델링(test 80°C vs field 85°C)하고, guardband를 sweep하며 escape·overkill·ship/repair/scrap disposition·yield를 정량화해 비용 최적 guardband를 고른다. fail pattern은 rule / RandomForest / CNN으로 분류하되 라벨을 데이터 생성 latent로 두어 circularity를 제거했다.

- guardband 전/후(cost-opt gb=0.05): bit escape 550→90 DPPM, 미검 die 86→0, 총비용 약 −80% (대표 seed; 20-seed 전형값 약 −77%)
- 분류(held-out): rule 0.982 / RandomForest 1.000 / CNN 0.998. RF=1.000은 leakage가 아니라 합성 패턴의 feature-space 분리성 때문임을 명시
- robustness 4축(노이즈·defect·비용비·온도): operating point는 상수가 아니라 함수
- shmoo(E3)·HBM stacking(E4): Vdd×timing×온도 operating window로 FA 가설별 형상을 구분하고, die escape를 `(1−p)^K`로 환산
- 불확실성·한계·throughput(E2/E1/E5): 다중 seed+bootstrap 95% CI(cost-opt gb=0.05가 20 seed 전부), 전형 절감 약 −77%; RF=1.000은 합성 분리성임을 노이즈/도메인시프트 하락 곡선(1.00→0.57)으로 확인; 단위테스트 22개

### 2. secom_quality_8d — pass/fail 기준선과 escape/overkill trade-off

공개 SECOM 센서 데이터(1,567 sample, 590 feature)로 pass/fail을 예측하고 threshold를 움직여 trade-off를 본 뒤, 결과를 8D report와 Streamlit 데모로 정리했다.

- median_all · random_forest · threshold 0.10: defect recall 0.774 / false alarm rate 0.248 / missed defect 7
- nested CV로 누수 없이 재검증(같은 threshold에서 recall 0.667, 95% CI 0.524–0.793)

### 3. manufacturing_quality_platform — 분석을 운영 흐름으로 (DB→API→dashboard→alert→8D)

SECOM 분석을 CSV에서 끝내지 않고 SQLite → FastAPI → Streamlit → alert → 8D report로 확장하고, 시연용 synthetic digital-thread layer(모든 row `is_synthetic=true`)를 보조로 붙인 미니 품질 플랫폼이다.

- first-40-sensor logistic regression · threshold 0.50: false alarm 494 / missed defect 33 (결함 104개 기준), threshold 변경이 운영자 화면에서 escape/overkill을 어떻게 바꾸는지 보이기 위한 baseline

> secom_quality_8d와 manufacturing_quality_platform의 SECOM 수치가 다른 것은 모델(random forest vs logistic regression), threshold(0.10 vs 0.50), feature 범위(전체 vs first-40)가 다르기 때문이다. 하나로 합치지 않고 설정과 함께 표기했다.

### 4. wafermap_wm811k — lot-group 누수 제거 평가 + 취약클래스 recall 개선

공개 WM-811K wafer map 데이터로 결함 패턴을 분류하되, 같은 lot의 wafer가 train/test에 섞이는 lot leakage를 제거하는 데 초점을 둔다. accuracy가 아니라 macro-F1과 취약클래스 recall을 lot 단위 group split 위에서 본다.

- lot-group split(헤드라인): macro-F1 random 0.584 → lot-group 0.621 → 재학습 0.675, Center recall 0.505 → 0.765 (seed=42)
- 재학습 = sqrt sampler + augmentation + focal loss(γ=2). Center·Loc·macro-F1 단조 개선
- Scratch(최소 클래스 n=1,193)는 단일 seed 분산이 커 개선을 단정하지 않고 다중 seed가 필요하다고 정직 보고
- `lotName`은 공개 데이터 proxy이며 실제 fab lot 이력이 아니다

## 공통 프레임 — cost-weighted operating point

이 중 cost 기반 세 프로젝트(fail_bit_map_pe · secom_quality_8d · manufacturing_quality_platform)는 같은 구조를 공유한다. operating point(threshold/guardband)를 accuracy가 아니라 escape↔overkill 비대칭 비용으로 고른다.

> cost ≈ c_escape·escapes + c_overkill·overkills + c_retest·flagged (+ DPPM penalty)

| 프로젝트 | operating point | escape | overkill | cost 적용 |
| --- | --- | --- | --- | --- |
| fail_bit_map_pe | guardband | 미검 die/bit | 과검 die/bit | `fbm_total_cost`로 cost-opt guardband 선정(E5에서 throughput까지 환산) |
| secom_quality_8d | threshold | missed defect | false alarm | threshold sweep + cost curve, nested-CV로 누수 없이 선택 |
| manufacturing_quality_platform | threshold (API) | missed defect | false alarm | `GET /quality/metrics?threshold=`로 escape/overkill 미리보기 |

## 분석 흐름 — 다음 실험으로 검증

- 단일 split 결과(recall 0.774)에 operating-point 누수 편향이 있다고 보고, 누수를 제거한 repeated nested CV로 다시 검증했다(0.774 → 0.667). → `secom_quality_8d/reports/rigor_summary.md`
- 분류 정확도 1.000이 leakage인지 의심해, 노이즈·혼재·도메인 시프트로 난이도를 올리는 실험을 설계했다(1.000 → 0.57). → `fail_bit_map_pe/reports/e1_synthetic_limits_summary.md`
- 대표 seed 하나의 수치가 우연인지 확인하려고, 다중 seed(20) + bootstrap로 CI를 붙였다. → `fail_bit_map_pe/reports/e2_uncertainty_summary.md`
- 한 불량 die를 두고 원인 가설 → 측정 설계 → 확정 전 멈춤 기준으로 다음 측정을 설계했다. → `fail_bit_map_pe/FA_walkthrough.md`

## 저장소 구성

- 대용량 원시 데이터와 다중 MB 규모 생성 CSV는 저장소에 포함하지 않는다(재실행으로 재생성). 사람이 읽는 markdown 리포트와 소형 표만 커밋한다.
- 각 프로젝트의 상세 실행 방법과 데이터 출처는 프로젝트별 `README.md`를 참고한다.
