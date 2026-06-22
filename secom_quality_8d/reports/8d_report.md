# 8D Report: SECOM 공정 센서 pass/fail 분석

기준일: 2026-05-09

## D1. Team / Scope

- 목적: 공개 제조 공정 sensor data에서 pass/fail 예측 결과를 품질 문제 해결 문법으로 정리한다.
- 범위: UCI SECOM dataset. 실제 또는 특정 제조사의 현장 데이터가 아니다.
- 분석 역할: 데이터 결측/불균형 확인, baseline 모델 비교, threshold trade-off 분석, 원인 후보 제한 해석.

## D2. Problem Description

- 전체 sample: 1567
- pass sample: 1463
- defect sample: 104
- defect rate: 6.637%
- 품질 관점 문제: defect class가 희소하므로 단순 accuracy는 높은데 defect를 놓치는 모델이 나올 수 있다.

## D3. Interim Containment

- 운영 threshold를 고정 0.50으로만 두지 않고 defect recall과 false alarm rate를 함께 보며 조정한다.
- defect score가 높은 sample은 재검사 또는 추가 sensor 확인 대상으로 분리한다.
- false alarm 비용이 큰 경우, threshold별 alarm volume을 별도 모니터링한다.

## D4. Root Cause Candidates

확정 원인이 아니라 후속 점검 후보로만 기록한다.

- 결측치가 많은 feature가 일부 존재해 imputation 전략에 따라 판단 경계가 달라질 수 있다.
- defect class 비중이 낮아 class imbalance가 recall 저하를 만든다.
- feature 이름이 익명화되어 있어 상위 중요 feature를 실제 설비·공정 원인으로 단정할 수 없다.
- label noise 또는 공정 조건 drift 가능성은 원본 데이터만으로 확인할 수 없다.

## D5. Corrective Actions Selected

- 결측 처리 2개 전략 비교: 모든 feature 유지 후 median imputation, 결측률 50% 초과 feature 제거 후 median imputation.
- baseline 2개 비교: logistic regression, random forest.
- threshold sweep으로 missed defect와 false alarm trade-off를 표로 남긴다.
- feature importance는 원인 확정이 아니라 추가 점검 우선순위 후보로만 사용한다.

## D6. Validation

선택 operating point:

- strategy: median_all
- model: random_forest
- threshold: 0.10
- defect recall: 0.774
- false alarm rate: 0.248
- missed defect count: 7
- false alarm count: 109

전체 결과는 `reports/metrics.csv`, threshold별 결과는 `reports/threshold_tradeoff.csv`를 기준으로 확인한다.

## D7. Prevent Recurrence

- 운영 시 accuracy가 아니라 defect recall, false alarm rate, missed defect count를 함께 monitoring한다.
- feature별 결측률 drift를 정기 점검한다.
- 고위험 score sample과 missed defect case를 failure gallery로 축적한다.
- 실제 현장 데이터에서는 feature 이름, 설비 조건, 공정 step, 작업 이력과 연결해 원인 검증을 별도로 수행한다.

## D8. Closure / Note

이 산출물은 실제 현장 개선 완료 사례가 아니라 공개 제조 데이터를 활용한 분석 프로젝트다. 제조 데이터 품질 판단에서 미검출과 오경보를 분리해 보고, 모델 결과를 8D report 형식으로 정리한 것이다.
