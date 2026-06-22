# Fail-pattern 분류기 비교 (학습기반 vs rule-based)

> 합성 데이터. **라벨 = 데이터 생성 시 주입한 패턴(generative latent)** — rule classifier 출력이 아님.
> 따라서 rule-based 도 ML 과 **동일한 generative truth 로 채점**된다(circularity 차단). seed=13.
> train 1050 / test 450 (stratified 70/30). 6-class: PASS, SINGLE_BIT, ROW, COLUMN, CLUSTER, EDGE.

## held-out 정확도

| model | test accuracy | macro F1 | 입력 |
|---|---|---|---|
| rule_based | **0.9822** | 0.9859 | fail map (규칙) |
| random_forest | **1.0000** | 1.0000 | engineered features(15) |
| cnn | **0.9978** | 0.9960 | raw fail map 32×32 (학습) |

## class별 recall (held-out)

| class | rule_based | random_forest | cnn |
|---|---|---|---|
| PASS | 0.962 | 1.000 | 1.000 |
| SINGLE_BIT | 1.000 | 1.000 | 1.000 |
| ROW | 0.982 | 1.000 | 1.000 |
| COLUMN | 0.982 | 1.000 | 1.000 |
| CLUSTER | 1.000 | 1.000 | 0.971 |
| EDGE | 0.974 | 1.000 | 1.000 |

## 해석 (정직)

- **circularity 차단:** 라벨이 생성 latent 이므로 rule-based 의 점수는 더 이상 자명한 1.0 이 아니다(0.982).
  세 분류기가 같은 기준으로 비교된다. (이전 v1 의 "rule on rule-data 0.88"은 자기참조였음.)
- rule-based 의 주 오류원은 PASS↔SINGLE_BIT 경계(측정 노이즈 speckle)와 **부분(50% 미만) line** 이
  고정 임계(0.5)를 못 넘겨 SINGLE_BIT 로 새는 경우다. 학습기반은 행/열 집중도를 **연속값**으로 학습해 흡수한다.
- **RandomForest 가 1.0 인 것은 leakage 가 아니라** 합성 6-class 가 feature 공간에서 깨끗이 분리되기 때문이다
  (train/test 는 분리). 즉 이 실험의 가치는 "높은 점수"가 아니라 **(1) 비순환 평가 + (2) raw map 만으로 학습하는 CNN**
  으로 규칙 의존 없이 패턴을 구분함을 보인 데 있다. 실제 wafer 의 혼재 패턴에서는 점수가 내려갈 것이다.
- **RandomForest 상위 feature**: col_count_std, row_count_std, max_col_frac, ring_fill, max_row_frac (전체 `reports/classifier_rf_feature_importance.csv`).
- **CNN** 은 hand-rule feature 없이 raw fail map 만으로 학습 → 규칙 의존 없이도 패턴을 구분함을 보임.
- 한계: 합성 패턴이라 실제 wafer 의 복합/혼재 패턴, tester 조건, ECC 상호작용은 포함하지 않는다.

## 산출물

- `classifier_comparison.csv`, `classifier_per_class_recall.csv`
- `classifier_confusion_{rule_based,random_forest,cnn}.csv`
- `classifier_rf_feature_importance.csv`, `figures/classifier_compare.png`
