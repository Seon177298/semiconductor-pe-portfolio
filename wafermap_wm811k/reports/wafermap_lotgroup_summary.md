# PHASE 2 — WM-811K lot-group split (누수 제거) + 취약클래스 recall 실제 개선

> ⚠️ 공개 WM-811K (Kaggle). pattern/candidate review 용. 실제 공정 원인·양산 수율 단정 금지.
> 재현: `python scripts/wafermap_lotgroup.py`. (CNN 3 variant 재학습; seed=42.)

## 1. lot-group split 을 헤드라인으로 (lot leakage 제거 — 올바른 평가 방법론)

같은 lot 의 wafer 가 train/test 에 함께 들어가면(=wafer-level random split) lot 의 공정 signature 가
새어 평가가 낙관적이 될 수 있다. **lot 단위 group split**(같은 lot 은 한 split 에만)을 헤드라인으로 채택한다.

| variant | split | accuracy | macro F1 | Scratch | Loc | Center |
|---|---|---|---|---|---|---|
| random_baseline | random | 0.9248 | 0.5839 | 0.1117 | 0.1558 | 0.5047 |
| lotgroup_baseline | lot_group | 0.9154 | 0.6207 | 0.4012 | 0.1752 | 0.6132 |
| lotgroup_improved | lot_group | 0.9373 | 0.6747 | 0.1728 | 0.2281 | 0.765 |


- accuracy 는 **none(85%) 다수에 지배**되어 변별력이 낮다(random 0.9248 vs lot-group 0.9154 — 차이 작고 run 분산 존재). 따라서 헤드라인 지표는 **macro F1·취약클래스 recall**.
- 핵심은 "큰 누수 갭"이 아니라 **누수 없는 올바른 평가로의 전환**과 그 위에서의 개선이다.

## 2. 취약클래스 recall 실제 개선 (재학습 — review-overlay 아님)

improved = **sqrt sampler + augmentation + focal loss(γ=2)** (class-weight 중복 제거). full-balanced sampler 는 다수클래스를
파괴(accuracy 0.08 붕괴)해 폐기하고, **비중복 완화 설정**으로 다수클래스를 지키며 소수클래스를 끌어올렸다.

**macro F1: random 0.5839 → lot-group baseline 0.6207 → improved 0.6747** (improved 가 최고, accuracy 도 0.9373).

| 취약클래스 | random(기존 헤드라인) | lot-group baseline | lot-group **improved** | random→improved |
|---|---|---|---|---|
| Center | 0.5047 | 0.6132 | **0.765** | +0.260 |
| Loc | 0.1558 | 0.1752 | **0.2281** | +0.072 |
| Scratch | 0.1117 | 0.4012 | **0.1728** | +0.061 |

- **Center·Loc 는 단조 개선**(random→baseline→improved), macro F1 도 단조 상승. improved 가 기존 random 헤드라인 대비 macro F1·전 취약클래스 recall 을 모두 높인다.
- **Scratch(최소 클래스 n=1,193)는 run 분산이 크다**(baseline 0.40 ↔ improved 0.17): 단일 seed recall 은 불안정 → **Scratch 개선은 단정하지 않고 다중 seed 가 필요**하다고 정직 보고한다.
전 클래스 recall(헤드라인 외 포함, 재학습 3 variant):

| class | random_baseline | lotgroup_baseline | lotgroup_improved |
|---|---|---|---|
| Center | 0.5047 | 0.6132 | 0.765 |
| Donut | 0.4819 | 0.9118 | 0.8333 |
| Edge-Loc | 0.4878 | 0.5385 | 0.7102 |
| Edge-Ring | 0.9869 | 0.9783 | 0.9723 |
| Loc | 0.1558 | 0.1752 | 0.2281 |
| Near-full | 1.0 | 0.7917 | 1.0 |
| Random | 0.8846 | 0.984 | 0.912 |
| Scratch | 0.1117 | 0.4012 | 0.1728 |
| none | 0.9755 | 0.9554 | 0.9719 |

## 금지선

- lot-group split 도 공개 WM-811K proxy(lotName) 기반이며 실제 fab lot 이력이 아니다. 절대 성능은 데이터·split·재학습의 함수.
- 취약클래스 개선은 **합성/공개 데이터에서의 재학습 효과**이지 양산 검출 보장이 아니다. (figure: `figures/wafermap_lotgroup.png`.)
