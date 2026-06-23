# WM-811K wafer map — lot-group 누수 제거 평가 + 취약클래스 recall 개선

공개 WM-811K wafer map 데이터로 결함 패턴을 분류하되, **같은 lot의 wafer가 train/test에 섞여 평가가 낙관적이 되는 lot leakage**를 제거하는 데 초점을 둔 프로젝트다. accuracy가 아니라 macro-F1과 취약클래스 recall을, 그것도 lot 단위 group split 위에서 본다.

## 문제 정의

- wafer-level random split은 같은 lot의 공정 signature가 train/test로 새어 평가가 낙관적이 될 수 있다.
- accuracy는 none(약 85%) 다수 클래스에 지배되어 변별력이 낮다.
- 따라서 **lot 단위 group split**을 헤드라인으로 두고, macro-F1과 취약클래스(Center / Loc / Scratch) recall로 평가한다.

## 데이터

- 출처: Kaggle, WM-811K Wafer Map Dataset (MIR Lab)
- URL: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
- 원본 파일: `LSWMD.pkl` (약 81만 wafer map, 라벨링 약 17만)
- 원시 데이터는 저장소에 포함하지 않으며, 실행 시 `~/.cache/semiconductor_fdc_yield_analysis/wm811k/`에 캐시한다.
- 제한: `lotName`은 공개 데이터의 proxy이며 실제 fab lot 이력이 아니다. 결과는 원인 확정이 아니라 추가 점검 후보로 해석한다.

## 결과 (seed=42)

| variant | split | accuracy | macro-F1 | Center | Loc | Scratch |
|---|---|---|---|---|---|---|
| random_baseline | random (누수 위험) | 0.925 | 0.584 | 0.505 | 0.156 | 0.112 |
| lotgroup_baseline | lot_group | 0.915 | 0.621 | 0.613 | 0.175 | 0.401 |
| lotgroup_improved | lot_group | 0.937 | **0.675** | **0.765** | 0.228 | 0.173 |

- improved = sqrt sampler + augmentation + focal loss(γ=2). Center · Loc · macro-F1은 단조 개선.
- Scratch(최소 클래스 n=1,193)는 단일 seed 분산이 커(0.40 ↔ 0.17) 개선을 단정하지 않고 다중 seed 검증이 필요함을 정직 보고한다.
- 상세: `reports/wafermap_lotgroup_summary.md`, 그림: `figures/wafermap_lotgroup.png`.

## 실행 방법

```bash
# 1) Kaggle에서 WM-811K(LSWMD.pkl)를 내려받아 캐시 경로에 배치
#    ~/.cache/semiconductor_fdc_yield_analysis/wm811k/raw/LSWMD.pkl
# 2) 전처리 + baseline (processed 캐시 생성)
python scripts/run_wafermap_analysis.py
# 3) lot-group split + 취약클래스 recall 재학습 (seed=42)
python scripts/wafermap_lotgroup.py
```

의존 패키지: numpy, pandas, scikit-learn, torch, matplotlib (저장소 루트 `constraints.txt` 참고).

## 경계 (boundary)

- 공개 WM-811K proxy(`lotName`) 기반이며 실제 fab lot 이력·양산 수율이 아니다.
- 취약클래스 recall 개선은 공개 데이터 재학습 효과이지 양산 검출 보장이 아니다.
- 절대 성능은 데이터 · split · 재학습 설정의 함수이며, 메커니즘과 상대 비교가 핵심이다.
