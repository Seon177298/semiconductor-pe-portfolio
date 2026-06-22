# Data Sources & Licenses

이 저장소의 코드/문서는 루트 `LICENSE`(MIT)를 따릅니다. **MIT는 저자 본인의 코드와 문서에만
적용되며, 아래 third-party 데이터셋에는 적용되지 않습니다.** 원시 데이터는 이 저장소에 포함/재배포하지
않으며(`.gitignore`로 제외), 각 스크립트가 사용 시점에 원 출처에서 내려받습니다. 각 데이터셋은 원
제공자의 약관을 따릅니다.

| Dataset | 용도 | 출처 | 약관 / 비고 |
| --- | --- | --- | --- |
| **UCI SECOM** | 반도체 공정 sensor pass/fail | https://archive.ics.uci.edu/dataset/179/secom | 공개 연구용 데이터셋. feature 익명화 — 실제 공정/설비 원인으로 단정하지 않음. |
| **MVTec AD** (옵션) | 비전 검사 데모 | https://www.mvtec.com/company/research/datasets/mvtec-ad | **비상업 연구·교육용 라이선스.** 이 저장소에 포함하지 않으며, 미설치 시 UI가 "not installed"로 표시. 현장/회사 데이터로 표현하지 않음. |

## 합성(synthetic) 데이터

`manufacturing_quality_platform`의 EBOM/MBOM/PBOM/BOP/quality-gate digital-thread layer는 공개
digital-thread 사례를 참고해 만든 **합성 데이터**이며, 모든 row에 `is_synthetic = true`와 boundary note가
붙어 있습니다. 실제 회사·MES·PLM·ERP 데이터가 아닙니다.

`fail_bit_map_pe`의 fail bit map·물리 margin(Vt/retention/Arrhenius) 모델·guardband/escape-overkill·
disposition 수치는 **합성 데이터**이며 seed 고정으로 재현됩니다(`reports/fbm_dataset.npz`는
재생성물이라 미포함). 비용 단위는 illustrative한 상대값입니다. 실제 fab·계측 데이터가 아닙니다.
SECOM 동기화 부분만 위 표의 공개 UCI SECOM을 사용합니다.

## 경계 (boundary)

- 이 저장소의 모든 분석은 공개 또는 합성 데이터 기반이며, 실제 특정 회사·FAB·라인 실측 데이터가
  아닙니다.
- 모델/분석 결과는 물리적 원인 확정이 아니라 **추가 점검 후보**로 해석합니다.
- SECOM feature는 익명화되어 있어 실제 공정 step·설비·recipe로 단정하지 않습니다.
