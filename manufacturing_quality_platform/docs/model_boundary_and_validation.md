# Model Boundary and Validation

## 역할 분리

- `projects/secom_quality_8d/`: SECOM 분석 원장. random forest, threshold trade-off, CSV report, 8D report 초안이 있는 분석 중심 프로젝트다.
- `projects/manufacturing_quality_platform/`: 운영형 플랫폼 산출물. DB, API, policy, alert history, review note, Streamlit UI를 통해 "분석 결과를 어떻게 운영 흐름으로 연결할지"를 보여준다.

## 현재 Platform Model

- 데이터: UCI SECOM 공개 제조 센서 데이터 1,567개 sample.
- feature: anonymous sensor 중 first 40 sensors만 사용한다.
- 모델: median imputation, standard scaling, class-weighted logistic regression.
- 목적: production-grade 성능 주장보다 threshold 정책, false alarm, missed defect, follow-up workflow를 설명하는 baseline이다.
- 금지: `sensor_000` 같은 anonymous feature를 실제 설비 원인으로 말하지 않는다.

## Validation Rules

- `GET /quality/metrics?threshold=...`는 DB를 변경하지 않는 preview endpoint다.
- `POST /quality/threshold`는 policy row를 만들고, 새 policy_id 기준 alert를 추가한다.
- 기존 alert history는 삭제하지 않는다.
- `GET /alerts`는 기본적으로 active policy의 open alert만 보여준다.
- `GET /alerts?policy_id=...`로 이전 policy의 alert history를 재조회할 수 있다.
- `GET /quality/failure-cases`는 false alarm과 missed defect를 서버에서 계산한다.
- `POST /quality/reviews`는 root cause 확정이 아니라 후속 검토 메모만 저장한다.
- `GET /digital-thread/*` endpoint는 read-only다.
- `process_lots`, `bom_items`, `bop_steps`, `quality_gates`는 모두 `is_synthetic = true`여야 한다.
- 8D report는 review note를 참고하되, 실제 원인 확정처럼 표현하지 않는다.

## Known Limits

- 실제 fab, MES, PLM, ERP 데이터가 아니다.
- 공정 step과 equipment event는 synthetic context다.
- EBOM/MBOM/PBOM/BOP/quality gate는 공개 digital-thread 사례에서 확인되는 구조를 축소한 synthetic digital-thread layer다.
- defect severity, rework cost, operator capacity, maintenance history가 없다.
- 실제 운영 적용에는 label review, drift monitoring, sensor calibration, equipment history가 필요하다.

## Validation Commands

```bash
make seed
make test
make demo-check
```

Expected invariant:

- SECOM sample: 1,567
- quality prediction: 1,567
- quality gate: 1,567 synthetic rows
- Streamlit UI: CSV 직접 읽기 금지, API 호출만 사용
- Digital Thread tab: API만 호출하고 source boundary banner를 표시
- `threshold=0.10`과 `threshold=0.50`의 recall/false alarm 값이 다름
