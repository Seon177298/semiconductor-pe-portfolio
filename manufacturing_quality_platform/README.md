# Manufacturing Quality Platform

공개 SECOM 품질 데이터를 입력으로 받아 `SQLite -> FastAPI -> Streamlit -> alert -> 8D report` 흐름을 구현한 미니 품질 플랫폼이다. 모델 점수에서 끝내지 않고, threshold를 바꿀 때 yield(출하율), bin(점수대 분포), escape(미검 출하), overkill(양품 과검)이 운영자 화면에서 어떻게 움직이는지 보여주는 것이 목표다. 여기에 EBOM/MBOM/PBOM/BOP process-traceability를 흉내 낸 synthetic digital-thread 계층을 보조로 붙여 `sample -> lot -> BOM -> BOP -> quality gate -> prediction -> alert -> review note` trace를 제공한다.

## 무엇을 하는가

- 데이터: UCI SECOM 제조 센서 샘플 1,567개를 DB에 적재한다. 샘플마다 익명 sensor 중 first 40 sensor를 `sensor_readings`에 저장한다.
- 모델: first-40-sensor 기반 baseline logistic regression(median imputation, standard scaling, class-weighted). production-grade 성능을 주장하는 모델이 아니라 threshold와 failure-case 동작을 드러내기 위한 baseline이다.
- yield/bin/escape-overkill: `GET /quality/yield-summary?threshold=...`가 yield, 점수대 bin 분포(bin별 true-defect rate 포함), escape, overkill을 DB 변경 없이 미리 계산한다.
- threshold 정책: `GET /quality/metrics`는 read-only preview이고, `POST /quality/threshold`는 새 policy를 만들고 해당 policy 기준 alert를 append한다. 이전 policy history는 삭제하지 않는다.
- failure case: false alarm과 missed defect를 서버에서 계산해 검토 큐로 제공한다. `POST /quality/reviews`는 후속 검토 메모만 저장하며 root cause 확정이 아니다.
- digital-thread trace: 하나의 SECOM sample을 lot, EBOM/MBOM/PBOM, BOP, quality gate, prediction, alert, review note로 연결한다.
- 8D report: alert 기반으로 8D report draft를 생성한다.
- UI: Streamlit 대시보드가 CSV를 직접 읽지 않고 API를 호출한다.
- 옵션: MVTec AD bottle 상태 패널(데이터 미설치 시 명시적으로 표시).

## Threshold 동작

threshold를 낮추면 recall은 올라가지만 false alarm이 급증하고, 높이면 missed defect가 생긴다.

| threshold | precision | recall | false alarm | false alarm count | missed defect count |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 0.069 | 1.000 | 0.952 | 1393 | 0 |
| 0.50 | 0.126 | 0.683 | 0.338 | 494 | 33 |

- `threshold=0.10`: 결함을 놓치지 않지만 정상 1,463개 중 1,393개가 알람으로 잡혀 운영자가 감당하기 어렵다.
- `threshold=0.50`: false alarm은 494개로 줄지만 결함 104개 중 33개를 놓친다.

핵심은 모델 점수 자체가 아니라 `threshold preview -> policy apply -> alert refresh -> 8D follow-up` 흐름이다.

## 아키텍처

```text
UCI SECOM public data
  -> seed script
  -> SQLite
  -> FastAPI
  -> Streamlit dashboard
  -> alert policy history
  -> failure review note
  -> 8D report
```

주요 테이블:

- `production_lines`, `equipment`: sensor reading/alert join을 위한 virtual handle. 실제 설비가 아니다.
- `sensor_readings`: 샘플당 first 40 익명 SECOM sensor.
- `quality_predictions`: logistic regression defect score와 SECOM true label.
- `threshold_policies`: threshold policy history(precision, recall, F1, false alarm, missed defect, confusion matrix).
- `alerts`: `policy_id`에 연결된 append-only alert.
- `quality_reviews`: false alarm/missed defect 후속 검토 메모.
- `production_orders`, `process_steps`, `equipment_events`: synthetic operations context.
- `process_lots`, `bom_items`, `bop_steps`, `quality_gates`: synthetic digital-thread row.
- `eight_d_reports`: 생성된 8D report draft.

## 문서

- [Architecture](docs/architecture.md)
- [Model boundary and validation](docs/model_boundary_and_validation.md)
- [Digital thread source map](docs/digital_thread_source_map.md)
- [Digital thread schema](docs/digital_thread_schema.md)
- [Threshold comparison](docs/threshold_comparison.md)

## 프로젝트 구조

```text
manufacturing_quality_platform/
  app/
    db.py
    main.py
    seed.py
  docs/
  scripts/
    seed_database.py
  ui/
    streamlit_app.py
  tests/
  docker-compose.yml
```

## 실행

로컬:

```bash
cd manufacturing_quality_platform
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
make seed
uvicorn app.main:app --reload --port 8000
```

다른 터미널에서 UI 실행:

```bash
cd manufacturing_quality_platform
. .venv/bin/activate
API_BASE_URL=http://localhost:8000 streamlit run ui/streamlit_app.py
```

Docker:

```bash
cd manufacturing_quality_platform
docker compose up --build
```

접속:

- API docs: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501

## API

- `GET /health`
- `POST /ingest/sensor-batch`
- `GET /lines`
- `GET /equipment/{equipment_id}/readings`
- `GET /quality/predictions`
- `GET /quality/yield-summary?threshold=0.50` — yield / bin / escape-overkill
- `GET /quality/metrics?threshold=0.10` — read-only preview
- `POST /quality/threshold` — policy 생성 및 alert append
- `GET /quality/policies`
- `GET /alerts`, `GET /alerts?policy_id=...`
- `GET /quality/failure-cases?threshold=0.50&case_type=false_alarm`
- `GET /quality/failure-cases?threshold=0.50&case_type=missed_defect`
- `POST /quality/reviews`
- `GET /operations/events`
- `GET /digital-thread/source-map`
- `GET /digital-thread/lots`
- `GET /digital-thread/trace?sample_id=1`
- `GET /digital-thread/lot/{lot_id}`
- `POST /reports/8d`
- `GET /reports/8d/{report_id}`
- `GET /vision/status`

`GET /quality/metrics`는 threshold policy를 삽입하지 않고 alert를 갱신하지 않는 preview endpoint다. `POST /quality/threshold`는 새 policy를 기록하고 이전 history를 지우지 않은 채 alert를 추가하는 operations endpoint다. `GET /alerts`는 기본적으로 active policy의 open alert를 반환하며, `policy_id`로 이전 history를 조회한다. `POST /quality/reviews`가 저장하는 메모는 확정된 root cause가 아니다.

### Digital thread trace 예시

`GET /digital-thread/trace?sample_id=1`은 SECOM sample 하나를 lot, BOM, BOP, quality gate, prediction, alert, review로 잇는다. 응답 요약:

```jsonc
{
  "boundary_note": "...synthetic digital-thread layer over UCI SECOM...",
  "prediction": {
    "sample_id": 1,
    "equipment_id": "EQP-SECOM-001",
    "model_name": "logistic_regression_first_40_sensors",
    "defect_score": 0.498,
    "true_defect": 0
  },
  "lot": { "lot_id": "LOT-LOGIC-001", "lot_type": "logic_wafer_lot", "is_synthetic": true },
  "bom_items": "EBOM×3, MBOM×3, PBOM×2",
  "bop_steps": 4,
  "quality_gate": { "sample_id": 1 },
  "active_alert": "...", "latest_review": "...", "eight_d_report_candidate": "..."
}
```

모든 digital-thread row는 `is_synthetic: true`와 `boundary_note`를 포함한다.

### Streamlit UI 탭

| 탭 | 내용 |
|---|---|
| Overview | production line, 최근 alert, synthetic operations context |
| Sensor | SECOM sensor readings 조회 |
| Quality | yield / bin / escape-overkill + threshold metrics preview, policy history |
| Failure | false alarm / missed defect 검토 큐 |
| Digital Thread | lot, EBOM/MBOM/PBOM, BOP, quality gate trace |
| Vision | MVTec AD bottle 상태 (옵션, 미설치 시 명시) |
| Report | alert 기반 8D report |

MVTec AD는 옵션이다. `data/mvtec/bottle/`가 없으면 UI가 `MVTec data not installed`를 표시하며 SECOM 기능은 그대로 동작한다. MVTec AD는 비상업적 연구·교육용 라이선스다.

## 한계

실제 현장 적용에는 설비 이력, 공정 step, 작업 조건, defect severity, rework cost, label review, sensor drift/calibration 정보가 추가로 필요하다. 이 프로젝트는 그러한 데이터가 있었다면 어떤 테이블/API/UI 연결이 필요한지를 read-only MVP로 보여주는 데 목적이 있다.

## 테스트

```bash
cd manufacturing_quality_platform
make seed
make test
make demo-check
```
