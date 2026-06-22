# Architecture

## System Flow

```text
UCI SECOM public data
  -> seed script
  -> SQLite
  -> FastAPI
  -> Streamlit
  -> synthetic digital thread trace
  -> alert policy history
  -> failure review note
  -> 8D report
```

This is a public-data portfolio platform. It is not field data, fab data, MES, PLM, or ERP operation. The digital-thread layer is a public-case-informed synthetic schema over SECOM samples.

## Database Map

- `production_lines`: virtual public SECOM line metadata.
- `equipment`: virtual equipment handles for joining sensor readings and alerts.
- `sensor_readings`: first 40 anonymous SECOM sensors per sample.
- `quality_predictions`: logistic regression defect score and true SECOM label.
- `threshold_policies`: threshold policy history with precision, recall, F1, false alarm, missed defect, and confusion matrix counts.
- `alerts`: append-only alert rows linked to `policy_id`.
- `quality_reviews`: false alarm/missed defect follow-up review notes.
- `production_orders`: synthetic production order context.
- `process_steps`: synthetic process step context.
- `equipment_events`: synthetic equipment event log.
- `process_lots`: synthetic wafer-lot handles.
- `bom_items`: synthetic EBOM/MBOM/PBOM rows linked to blocks.
- `bop_steps`: synthetic bill-of-process rows linked to blocks.
- `quality_gates`: synthetic quality gates linking SECOM `sample_id` to block and BOP context.
- `eight_d_reports`: generated 8D follow-up report drafts.

## API Map

- `GET /health`: service health.
- `POST /ingest/sensor-batch`: demo sensor reading ingestion.
- `GET /lines`: production line summary.
- `GET /equipment/{equipment_id}/readings`: sensor reading lookup.
- `GET /quality/predictions`: prediction lookup.
- `GET /quality/metrics?threshold=...`: read-only threshold preview.
- `POST /quality/threshold`: create threshold policy and append alerts for that policy.
- `GET /quality/policies`: active and previous policy history.
- `GET /alerts`: active policy open alerts by default.
- `GET /alerts?policy_id=...`: historical policy alerts.
- `GET /quality/failure-cases?threshold=...&case_type=false_alarm|missed_defect`: server-side failure-case lookup.
- `POST /quality/reviews`: store follow-up review note.
- `GET /operations/events`: synthetic production order, process step, and equipment event context.
- `GET /digital-thread/source-map`: public source map and boundary note.
- `GET /digital-thread/lots`: synthetic lot list with EBOM/MBOM/PBOM/BOP counts.
- `GET /digital-thread/trace?sample_id=...`: SECOM sample to lot, BOM, BOP, quality gate, prediction, alert, review trace.
- `GET /digital-thread/lot/{lot_id}`: lot-centric EBOM/MBOM/PBOM/BOP/quality gate detail.
- `POST /reports/8d`: create 8D report draft.
- `GET /reports/8d/{report_id}`: read 8D report draft.
- `GET /vision/status`: optional MVTec status check.

## UI Tab Map

- Overview: line count, prediction count, active alerts, recall, false alarms, recent alerts, synthetic operations context.
- Sensor Data: SECOM sensor readings by virtual equipment.
- Quality Model: threshold metrics, confusion summary, top defect scores, policy comparison.
- Failure Cases: false alarm/missed defect endpoint results and review note entry.
- Digital Thread: source boundary, lot selector, EBOM/MBOM/PBOM comparison, BOP steps, quality gates, sample trace.
- Vision Inspection: optional MVTec AD status.
- 8D Report: alert selection and 8D report generation.

## Boundary Statements

- Virtual line and equipment names are DB join handles.
- Synthetic operations context is included to show which real records would be needed later.
- Synthetic digital-thread rows are not records; they are deterministic portfolio seed rows informed by public case structure.
- Review notes are follow-up notes, not root-cause confirmation.
