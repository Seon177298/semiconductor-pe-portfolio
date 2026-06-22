# Digital Thread Schema

기준일: 2026-05-11

## Purpose

`manufacturing_quality_platform`은 UCI SECOM 공개 품질 데이터를 DB/API/UI workflow로 연결한다. Digital thread MVP는 여기에 synthetic process-traceability layer를 추가해 `sample -> lot -> BOM -> BOP -> quality gate -> prediction -> alert -> review note` 흐름을 보여준다.

이 계층은 실제 특정 회사 데이터가 아니다. 모든 row는 `is_synthetic = true`와 source boundary note를 가진다.

## Tables

### `process_lots`

- `lot_id`: synthetic wafer-lot key.
- `device_family`: synthetic device-family handle.
- `lot_type`: logic / dram / nand 등 synthetic wafer-lot 분류.
- `source_note`: public-case-informed synthetic boundary.
- `is_synthetic`: always true.

### `bom_items`

- `bom_item_id`: synthetic BOM item key.
- `bom_type`: `EBOM`, `MBOM`, `PBOM` 중 하나.
- `lot_id`: linked synthetic wafer lot.
- `part_code`, `part_name`, `revision`, `quantity`, `parent_part_code`: deterministic seed values.
- `source_note`, `is_synthetic`: source boundary fields.

### `bop_steps`

- `bop_step_id`: synthetic process step key.
- `lot_id`: linked synthetic wafer lot.
- `step_sequence`: operation order.
- `operation_name`: reduced operation such as work package release, fit-up, inspection hold point.
- `resource_type`: planning, assembly cell, quality station, production control.
- `expected_minutes`: deterministic planning duration.
- `source_note`, `is_synthetic`: source boundary fields.

### `quality_gates`

- `gate_id`: synthetic quality gate key.
- `sample_id`: UCI SECOM sample linked to this digital-thread trace.
- `lot_id`, `bop_step_id`: deterministic synthetic mapping.
- `gate_name`: SECOM-linked synthetic quality gate.
- `check_type`: anonymous sensor model review.
- `quality_signal`: one of `model_defect_candidate`, `missed_defect_review_candidate`, `routine_pass_monitoring`.
- `source_note`, `is_synthetic`: source boundary fields.

## Seed Rules

- SECOM sample count remains 1,567.
- `quality_predictions` count remains 1,567.
- One `quality_gates` row is created for each SECOM sample.
- Synthetic blocks, BOM items, and BOP steps are deterministic seed data.
- No write API is added for the digital-thread MVP.

## API Contracts

- `GET /digital-thread/source-map`: source boundary and public source map.
- `GET /digital-thread/lots`: lot list with EBOM/MBOM/PBOM/BOP/quality gate counts.
- `GET /digital-thread/lot/{lot_id}`: full lot-centric BOM, BOP, and quality gate view.
- `GET /digital-thread/trace?sample_id=...`: sample-centric trace with `lot`, `bom_items`, `bop_steps`, `quality_gate`, `prediction`, `active_alert`, `latest_review`, and `eight_d_report_candidate`.
