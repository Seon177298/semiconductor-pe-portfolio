# Digital Thread Source Map

기준일: 2026-05-11

핵심 문장:

> 실제 특정 회사 데이터가 아니라, 공개 digital-thread 사례에서 확인되는 EBOM-MBOM-PBOM-BOP-생산 모니터링 흐름을 참고해 공개 SECOM 품질 데이터 위에 synthetic digital-thread layer를 설계했다.

## Source Boundary

이 프로젝트는 실제 내부 데이터, fab 운영 데이터, Teamcenter export, MES/PLM/ERP DB를 사용하지 않는다. 공개 제조 센서 데이터인 UCI SECOM 위에, 공개 사례에서 확인되는 설계-생산 데이터 흐름을 축소한 synthetic schema를 붙인 포트폴리오 MVP다.

## Public Sources Used

| Source | URL | Used for | Boundary |
| --- | --- | --- | --- |
| Siemens case study: Generic semiconductor design-to-test flow (public concept) | https://en.wikipedia.org/wiki/Semiconductor_device_fabrication | design->mask/process->wafer-test traceability vocabulary (EBOM/MBOM/PBOM/BOP) | 공개 개념 구조만 참고. 실제 fab/PLM 데이터가 아니다. |
| PLM digital-thread concept (public descriptions) | https://en.wikipedia.org/wiki/Digital_thread | EBOM/MBOM/PBOM/BOP design-to-production traceability structure | 공개 개념 구조만 참고. 상용 PLM 플랫폼에 접근한 것이 아니다. |

## How The Sources Become MVP Scope

- EBOM/MBOM/PBOM: 공개 사례의 BOM 분화 개념을 synthetic `bom_items.bom_type`으로 축소했다.
- BOP: PBOM과 생산 resource가 process sequence로 이어지는 구조를 synthetic `bop_steps`로 표현했다.
- Production monitoring: SECOM `sample_id`, model score, alert, review note를 `quality_gates`와 trace endpoint로 연결했다.
- Portfolio evidence: `GET /digital-thread/source-map`, `GET /digital-thread/lots`, `GET /digital-thread/trace`, `GET /digital-thread/lot/{lot_id}`로 read-only 조회만 제공한다.
