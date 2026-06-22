# FA 워크스루 #1 — Column-line Fail (합성 die #22)

> **측정 설계 문서.** 대상 die 는 `synth_fail_bit_map.py` 가 만든 **합성 fail bit map**(die #22, BIN4 COLUMN_FAIL, bad columns=5, total fail bits=659, disposition=scrap)이다.
> 실측 장비 데이터가 없으므로 **원인 가설 → 필요한 전기적 측정 설계 → 확정 전 멈춤 기준**까지만 정직하게 기술한다. 물리 원인을 단정하지 않는다.

## 0. 입력 관찰 (FBM 단계)

- die #22 fail bit map: 5개 column track 이 거의 전길이 fail + 산발 single-bit 19개.
- 자동 분류: COLUMN_FAIL(BIN4). spare column 한도(=4) 초과(5) → redundancy 로 복구 불가 → scrap 후보.
- 같은 wafer/lot 의 다른 die 와 비교해 **동일 column 주소에 반복되는지**가 1차 분기점(systematic vs random).

## 1. 원인 가설 (확정 아님, 후보 목록)

| # | 가설 | 예상 전기적 signature | 복구성 |
|---|---|---|---|
| H1 | Bitline open / 단선 (metal·contact 결함) | 해당 column 전 cell hard fail, Vdd·온도 무관, retention 무관 | spare column 으로 repair (한도 내) |
| H2 | Bitline–bitline short / 인접 coupling | 인접 column 동반 fail, 특정 data pattern(예: checkerboard)에서만 fail | pattern 의존 |
| H3 | Sense amplifier offset / fault | column 단위 margin fail, **Vdd shmoo 에서 경계선**(hard 아님), 온도 의존 | margin/trim 으로 회복 가능성 |
| H4 | Column decoder / column-select(YSEL) stuck | **동일 address-decode 그룹**(물리적으로 인접 아닐 수 있음 — 주소↔물리 매핑 scramble) 동시 fail, address 패턴 의존 | 구조적, repair 한도 의존 |
| H5 | Retention / leakage (cell·junction) | refresh 주기↑·고온에서만 fail, Vdd 의존 약함 | 비복구(누설) |
| H6 | **측정 노이즈 / test escape (실제 fail 아님)** | retest 시 비재현, guardband 내 marginal | FA 대상 아님 → test 조건 review |

## 2. 필요한 전기적 측정 설계

순서는 "싸고 빠른 비파괴 → 비싼 파괴" 로 둔다. 각 단계는 가설을 **배제(rule-out)** 하는 데 쓴다.

1. **Retest / reproducibility (가장 먼저)**
   - 동일 조건 N회(예: 5회) 재측정 + insertion 간 비교. 목적: H6(노이즈/escape) 배제.
   - FBM 의 measurement noise(σ) 와 guardband 설정(본 프로젝트 `cost_model.py`)을 함께 본다 — marginal cell 이면 retest 에서 흔들린다.

2. **Shmoo plot (Vdd × timing, Vdd × 온도)**
   - fail address 를 고정하고 Vdd, access timing(tRCD/tAA 등), 온도를 격자로 sweep → fail 경계 형상.
   - 해석: 경계가 Vdd 에 선형 이동 = margin/sense(H3) ; 전구간 hard = open/short(H1/H2) ; 고온에서만 = retention(H5).

3. **Vdd corner & 온도 corner**
   - (low/nom/high Vdd) × (cold/room/hot). H3(전압 의존)·H5(온도 의존) 분리.

4. **Address / data-pattern 시험**
   - marching(MarchC-), checkerboard, walking 1/0, 인접 column aggressor 패턴.
   - H2(short/coupling)·H4(decoder/YSEL) 는 특정 패턴·주소 묶음에서만 재현.

5. **Retention test**
   - refresh pause / tREF sweep(고온 포함). H5(누설) 확정용 분기.

6. **(probe 단계) continuity / curve-trace**
   - 의심 bitline 의 open/short 를 I–V 로 직접 확인(H1/H2). 가능 시 비파괴 우선.

## 3. 의사결정 트리 (요약)

```
retest 재현 안됨 ───────────────► STOP: test escape/노이즈 → guardband·test 조건 review (FA 종료)
retest 재현
 ├ shmoo: Vdd 선형 이동 + 회복 ─► H3 sense/margin 의심 → trim/repair 검토 (물리 FA 보류)
 ├ shmoo: 전구간 hard fail ─────► H1/H2 open/short 의심 → pattern 시험으로 H1↔H2 분리
 ├ 고온/refresh 의존 ───────────► H5 retention 의심
 └ 주소 묶음 동시 fail ─────────► H4 decoder/YSEL 의심
```

## 4. 확정 전 멈춤 기준 (STOP criteria) — 정직성

- **retest 비재현이면 멈춘다.** 단발 fail 을 물리 결함으로 escalate 하지 않는다(escape/노이즈일 수 있음).
- **전기 측정은 위치·동작 분류까지만**이다. open/short·particle·decoder 같은 **물리 원인은 PFA
  (deprocessing + SEM/TEM/EDS) 로 단면을 보기 전까지 단정하지 않는다.** 전기 FA 는 후보를 좁힐 뿐이다.
- 단일 die 1건으로 lot 전체 systematic 을 주장하지 않는다 — **동일 column 주소의 wafer 간 반복성**이
  확인돼야 systematic, 아니면 random defect 로 둔다.
- 본 문서의 die 는 **합성 데이터**다. 위 측정값은 설계일 뿐 실측이 아니며, 수치 결론은 내지 않는다.

## 5. PE 관점 연결

- escape(이 die 를 ship)했다면 customer column-fail → field return. overkill(양품 scrap)이면 yield loss.
  → 이 die 의 disposition(scrap)과 guardband·비용 최적화는 `cost_model.py` 의 escape/overkill 비용 환산과 직결된다.
- FA 의 목적은 "원인 확정"이 아니라 **다음 조치(repair / trim / test-조건 / PFA 의뢰)를 비용 기준으로
  고르기 위한 증거 좁히기**라는 점을 강조한다.
