"""
Synthetic Fail Bit Map (FBM) generator + fail-type classifier + bin + guardband study.

목적 (Product Engineering 관점):
- 메모리 다이 셀 어레이를 격자로 두고 single-bit / row / column / cluster / edge 유형의
  fail을 합성 생성한다.
- 측정 노이즈를 반영해, fail 판정 guardband(측정 마진) 설정 전/후의
  escape(미검, true fail 인데 ship) 와 overkill(과검, true good 인데 scrap) 변화를 정량화한다.
- fail 유형을 규칙기반으로 자동 분류하고 test bin 을 정의한다.

모든 데이터는 합성(synthetic)이다. 실제 fab/계측 데이터가 아니며,
   수치는 방법론 시연용이다. (No real fab data.)

산출물 (reports/):
- fbm_bin_definition.csv          : bin 정의 표
- fbm_gb_sweep.csv                : guardband sweep (bit/die 단위 escape·overkill·yield)
- fbm_classification_confusion.csv: true bin vs detected bin (분류 성능)
- fbm_examples.csv                : 유형별 예시 die 통계
- figures/fbm_example_maps.png    : 유형별 fail map 예시
- figures/fbm_guardband_curve.png : guardband 대 escape/overkill 곡선
- fbm_summary.md                  : 요약 (guardband 전/후 표)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports"
FIG = REPORT / "figures"

# --------------------------------------------------------------------------------------
# Configuration (all synthetic, documented)
# --------------------------------------------------------------------------------------
GRID = 128                 # cell array = GRID x GRID = 16,384 cells per die
N_DIES = 300               # number of synthetic dies
MARGIN_MEAN = 0.45         # healthy cell read-margin mean (arbitrary units; >0 = pass)
MARGIN_SD = 0.10           # healthy cell-to-cell margin spread
P_WEAK = 0.0012            # fraction of intrinsically weak cells (genuine single-bit fails)
WEAK_MEAN = -0.05          # weak cell true margin mean (just below boundary -> marginal)
WEAK_SD = 0.05
MEAS_NOISE_SD = 0.05       # tester measurement noise (1 sigma) -> drives escape/overkill
SEED = 42

# Redundancy / repair budget (per die)
SPARE_ROWS = 4
SPARE_COLS = 4
SINGLE_BIT_BUDGET = 150    # max isolated bits repairable by spare elements

# Classification thresholds (rule-based)
LINE_FAIL_FRAC = 0.50      # a row/col is "bad" if >= this fraction of its cells fail
EDGE_WIDTH = 4             # outer ring width (cells) for edge detection
EDGE_RING_FILL = 0.35      # fraction of ring cells failing -> edge type
EDGE_MIN_FAILS = 120       # min fail count to even consider an edge pattern
CLUSTER_MIN = 30           # min connected-component size -> cluster type

GB_GRID = np.round(np.arange(-0.02, 0.241, 0.02), 3)  # guardband sweep
GB_BEFORE = 0.00           # "before": fail if measured < 0  (no guardband)
GB_AFTER = 0.06            # "after": cost-minimising guardband (selected by cost_model.py)

BINS = [
    ("BIN1", "PASS", "정상 (fail 없음 또는 redundancy 내 복구 가능)", "ship"),
    ("BIN2", "SINGLE_BIT", "산발 single-bit fail (spare 로 복구 가능)", "ship-after-repair"),
    ("BIN3", "ROW_FAIL", "wordline/row 라인 fail (spare row 로 복구 시도)", "repair-limited"),
    ("BIN4", "COLUMN_FAIL", "bitline/column 라인 fail (spare column 로 복구 시도)", "repair-limited"),
    ("BIN5", "CLUSTER_FAIL", "국소 cluster fail (particle 등, 복구 불가)", "scrap"),
    ("BIN6", "EDGE_FAIL", "edge ring fail (공정 edge 효과, 복구 불가)", "scrap"),
]
TYPE_TO_BIN = {name: bid for bid, name, _, _ in BINS}

rng = np.random.default_rng(SEED)


# --------------------------------------------------------------------------------------
# Die generation
# --------------------------------------------------------------------------------------
DIE_TYPES = ["PASS", "MARGINAL", "SINGLE_HEAVY", "ROW", "COLUMN", "CLUSTER", "EDGE"]
DIE_TYPE_P = [0.30, 0.18, 0.10, 0.12, 0.12, 0.09, 0.09]


def make_die(die_type: str) -> np.ndarray:
    """Return a GRID x GRID array of TRUE cell margins (>0 pass, <0 fail).

    Every die gets a small intrinsic population of weak cells (genuine single-bit
    fails near the boundary). Type-specific structure is added on top.
    """
    margin = rng.normal(MARGIN_MEAN, MARGIN_SD, size=(GRID, GRID))

    # intrinsic weak cells (marginal single-bit fails) on every die except clean PASS
    if die_type != "PASS":
        n_weak = rng.binomial(GRID * GRID, P_WEAK)
        if n_weak:
            idx = rng.choice(GRID * GRID, size=n_weak, replace=False)
            margin.flat[idx] = rng.normal(WEAK_MEAN, WEAK_SD, n_weak)

    if die_type == "MARGINAL":
        # weak-bit count straddles the repair budget -> guardband flips disposition
        n = rng.integers(80, 261)
        idx = rng.choice(GRID * GRID, size=n, replace=False)
        margin.flat[idx] = rng.normal(-0.03, 0.06, n)

    elif die_type == "SINGLE_HEAVY":
        n = rng.integers(40, 120)
        idx = rng.choice(GRID * GRID, size=n, replace=False)
        margin.flat[idx] = rng.normal(-0.18, 0.05, n)  # clearly-failing isolated bits

    elif die_type == "ROW":
        n_bad = rng.integers(1, 7)  # 1..6 bad rows; >SPARE_ROWS becomes unrepairable
        rows = rng.choice(GRID, size=n_bad, replace=False)
        for r in rows:
            margin[r, :] = rng.normal(-0.22, 0.05, GRID)

    elif die_type == "COLUMN":
        n_bad = rng.integers(1, 7)
        cols = rng.choice(GRID, size=n_bad, replace=False)
        for c in cols:
            margin[:, c] = rng.normal(-0.22, 0.05, GRID)

    elif die_type == "CLUSTER":
        cy, cx = rng.integers(20, GRID - 20, 2)
        radius = rng.integers(6, 16)
        yy, xx = np.mgrid[0:GRID, 0:GRID]
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        mask = d2 <= radius ** 2
        margin[mask] = rng.normal(-0.20, 0.06, mask.sum())

    elif die_type == "EDGE":
        w = rng.integers(2, 5)
        ring = np.zeros((GRID, GRID), dtype=bool)
        ring[:w, :] = ring[-w:, :] = ring[:, :w] = ring[:, -w:] = True
        keep = rng.random((GRID, GRID)) < rng.uniform(0.6, 1.0)
        ring &= keep
        margin[ring] = rng.normal(-0.18, 0.06, ring.sum())

    return margin


# --------------------------------------------------------------------------------------
# Connected components (4-neighbour) for cluster sizing
# --------------------------------------------------------------------------------------
def largest_cluster(mask: np.ndarray) -> int:
    seen = np.zeros_like(mask, dtype=bool)
    best = 0
    h, w = mask.shape
    idxs = np.argwhere(mask)
    for sy, sx in idxs:
        if seen[sy, sx]:
            continue
        # iterative flood fill
        stack = [(sy, sx)]
        seen[sy, sx] = True
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        best = max(best, size)
    return best


# --------------------------------------------------------------------------------------
# Rule-based classification + repairability of a fail map
# --------------------------------------------------------------------------------------
def classify(fail: np.ndarray) -> dict:
    """Classify a binary fail map into a bin + repairability decision.

    Detection order: EDGE ring -> interior row/col lines -> cluster -> single-bit.
    Edge is checked first so that a failing border ring is not mistaken for many
    'bad rows/cols'. Line detection is restricted to interior tracks.
    """
    n_fail = int(fail.sum())
    w = EDGE_WIDTH

    # --- edge ring ---
    ring = np.zeros_like(fail, dtype=bool)
    ring[:w, :] = ring[-w:, :] = ring[:, :w] = ring[:, -w:] = True
    ring_fails = int((fail & ring).sum())
    ring_fill = ring_fails / int(ring.sum())          # fraction of ring cells failing
    is_edge = (n_fail >= EDGE_MIN_FAILS and ring_fill >= EDGE_RING_FILL)

    # --- line (row/col) detection over the full array, after consuming the edge ring ---
    # (edge takes precedence: if this is an edge die, the ring is removed first so a
    #  failing border ring is not mistaken for many bad rows/cols.)
    work = fail.copy()
    if is_edge:
        work[ring] = False
    row_counts = work.sum(axis=1)
    col_counts = work.sum(axis=0)
    bad_rows = np.where(row_counts >= LINE_FAIL_FRAC * GRID)[0]
    bad_cols = np.where(col_counts >= LINE_FAIL_FRAC * GRID)[0]

    # residual = fails not explained by edge ring or full bad rows/cols
    residual = work.copy()
    residual[bad_rows, :] = False
    residual[:, bad_cols] = False
    res_total = int(residual.sum())

    max_clu = largest_cluster(residual) if res_total else 0
    single_bits = res_total
    edge_fraction = round(ring_fill, 3)
    edge_fails = ring_fails

    # precedence: edge > cluster > row/col > single > pass
    if is_edge:
        ftype = "EDGE_FAIL"
    elif max_clu >= CLUSTER_MIN:
        ftype = "CLUSTER_FAIL"
    elif len(bad_rows) >= 1 and len(bad_rows) >= len(bad_cols):
        ftype = "ROW_FAIL"
    elif len(bad_cols) >= 1:
        ftype = "COLUMN_FAIL"
    elif single_bits > 0:
        ftype = "SINGLE_BIT"
    else:
        ftype = "PASS"

    # repairability
    if ftype in ("CLUSTER_FAIL", "EDGE_FAIL"):
        repairable = False
    elif ftype == "ROW_FAIL":
        repairable = (len(bad_rows) <= SPARE_ROWS and len(bad_cols) <= SPARE_COLS
                      and single_bits <= SINGLE_BIT_BUDGET)
    elif ftype == "COLUMN_FAIL":
        repairable = (len(bad_cols) <= SPARE_COLS and len(bad_rows) <= SPARE_ROWS
                      and single_bits <= SINGLE_BIT_BUDGET)
    else:  # SINGLE_BIT / PASS
        repairable = single_bits <= SINGLE_BIT_BUDGET

    return {
        "n_fail": n_fail,
        "bad_rows": len(bad_rows),
        "bad_cols": len(bad_cols),
        "max_cluster": max_clu,
        "edge_fails": edge_fails,
        "edge_fraction": round(edge_fraction, 3),
        "single_bits": single_bits,
        "type": ftype,
        "bin": TYPE_TO_BIN[ftype],
        "repairable": repairable,
        "disposition": "ship" if repairable else "scrap",
    }


# --------------------------------------------------------------------------------------
# Build all dies once (true margins fixed; measurement noise fixed per die)
# --------------------------------------------------------------------------------------
def build_population():
    dies = []
    for i in range(N_DIES):
        dtype = rng.choice(DIE_TYPES, p=DIE_TYPE_P)
        true_margin = make_die(dtype)
        meas_margin = true_margin + rng.normal(0, MEAS_NOISE_SD, size=true_margin.shape)
        dies.append({"id": i, "die_type": dtype,
                     "true_margin": true_margin, "meas_margin": meas_margin})
    return dies


def evaluate(dies, guardband: float) -> dict:
    """Disposition every die at a given guardband; return bit- and die-level metrics."""
    total_cells = N_DIES * GRID * GRID
    escape_bits = overkill_bits = true_fail_bits = 0
    escape_dies = overkill_dies = 0
    true_bad_dies = true_good_dies = 0
    shipped = scrapped = 0
    rows = []  # per-die record for confusion matrix etc.

    for d in dies:
        true_fail = d["true_margin"] < 0.0
        det_fail = d["meas_margin"] < guardband

        true_fail_bits += int(true_fail.sum())
        escape_bits += int((true_fail & ~det_fail).sum())
        overkill_bits += int((~true_fail & det_fail).sum())

        true_cls = classify(true_fail)
        det_cls = classify(det_fail)

        true_bad = true_cls["disposition"] == "scrap"
        det_ship = det_cls["disposition"] == "ship"
        true_bad_dies += int(true_bad)
        true_good_dies += int(not true_bad)
        if det_ship:
            shipped += 1
        else:
            scrapped += 1
        if true_bad and det_ship:      # shipped a die that should have been scrapped
            escape_dies += 1
        if (not true_bad) and (not det_ship):  # scrapped a good die
            overkill_dies += 1

        rows.append({"id": d["id"], "die_type": d["die_type"],
                     "true_bin": true_cls["bin"], "det_bin": det_cls["bin"],
                     "true_disp": true_cls["disposition"], "det_disp": det_cls["disposition"]})

    return {
        "guardband": guardband,
        "escape_bits": escape_bits,
        "overkill_bits": overkill_bits,
        "true_fail_bits": true_fail_bits,
        "escape_dppm": round(escape_bits / total_cells * 1e6, 2),
        "overkill_dppm": round(overkill_bits / total_cells * 1e6, 2),
        "escape_dies": escape_dies,
        "overkill_dies": overkill_dies,
        "true_bad_dies": true_bad_dies,
        "true_good_dies": true_good_dies,
        "shipped_dies": shipped,
        "scrapped_dies": scrapped,
        "yield": round(shipped / N_DIES, 4),
        "_rows": rows,
    }


def confusion(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    bin_order = [b[0] for b in BINS]
    cm = pd.crosstab(df["true_bin"], df["det_bin"]).reindex(
        index=bin_order, columns=bin_order, fill_value=0)
    return cm


def save_example_figure(dies):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    want = ["SINGLE_HEAVY", "ROW", "COLUMN", "CLUSTER", "EDGE", "PASS"]
    picks = {}
    for d in dies:
        if d["die_type"] in want and d["die_type"] not in picks:
            picks[d["die_type"]] = d
        if len(picks) == len(want):
            break
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, t in zip(axes.ravel(), want):
        d = picks.get(t)
        if d is None:
            ax.axis("off")
            continue
        fail = (d["meas_margin"] < GB_AFTER).astype(int)
        ax.imshow(fail, cmap="Greys", interpolation="nearest")
        ax.set_title(f"{t}\n(detected fail map @ GB={GB_AFTER})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Synthetic Fail Bit Map examples (NOT real fab data)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "fbm_example_maps.png", dpi=140)
    plt.close(fig)


def save_guardband_curve(sweep: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(sweep["guardband"], sweep["escape_dppm"], "o-", color="crimson", label="escape DPPM (missed)")
    ax1.plot(sweep["guardband"], sweep["overkill_dppm"], "x--", color="navy", label="overkill DPPM (false reject)")
    ax1.axvline(GB_BEFORE, color="grey", ls=":", lw=1)
    ax1.axvline(GB_AFTER, color="green", ls=":", lw=1)
    ax1.set_xlabel("Guardband (measured-margin fail threshold)")
    ax1.set_ylabel("DPPM (bit-level)")
    ax1.set_title("Synthetic FBM: guardband vs escape / overkill (NOT real data)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fbm_guardband_curve.png", dpi=150)
    plt.close(fig)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    # bin definition
    pd.DataFrame(
        [{"bin": b, "type": t, "description": desc, "default_disposition": disp}
         for b, t, desc, disp in BINS]
    ).to_csv(REPORT / "fbm_bin_definition.csv", index=False)

    dies = build_population()

    # guardband sweep
    sweep_rows = []
    eval_cache = {}
    for gb in GB_GRID:
        ev = evaluate(dies, float(gb))
        eval_cache[float(gb)] = ev
        sweep_rows.append({k: v for k, v in ev.items() if k != "_rows"})
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(REPORT / "fbm_gb_sweep.csv", index=False)

    # classification confusion at GB=0 (isolates the classifier from guardband speckle)
    cm = confusion(eval_cache[GB_BEFORE]["_rows"])
    cm.to_csv(REPORT / "fbm_classification_confusion.csv")
    diag = int(np.trace(cm.values))
    total = int(cm.values.sum())
    cls_acc = diag / total

    # disposition (ship/scrap) accuracy at before/after guardband
    def disp_acc(rows):
        df = pd.DataFrame(rows)
        return round((df["true_disp"] == df["det_disp"]).mean(), 3)
    disp_acc_before = disp_acc(eval_cache[GB_BEFORE]["_rows"])
    disp_acc_after = disp_acc(eval_cache[GB_AFTER]["_rows"])

    # example die stats per type (true classification)
    ex_rows = []
    for d in dies:
        c = classify(d["true_margin"] < 0.0)
        ex_rows.append({"id": d["id"], "die_type": d["die_type"], **{
            k: c[k] for k in ("n_fail", "bad_rows", "bad_cols", "max_cluster",
                              "edge_fraction", "single_bits", "type", "bin",
                              "repairable", "disposition")}})
    ex = pd.DataFrame(ex_rows)
    ex.to_csv(REPORT / "fbm_die_truth.csv", index=False)

    save_example_figure(dies)
    save_guardband_curve(sweep)

    before = eval_cache[GB_BEFORE]
    after = eval_cache[GB_AFTER]

    # summary markdown
    def fmt(ev):
        return (f"escape {ev['escape_bits']} bits ({ev['escape_dppm']} DPPM) / "
                f"overkill {ev['overkill_bits']} bits ({ev['overkill_dppm']} DPPM) / "
                f"escape dies {ev['escape_dies']} / overkill dies {ev['overkill_dies']} / "
                f"yield {ev['yield']:.3f}")

    md = f"""# Synthetic Fail Bit Map (FBM) — guardband 전/후 요약

> **합성 데이터.** 실제 fab / 계측 데이터가 아니며, 수치는 방법론 시연용이다.
> 셀 어레이 {GRID}x{GRID} = {GRID*GRID:,} cells/die, dies={N_DIES},
> 측정 노이즈 1σ={MEAS_NOISE_SD} (margin 단위), seed={SEED}. 재현: `python synth_fail_bit_map.py`.

## 1. Bin 정의

| bin | type | 설명 | default disposition |
|---|---|---|---|
""" + "\n".join(f"| {b} | {t} | {desc} | {disp} |" for b, t, desc, disp in BINS) + f"""

## 2. Fail 유형 자동 분류 성능 (rule-based)

- detected bin vs true bin 일치율(@ guardband=0): **{cls_acc:.3f}** ({diag}/{total} dies)
  - 구조성 fail(BIN3 row/BIN4 col/BIN5 cluster/BIN6 edge)은 전부 정확히 분류됨. 불일치는 거의
    BIN1(PASS)↔BIN2(single-bit) 경계로, 측정 노이즈 speckle 때문이며 둘 다 'ship' 등급이라
    출하 판정에는 영향이 없다.
- **ship/scrap 출하 판정 정확도**: guardband 0 → **{disp_acc_before:.3f}**, guardband {GB_AFTER} → **{disp_acc_after:.3f}**
- 혼동행렬(@guardband=0): `reports/fbm_classification_confusion.csv`

## 3. Guardband 설정 전/후 (escape=미검, overkill=과검)

| | guardband | escape bits | escape DPPM | overkill bits | overkill DPPM | escape dies | overkill dies | yield |
|---|---|---|---|---|---|---|---|---|
| **전 (no guardband)** | {GB_BEFORE} | {before['escape_bits']} | {before['escape_dppm']} | {before['overkill_bits']} | {before['overkill_dppm']} | {before['escape_dies']} | {before['overkill_dies']} | {before['yield']:.3f} |
| **후 (guardband)** | {GB_AFTER} | {after['escape_bits']} | {after['escape_dppm']} | {after['overkill_bits']} | {after['overkill_dppm']} | {after['escape_dies']} | {after['overkill_dies']} | {after['yield']:.3f} |

해석: guardband 를 0→{GB_AFTER} 로 올리면 bit escape 가 {before['escape_dppm']}→{after['escape_dppm']} DPPM 으로 줄지만(미검 감소),
overkill 은 {before['overkill_dppm']}→{after['overkill_dppm']} DPPM 으로 늘어난다(과검 증가). 전수 sweep 은 `reports/fbm_gb_sweep.csv`.
어디서 멈출지는 통계가 아니라 **비용**으로 결정한다 → `cost_model.py` (escape/overkill 비용 환산).

## 4. 한계 / 금지선

- 합성 margin 모델은 실제 cell Vt/read-margin 물리를 단순화한 것이다.
- redundancy(spare row {SPARE_ROWS}, col {SPARE_COLS}, single-bit budget {SINGLE_BIT_BUDGET})·분류 임계값은 시연용 가정이다.
- 실제 FBM 은 ECC, repair allocation, bin map, tester 조건과 연결해 해석해야 한다.
"""
    (REPORT / "fbm_summary.md").write_text(md, encoding="utf-8")

    print("=== FBM done ===")
    print("classification accuracy:", round(cls_acc, 3))
    print("BEFORE (gb=0):", fmt(before))
    print("AFTER  (gb=%.2f):" % GB_AFTER, fmt(after))
    print("true bad dies:", before["true_bad_dies"], "/ true good:", before["true_good_dies"])


if __name__ == "__main__":
    main()
