"""shmoo.py — E3: 2D Shmoo / operating-window (Vdd × timing × temperature).

합성 데이터. 실제 fab/tester 데이터 아님. seed 고정 재현.

Shmoo 의 핵심: pass/fail 창(operating window)의 **형상**이 불량 메커니즘을 진단한다.
같은 물리 코어(margin = margin0 + γ(Vdd−Vdd_nom) − leak·t·A(T))로 세 FA 가설 die 를 만들어
Vdd × timing(refresh/retention time) × temperature 를 sweep, 각 corner 에서 die 가
구제 가능(disposition ≠ scrap)한지로 pass/fail 을 칠한다.

  - retention-limited : leak 셀 → 고온/긴 timing 에서 fail. 창이 온도에 따라 줄어든다 (Arrhenius).
  - Vdd-limited       : margin0 가 0 근처(파라메트릭) → 저전압에서 fail, 고전압에서 회복. 온도/timing 무관 → 수직 벽.
  - hard structural   : cluster (margin0≪0) → Vdd/timing/온도 무관하게 항상 scrap → operating window 없음.

형상(수평/수직/없음)과 온도 의존성이 곧 신호처리적 경계검출·민감도 특징이고, 그 차이가 다음 FA
액션을 가른다(retention→가속/재측정·refresh, Vdd→트리밍/스크리닝, hard→물리 PFA).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import fbm_core as c

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports"
FIG = REPORT / "figures"

SEED = 5
VDD_GRID = np.round(np.linspace(1.00, 1.20, 11), 3)        # supply axis (V)
TIMING_GRID = np.round(np.linspace(0.5, 2.5, 11), 3)       # normalized retention/refresh time
TEMP_CORNERS_C = [25, 80, 85]                              # field worst = 85 C
HYPOTHESES = ["retention", "vdd", "hard"]
HYP_LABEL = {
    "retention": "retention-limited (leak)",
    "vdd": "Vdd/sensing-limited (parametric)",
    "hard": "hard structural (cluster)",
}


def make_hypothesis_die(kind: str, rng: np.random.Generator):
    """Return (margin0, leak) for one FA hypothesis. Physical, not pattern-injected."""
    margin0 = rng.normal(c.MARGIN0_MEAN, c.MARGIN0_SD, size=(c.GRID, c.GRID))
    leak = np.zeros((c.GRID, c.GRID), dtype=float)

    if kind == "retention":
        n = 400
        idx = rng.choice(c.N_CELLS, size=n, replace=False)
        leak.flat[idx] = rng.uniform(0.004, 0.010, n)        # retention-weak cells
    elif kind == "vdd":
        n = 400
        idx = rng.choice(c.N_CELLS, size=n, replace=False)
        margin0.flat[idx] = rng.normal(-0.02, 0.02, n)       # just-failing, Vdd-recoverable
    elif kind == "hard":
        cy, cx = 70, 70
        yy, xx = np.mgrid[0:c.GRID, 0:c.GRID]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= 11 ** 2
        margin0[mask] = rng.normal(-0.20, 0.02, mask.sum())  # hard cluster, corner-independent
    else:
        raise ValueError(kind)
    return margin0, leak


def die_passes(margin0, leak, temp_C: float, timing: float, vdd: float) -> bool:
    """Is the die salvageable (ship/repair, not scrap) at this (Vdd, timing, T) corner?
    Parametric operating window — no tester noise here (noise/guardband live in run_fbm)."""
    T = temp_C + 273.15
    margin = c.margin_at(margin0, leak, T, t=timing, Vdd=vdd)
    det_fail = margin < 0.0
    return c.repair_disposition(det_fail)["disposition"] != "scrap"


def shmoo_grid(margin0, leak, temp_C, vdd_grid=VDD_GRID, timing_grid=TIMING_GRID):
    """Boolean pass map of shape (len(timing_grid), len(vdd_grid)) at one temperature."""
    out = np.zeros((len(timing_grid), len(vdd_grid)), dtype=bool)
    for i, t in enumerate(timing_grid):
        for j, v in enumerate(vdd_grid):
            out[i, j] = die_passes(margin0, leak, temp_C, t, v)
    return out


def window_area_fraction(margin0, leak, temp_C, vdd_grid=VDD_GRID, timing_grid=TIMING_GRID):
    """Fraction of the (Vdd, timing) grid where the die is salvageable."""
    return float(shmoo_grid(margin0, leak, temp_C, vdd_grid, timing_grid).mean())


def build():
    import pandas as pd

    REPORT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    grids = {}        # (hyp, temp) -> pass map
    rows = []
    for hyp in HYPOTHESES:
        rng = np.random.default_rng(SEED)   # same die per hypothesis across temps
        m0, leak = make_hypothesis_die(hyp, rng)
        for tC in TEMP_CORNERS_C:
            g = shmoo_grid(m0, leak, tC)
            grids[(hyp, tC)] = g
            rows.append({"hypothesis": hyp, "temp_C": tC,
                         "window_area_frac": round(float(g.mean()), 3)})
    df = pd.DataFrame(rows)
    df.to_csv(REPORT / "shmoo_window.csv", index=False)

    save_figure(grids)
    write_summary(df)

    print("=== shmoo (E3) done ===")
    piv = df.pivot(index="hypothesis", columns="temp_C", values="window_area_frac")
    print(piv.to_string())


def save_figure(grids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#c62828", "#2e7d32"])  # fail=red, pass=green
    fig, axes = plt.subplots(len(HYPOTHESES), len(TEMP_CORNERS_C),
                             figsize=(11, 9), sharex=True, sharey=True)
    extent = [VDD_GRID[0], VDD_GRID[-1], TIMING_GRID[0], TIMING_GRID[-1]]
    for r, hyp in enumerate(HYPOTHESES):
        for col, tC in enumerate(TEMP_CORNERS_C):
            ax = axes[r, col]
            g = grids[(hyp, tC)]
            ax.imshow(g.astype(int), origin="lower", aspect="auto", extent=extent,
                      cmap=cmap, vmin=0, vmax=1)
            ax.set_title(f"{tC}°C  (window {g.mean()*100:.0f}%)", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{HYP_LABEL[hyp]}\n\ntiming (retention t)", fontsize=8)
            if r == len(HYPOTHESES) - 1:
                ax.set_xlabel("Vdd (V)", fontsize=9)
    fig.suptitle("2D Shmoo / operating window — pass(green)/fail(red), synthetic\n"
                 "shape diagnoses mechanism: retention=horizontal & T-shrinking, "
                 "Vdd=vertical wall, hard=no window", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "shmoo_operating_window.png", dpi=150)
    plt.close(fig)


def write_summary(df):
    import pandas as pd  # noqa: F401
    piv = df.pivot(index="hypothesis", columns="temp_C", values="window_area_frac")
    tbl = "| FA 가설 | 25°C | 80°C | 85°C | shmoo 형상 | 다음 FA 액션 |\n"
    tbl += "|---|---|---|---|---|---|\n"
    shape = {
        "retention": ("수평 경계, 온도↑/timing↑ 에서 창 축소",
                      "가속(hot) retention 재측정·refresh 강화·온도 guardband"),
        "vdd": ("수직 Vdd 벽, 온도/timing 무관",
                "Vdd/sense trim·스크리닝, 전압 마진 조정"),
        "hard": ("창 없음(전 corner fail)",
                 "파라메트릭 튜닝 무의미 → 물리 PFA/공정 이슈"),
    }
    for hyp in HYPOTHESES:
        s, act = shape[hyp]
        tbl += (f"| {HYP_LABEL[hyp]} | {piv.loc[hyp, 25]:.2f} | {piv.loc[hyp, 80]:.2f} | "
                f"{piv.loc[hyp, 85]:.2f} | {s} | {act} |\n")

    md = f"""# E3 — 2D Shmoo / operating window: 형상이 메커니즘을 진단한다

> **합성 데이터.** 같은 물리 코어(margin = margin0 + γ(Vdd−Vdd_nom) − leak·t·A(T),
> Ea={c.EA_EV}eV, γ={c.GAMMA_VDD}/V, Vdd_nom={c.VDD_NOM}V)로 세 FA 가설 die 를 만들어
> Vdd∈[{VDD_GRID[0]},{VDD_GRID[-1]}]V × timing∈[{TIMING_GRID[0]},{TIMING_GRID[-1]}] × {TEMP_CORNERS_C}°C 를 sweep.
> 각 corner pass = die 가 구제 가능(disposition ≠ scrap). seed={SEED}.

## window 면적(=pass 비율)과 형상 차이

{tbl}

- **retention-limited:** 저온에선 거의 전 영역 pass, 고온에서 창이 줄어든다(온도 의존). 경계는 주로
  **timing(retention) 축에 수평**이고 Vdd 로 약하게 기운다 → 누설/retention 한계의 지문.
- **Vdd-limited:** 창 면적이 온도에 거의 불변(누설 없음), 경계가 **Vdd 축에 수직인 벽** → 파라메트릭
  sensing/drive 마진 한계. Vdd 만 올리면 회복.
- **hard structural:** 어느 corner 에서도 창이 없다 → cluster 가 Vdd/timing/온도와 무관하게 hard fail.
  **파라메트릭 튜닝으로 못 고침**을 즉시 보여주는 신호.

## 신호처리·FA 연결

shmoo 는 (Vdd, timing) 평면의 2D pass/fail 장(field)이고, 진단 정보는 **경계의 방향(수평/수직)·온도에
따른 이동**에 있다 — 경계검출+민감도(∂window/∂T, ∂Vdd_min/∂t) 문제로 볼 수 있다. 형상 하나로 다음
액션이 갈린다: retention→가속/재측정, Vdd→트리밍/스크리닝, hard→물리 PFA. (figure:
`figures/shmoo_operating_window.png`, 표: `reports/shmoo_window.csv`.)

## 금지선

- 합성 die. γ·Vdd_nom·leak 분포는 illustrative 가정. 절대 전압/timing 값은 실제 스펙이 아니다.
- 형상→메커니즘 매핑은 **분석 사고의 구조화**이지 실제 원인 확정이 아니다(실측·PFA 없음).
"""
    (REPORT / "shmoo_summary.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    build()
