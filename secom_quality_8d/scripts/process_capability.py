"""process_capability.py — 품질공학 toolkit (Cp/Cpk + Gage R&R).

PE/품질 직무의 기본기 — 공정능력(Cpk)과 측정시스템분석(MSA, Gage R&R) — 을 구현한다.

데이터 정직성:
  - Cpk: 공개 UCI SECOM sensor 에 적용하되, **spec limit 은 UCI 가 제공하지 않으므로 pass-class 분포의
    ±3σ 로 illustrative 하게 유도**한다(익명화 데이터 → 실제 규격/공정 단정 금지). 방법 시연이 목적.
  - Gage R&R: SECOM 에는 반복측정 MSA 구조가 없으므로 **합성(synthetic) MSA 연구**(parts×operators×trials)로
    %GRR·ndc 산출 방법을 시연한다. 합성 데이터.
재현: `python scripts/process_capability.py`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


# ---------------------------------------------------------------------------
# Cp / Cpk
# ---------------------------------------------------------------------------
def cp_cpk(values, lsl, usl):
    """Process capability indices from sample mean/std (ddof=1)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    mu = v.mean()
    sigma = v.std(ddof=1)
    if sigma == 0:
        return float("inf"), float("inf")
    cp = (usl - lsl) / (6.0 * sigma)
    cpk = min((usl - mu) / (3.0 * sigma), (mu - lsl) / (3.0 * sigma))
    return float(cp), float(cpk)


# ---------------------------------------------------------------------------
# Gage R&R (balanced crossed two-way ANOVA: part × operator + repeatability)
# ---------------------------------------------------------------------------
def gage_rr_anova(rows):
    """rows: iterable of {part, operator, value}. Balanced design required (every
    part measured the same #trials by every operator). Returns variance-component
    breakdown (EV/AV/GRR/PV/TV), %GRR and ndc (AIAG ANOVA method)."""
    df = pd.DataFrame(list(rows))
    parts = sorted(df["part"].unique())
    opers = sorted(df["operator"].unique())
    p, o = len(parts), len(opers)
    r = len(df) // (p * o)

    grand = df["value"].mean()
    part_mean = df.groupby("part")["value"].mean()
    oper_mean = df.groupby("operator")["value"].mean()
    cell_mean = df.groupby(["part", "operator"])["value"].mean()

    ss_part = o * r * float(((part_mean - grand) ** 2).sum())
    ss_oper = p * r * float(((oper_mean - grand) ** 2).sum())
    ss_int = 0.0
    for pi in parts:
        for oj in opers:
            cm = cell_mean.loc[(pi, oj)]
            ss_int += (cm - part_mean.loc[pi] - oper_mean.loc[oj] + grand) ** 2
    ss_int *= r
    ss_total = float(((df["value"] - grand) ** 2).sum())
    ss_error = ss_total - ss_part - ss_oper - ss_int

    ms_part = ss_part / (p - 1)
    ms_oper = ss_oper / (o - 1) if o > 1 else 0.0
    ms_int = ss_int / ((p - 1) * (o - 1)) if (p > 1 and o > 1) else 0.0
    ms_error = ss_error / (p * o * (r - 1)) if r > 1 else 0.0

    var_repeat = max(ms_error, 0.0)                                  # EV^2
    var_oper = max((ms_oper - ms_int) / (p * r), 0.0) if o > 1 else 0.0   # AV^2
    var_int = max((ms_int - ms_error) / r, 0.0)                     # interaction^2
    var_part = max((ms_part - ms_int) / (o * r), 0.0)              # PV^2

    grr2 = var_repeat + var_oper + var_int
    tv2 = grr2 + var_part
    ev, av, inter, grr, pv, tv = (np.sqrt(x) for x in
                                  (var_repeat, var_oper, var_int, grr2, var_part, tv2))
    pct_grr = 100.0 * grr / tv if tv > 0 else 0.0
    ndc = 1.41 * pv / grr if grr > 0 else float("inf")
    return {"ev": float(ev), "av": float(av), "interaction": float(inter),
            "grr": float(grr), "pv": float(pv), "tv": float(tv),
            "pct_grr": float(pct_grr), "ndc": float(ndc),
            "p": p, "o": o, "r": r}


def make_synthetic_gage_study(rng, n_parts=10, n_operators=3, n_trials=3,
                              part_sd=5.0, operator_sd=0.6, repeat_sd=0.5, center=50.0):
    """Synthetic MSA study with injected part / operator / repeatability variance."""
    rows = []
    part_means = rng.normal(center, part_sd, n_parts)
    op_bias = rng.normal(0.0, operator_sd, n_operators)
    for pi in range(n_parts):
        for oj in range(n_operators):
            for _ in range(n_trials):
                rows.append({"part": pi, "operator": oj,
                             "value": part_means[pi] + op_bias[oj] + rng.normal(0.0, repeat_sd)})
    return rows


# ---------------------------------------------------------------------------
# build: Cpk on SECOM sensors (illustrative specs) + synthetic Gage R&R
# ---------------------------------------------------------------------------
def build():
    from run_analysis import load_data   # lazy: keeps unit-test import light

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    x, y = load_data()

    # pick the most quality-relevant, well-populated sensors (largest standardized
    # mean gap between pass/defect, missing < 5%)
    keep = x.columns[x.isna().mean() < 0.05]
    xk = x[keep]
    g0, g1 = xk[y == 0], xk[y == 1]
    pooled = xk.std(ddof=1).replace(0, np.nan)
    gap = ((g1.mean() - g0.mean()).abs() / pooled).dropna().sort_values(ascending=False)
    sensors = list(gap.head(6).index)

    rows = []
    for s in sensors:
        col = xk[s]
        passv = col[y == 0].dropna()
        lsl = passv.mean() - 3 * passv.std(ddof=1)   # illustrative spec from pass-class ±3σ
        usl = passv.mean() + 3 * passv.std(ddof=1)
        cp, cpk = cp_cpk(col.dropna(), lsl, usl)
        rows.append({"sensor": s, "lsl_illustrative": round(lsl, 4), "usl_illustrative": round(usl, 4),
                     "cp": round(cp, 3), "cpk": round(cpk, 3),
                     "class_gap_sigma": round(float(gap[s]), 3)})
    cpk_df = pd.DataFrame(rows)
    cpk_df.to_csv(REPORT_DIR / "process_capability_cpk.csv", index=False)

    # synthetic Gage R&R MSA study
    study = make_synthetic_gage_study(np.random.default_rng(42))
    grr = gage_rr_anova(study)
    grr_df = pd.DataFrame([grr])
    grr_df.to_csv(REPORT_DIR / "process_capability_gage_rr.csv", index=False)

    save_figure(cpk_df, grr)
    write_summary(cpk_df, grr)

    print("=== process_capability done ===")
    print(cpk_df.to_string(index=False))
    print(f"\nGage R&R (synthetic): %GRR={grr['pct_grr']:.1f}%, ndc={grr['ndc']:.1f} "
          f"(EV={grr['ev']:.3f}, AV={grr['av']:.3f}, PV={grr['pv']:.3f})")


def save_figure(cpk_df, grr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#43a047" if c >= 1.33 else "#fb8c00" if c >= 1.0 else "#e53935" for c in cpk_df["cpk"]]
    bars = ax.bar(cpk_df["sensor"], cpk_df["cpk"], color=colors, alpha=0.85)
    ax.bar_label(bars, fmt="%.2f")
    ax.axhline(1.33, color="green", ls="--", alpha=0.6, label="Cpk 1.33 (capable)")
    ax.axhline(1.0, color="red", ls=":", alpha=0.6, label="Cpk 1.0")
    ax.set_ylabel("Cpk (illustrative pass-class ±3σ specs)")
    ax.set_title("SECOM sensor Cpk — method demo (anonymized data)")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.legend(fontsize=8)

    comp = {"EV\n(repeat)": grr["ev"], "AV\n(operator)": grr["av"],
            "GRR": grr["grr"], "PV\n(part)": grr["pv"]}
    b2 = ax2.bar(list(comp.keys()), list(comp.values()),
                 color=["#e53935", "#fb8c00", "#8e24aa", "#1e88e5"], alpha=0.85)
    ax2.bar_label(b2, fmt="%.2f")
    ax2.set_ylabel("standard deviation (study units)")
    ax2.set_title(f"Synthetic Gage R&R: %GRR={grr['pct_grr']:.1f}%, ndc={grr['ndc']:.1f}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "process_capability.png", dpi=150)
    plt.close(fig)


def write_summary(cpk_df, grr):
    verdict = ("양호(<10%)" if grr["pct_grr"] < 10 else
               "한계(10–30%)" if grr["pct_grr"] <= 30 else "부적합(>30%)")
    cpk_tbl = "| sensor | Cp | Cpk | class gap(σ) | LSL~USL (illustrative) |\n|---|---|---|---|---|\n"
    for r in cpk_df.itertuples():
        cpk_tbl += f"| {r.sensor} | {r.cp} | {r.cpk} | {r.class_gap_sigma} | {r.lsl_illustrative}~{r.usl_illustrative} |\n"

    md = f"""# 공정능력(Cpk) · 측정시스템분석(Gage R&R)

> **정직성:** Cpk 의 spec limit 은 UCI SECOM 이 제공하지 않아 **pass-class 분포 ±3σ 로 illustrative 유도**
> (익명화 데이터 → 실제 규격/공정 단정 금지). Gage R&R 은 SECOM 에 MSA 구조가 없어 **합성 MSA 연구**.
> 둘 다 **방법(품질공학 toolkit) 시연**이지 실측 성능 보고가 아니다.

## 1. Cp/Cpk — quality-relevant SECOM sensor (방법 시연)

pass/defect 표준화 평균차가 큰 sensor 6개에 대해, pass-class ±3σ 를 규격으로 두고 전체 분포의 공정능력을 계산.

{cpk_tbl}

- Cpk < 1.0 인 sensor 는 (illustrative 규격 기준) 전체 분포가 pass 중심에서 벗어나 능력이 낮음 → defect 와 연관된
  공정 산포 후보. (class gap(σ) 가 클수록 defect 와의 분리가 큼.)
- **해석:** Cpk = min(Cpu, Cpl). 규격은 illustrative 이므로 절대 수치가 아니라 **상대 비교·방법**으로 읽는다.

## 2. Gage R&R (synthetic MSA) — 측정시스템이 부품 변동을 구분하는가

parts×operators×trials 합성 MSA 를 crossed ANOVA 로 분해:

| 성분 | 표준편차 |
|---|---|
| EV (repeatability) | {grr['ev']:.3f} |
| AV (reproducibility, operator) | {grr['av']:.3f} |
| interaction | {grr['interaction']:.3f} |
| **GRR (EV⊕AV⊕int)** | **{grr['grr']:.3f}** |
| PV (part) | {grr['pv']:.3f} |
| TV (total) | {grr['tv']:.3f} |

- **%GRR = {grr['pct_grr']:.1f}%** → AIAG 기준 **{verdict}**. **ndc = {grr['ndc']:.1f}** (≥5 권장: 측정계가 부품 등급을 구분 가능).
- %GRR 은 측정변동(GRR)이 전체변동(TV)에서 차지하는 비율 — 클수록 측정계가 부품 차이를 못 가린다.

## 금지선

- Cpk 규격·Gage R&R 데이터는 illustrative/synthetic. 결론은 **방법과 상대 비교**이지 실제 공정능력·MSA 결과가 아니다.
- (figure: `figures/process_capability.png`.)
"""
    (REPORT_DIR / "process_capability_summary.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    build()
