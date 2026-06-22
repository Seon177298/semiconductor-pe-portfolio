"""kgd_stacking.py — E4: HBM Known-Good-Die (KGD) stacking-yield amplification.

합성 데이터. die-escape 확률은 fail_bit_map_pe 의 guardband sweep
(`reports/fbm_v2_gb_sweep.csv`)에서 가져오고, 그것을 n-die stack 으로 곱셈 환산한다.

핵심 물리: HBM 큐브는 core die N 장을 TSV 로 적층한다. 적층 후 한 장이라도
true-bad 면 큐브 전체가 손실(또는 final stack test fail). 따라서 die 단위 escape 는
stack 높이 K 에 대해 1-(1-p_esc)^K 로 증폭된다 — die test 의 KGD 선별이 stack yield 를
지배하는 이유. 여기서는 우리 guardband 선별(전/후)이 큐브 yield·escape 를 얼마나 바꾸는지
정량화한다. (per-die intrinsic stacking/TSV yield 는 별도 가정으로 분리.)
"""
from __future__ import annotations


def stack_good_prob(p_escape: float, height: int) -> float:
    """P(all K core dies truly good) = (1 - p_escape)^K, the KGD yield-multiplication."""
    return (1.0 - p_escape) ** height


def cube_escape_prob(p_escape: float, height: int) -> float:
    """P(cube contains >=1 latent bad die) = 1 - (1 - p_escape)^K."""
    return 1.0 - stack_good_prob(p_escape, height)


def rule_of_three_upper(n: int) -> float:
    """95% upper bound on a rate when 0 events were observed in n trials (~3/n)."""
    return 3.0 / n


# --------------------------------------------------------------------------------------
# Report / figure builder (orchestration; the math above is unit-tested)
# --------------------------------------------------------------------------------------
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports"
FIG = REPORT / "figures"

HEIGHTS = [4, 8, 12, 16]      # HBM2/2E .. HBM3/3E core-die stack heights
REF_GOOD_DIE = 0.01           # canonical KGD reference: 0.99 good per die -> 0.99^K
HEADLINE_K = 12               # 12-high cube for the headline


def _escape_rate_among_shipped(row):
    """Of the dies we ship as KGD (ship + repair), the fraction that are truly bad.
    This outgoing-defective rate is what propagates into the stack."""
    shipped = int(row["ship"] + row["repair"])
    return int(row["escape_dies"]), shipped, int(row["escape_dies"]) / shipped


def build():
    import pandas as pd

    sweep = pd.read_csv(REPORT / "fbm_v2_gb_sweep.csv")
    cost_gb = float(sweep.loc[sweep["total_cost"].idxmin(), "guardband"])
    before = sweep[sweep["guardband"] == 0.0].iloc[0]
    after = sweep[sweep["guardband"] == cost_gb].iloc[0]

    esc_b, ship_b, p_before = _escape_rate_among_shipped(before)
    esc_a, ship_a, p_after_pt = _escape_rate_among_shipped(after)
    # cost-opt screen ships 0 escapes in this sample -> honest non-zero 95% ceiling
    p_after_ub = rule_of_three_upper(ship_a) if esc_a == 0 else p_after_pt

    rows = []
    for K in HEIGHTS:
        rows.append({
            "stack_height": K,
            "unscreened_cube_good": round(stack_good_prob(p_before, K), 4),
            "unscreened_cube_escape": round(cube_escape_prob(p_before, K), 4),
            "kgd_cube_good_point": round(stack_good_prob(p_after_pt, K), 4),
            "kgd_cube_escape_ub95": round(cube_escape_prob(p_after_ub, K), 4),
            "ref_0p99die_cube_good": round(stack_good_prob(REF_GOOD_DIE, K), 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(REPORT / "kgd_stacking.csv", index=False)

    save_figure(df, p_before, p_after_ub, cost_gb)
    write_summary(df, before, after, cost_gb, esc_b, ship_b, p_before,
                  esc_a, ship_a, p_after_ub)

    h = df[df["stack_height"] == HEADLINE_K].iloc[0]
    print("=== kgd_stacking (E4) done ===")
    print(f"per-KGD-die escape: unscreened {p_before*100:.2f}% ({esc_b}/{ship_b}) "
          f"-> cost-opt(gb={cost_gb}) {p_after_pt*100:.2f}% point "
          f"/ <= {p_after_ub*100:.2f}% (95% UB, {esc_a}/{ship_a})")
    print(f"{HEADLINE_K}-high cube all-good: unscreened {h.unscreened_cube_good:.3f} "
          f"-> KGD >= {1-h.kgd_cube_escape_ub95:.3f} (95% worst); ref 0.99^{HEADLINE_K}="
          f"{h.ref_0p99die_cube_good:.3f}")


def save_figure(df, p_before, p_after_ub, cost_gb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    K = df["stack_height"]
    ax.plot(K, df["unscreened_cube_good"], "o-", color="crimson",
            label=f"unscreened KGD (die escape {p_before*100:.1f}%)")
    ax.plot(K, 1 - df["kgd_cube_escape_ub95"], "s-", color="green",
            label=f"cost-opt KGD screen (95% worst, gb={cost_gb})")
    ax.plot(K, df["ref_0p99die_cube_good"], "^--", color="gray",
            label="reference: 0.99 good/die")
    ax.set_xlabel("HBM core-die stack height K")
    ax.set_ylabel("P(cube all-good) = (1 - p_escape)^K")
    ax.set_title("KGD selection vs stack yield (synthetic)")
    ax.set_xticks(list(K))
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # cube-escape exposure per 1000 finished cubes at the headline height
    h = df[df["stack_height"] == HEADLINE_K].iloc[0]
    bars = ax2.bar(["unscreened", "cost-opt KGD\n(95% worst)"],
                   [h.unscreened_cube_escape * 1000, h.kgd_cube_escape_ub95 * 1000],
                   color=["crimson", "green"], alpha=0.85)
    ax2.bar_label(bars, fmt="%.0f")
    ax2.set_ylabel(f"cubes carrying a latent bad die / 1000 ({HEADLINE_K}-high)")
    ax2.set_title("Stack-level escape exposure (synthetic)")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG / "kgd_stack_yield.png", dpi=150)
    plt.close(fig)


def write_summary(df, before, after, cost_gb, esc_b, ship_b, p_before,
                  esc_a, ship_a, p_after_ub):
    h = df[df["stack_height"] == HEADLINE_K].iloc[0]
    tbl = "| K | unscreened all-good | unscreened cube-escape | KGD all-good (95% worst) | ref 0.99^K |\n"
    tbl += "|---|---|---|---|---|\n"
    for _, r in df.iterrows():
        tbl += (f"| {int(r.stack_height)} | {r.unscreened_cube_good:.3f} | "
                f"{r.unscreened_cube_escape:.3f} | {1-r.kgd_cube_escape_ub95:.3f} | "
                f"{r.ref_0p99die_cube_good:.3f} |\n")

    md = f"""# E4 — HBM KGD stacking yield: die escape를 stack 곱셈으로 환산

> **합성 데이터.** die-escape 확률은 `reports/fbm_v2_gb_sweep.csv`(guardband sweep)에서,
> stack 환산은 (1−p_escape)^K. 합성 population 의 불량률(400/1500)이 높아 **절대 DPPM 은 illustrative**이며,
> 핵심은 (1) 곱셈 증폭 메커니즘과 (2) KGD 선별의 상대 임팩트다.

## 1. die test(KGD 선별)가 stack yield 를 지배하는 이유

HBM 큐브는 core die {HEADLINE_K}장을 TSV 로 적층한다. 적층 후 한 장이라도 true-bad 면 큐브 전체가
손실(또는 final stack test fail) → die 단위 escape p 는 stack 높이 K 에서 **1−(1−p)^K 로 증폭**된다.
die test 에서 새는 한 장이 die 한 개가 아니라 **완성 큐브 한 개**(core die {HEADLINE_K}장 + buffer + TSV
assembly + stack test)를 버리게 만든다 — KGD(Known-Good-Die) 선별이 HBM 수율의 핵심인 이유.

## 2. 우리 guardband 선별을 KGD escape 로 환산 (출하 die 기준)

stack 에 들어가는 것은 die test 를 통과해 **출하(ship+repair)** 되는 die 다. 그중 truly-bad = escape.

| KGD 선별 | guardband | escape die / 출하 die | per-die escape |
|---|---|---|---|
| **선별 약함** | 0.00 | {esc_b} / {ship_b} | **{p_before*100:.2f}%** |
| **cost-opt 선별** | {cost_gb} | {esc_a} / {ship_a} | **{after_pt_str(esc_a)}** (point) / ≤ {p_after_ub*100:.2f}% (95% UB, rule-of-three) |

cost-opt guardband 은 이 샘플에서 escape die=0 → 0/{ship_a} 의 정직한 상한은 rule-of-three 로 ≈3/{ship_a}={p_after_ub*100:.2f}%.

## 3. stack 높이별 큐브 yield (전/후 KGD)

{tbl}
- **{HEADLINE_K}-high 주요 수치:** 약한 선별이면 큐브의 **{h.unscreened_cube_escape*100:.0f}%가 latent bad die 를 품는다**
  (all-good {h.unscreened_cube_good*100:.0f}%). cost-opt KGD 선별이면 all-good **≥{(1-h.kgd_cube_escape_ub95)*100:.0f}% (95% 최악)**,
  큐브-escape **≤{h.kgd_cube_escape_ub95*100:.1f}%**. 참고선 0.99 good/die → 0.99^{HEADLINE_K}={h.ref_0p99die_cube_good:.3f}.
- 즉 die 단위로는 작아 보이는 escape 차이가 **{HEADLINE_K}장 적층에서 큐브 손실로 곱셈 증폭**되며, die test guardband
  한 칸이 finished-cube 수율을 좌우한다. (figure: `figures/kgd_stack_yield.png`, 표: `reports/kgd_stacking.csv`.)

## 금지선

- 절대 escape/DPPM 은 합성 가정값. per-die intrinsic TSV/stacking yield 는 별도 항으로 분리(여기선 KGD 선별만 격리).
- "die test 의 escape 가 stack 에서 곱셈 증폭된다"는 **메커니즘과 상대 임팩트**를 보인 것이지 실측 수율 예측이 아니다.
"""
    (REPORT / "kgd_summary.md").write_text(md, encoding="utf-8")


def after_pt_str(esc_a):
    return "0.00%" if esc_a == 0 else f"{esc_a}"


if __name__ == "__main__":
    build()
