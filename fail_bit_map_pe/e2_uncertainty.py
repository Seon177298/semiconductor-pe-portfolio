"""e2_uncertainty.py — E2: operating point 와 escape/overkill 의 불확실성(95% CI).

합성 데이터. point estimate(single seed) 만 보고하면 "운 좋은 한 점"처럼 보인다.
두 종류의 불확실성을 분리해 95% CI 를 붙인다:
  - 다중 seed : die population 을 새로 뽑을 때의 변동 (population variability)
  - bootstrap : 한 population 안에서 die 를 재표집할 때의 sampling 불확실성

대상: cost-optimal guardband 의 안정성, cost-opt 에서의 escape/overkill DPPM,
gb=0 에서의 미검 die 수, guardband 의 비용 절감률.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import fbm_core as c
from run_fbm import GB_GRID, N_DIES, make_population, sweep_guardbands

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports"
FIG = REPORT / "figures"

SEED_BASE = 100      # multi-seed populations: SEED_BASE .. SEED_BASE+N_SEEDS-1
N_SEEDS = 20
CANON_SEED = 7       # canonical population (matches run_fbm) for the bootstrap
N_BOOT = 2000


def ci95(samples):
    """95% percentile interval (2.5, 97.5)."""
    a = np.asarray(samples, dtype=float)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def bootstrap_mean_ci(values, n_boot=N_BOOT, seed=0):
    """95% CI of the mean of `values` by resampling items with replacement."""
    a = np.asarray(values, dtype=float)
    n = len(a)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = a[rng.integers(0, n, n)].mean()
    return ci95(means)


def multi_seed_study(n_seeds=N_SEEDS, n_dies=N_DIES):
    import pandas as pd

    rows = []
    for s in range(n_seeds):
        pop = make_population(np.random.default_rng(SEED_BASE + s), n_dies)
        sweep = sweep_guardbands(*pop, GB_GRID)
        cost_gb = float(sweep.loc[sweep["total_cost"].idxmin(), "guardband"])
        before = sweep[sweep["guardband"] == 0.0].iloc[0]
        after = sweep[sweep["guardband"] == cost_gb].iloc[0]
        red = (before.total_cost - after.total_cost) / before.total_cost * 100.0
        rows.append({
            "seed": SEED_BASE + s,
            "cost_opt_gb": cost_gb,
            "escape_dppm_opt": after.escape_dppm,
            "overkill_dppm_opt": after.overkill_dppm,
            "escape_dies_gb0": before.escape_dies,
            "escape_dies_opt": after.escape_dies,
            "yield_opt": after["yield"],
            "cost_reduction_pct": red,
        })
    return pd.DataFrame(rows)


def canonical_bootstrap(seed=CANON_SEED, n_dies=N_DIES, n_boot=N_BOOT):
    """Per-die escape/overkill at the canonical cost-opt guardband, then bootstrap CIs."""
    m_tests, m_fields, die_types = make_population(np.random.default_rng(seed), n_dies)
    sweep = sweep_guardbands(m_tests, m_fields, die_types, GB_GRID)
    opt_gb = float(sweep.loc[sweep["total_cost"].idxmin(), "guardband"])

    before = sweep[sweep["guardband"] == 0.0].iloc[0]
    after = sweep[sweep["guardband"] == opt_gb].iloc[0]

    true_bad = np.array([c.repair_disposition(mf < 0.0)["disposition"] == "scrap" for mf in m_fields])
    esc_bits = np.empty(n_dies)
    ovr_bits = np.empty(n_dies)
    esc_die = np.zeros(n_dies)
    ovr_die = np.zeros(n_dies)
    for i in range(n_dies):
        det = m_tests[i] < opt_gb
        tf = m_fields[i] < 0.0
        esc_bits[i] = int((tf & ~det).sum())
        ovr_bits[i] = int((~tf & det).sum())
        disp = c.repair_disposition(det)["disposition"]
        if true_bad[i] and disp != "scrap":
            esc_die[i] = 1
        if (not true_bad[i]) and disp == "scrap":
            ovr_die[i] = 1

    to_dppm = 1e6 / c.N_CELLS
    out = {
        "opt_gb": opt_gb,
        "seed7_escape_dppm_opt": float(after.escape_dppm),
        "seed7_escape_dies_gb0": int(before.escape_dies),
        "seed7_cost_reduction_pct": float((before.total_cost - after.total_cost) / before.total_cost * 100.0),
        "escape_dppm_opt": (esc_bits.mean() * to_dppm,) + tuple(x * to_dppm for x in bootstrap_mean_ci(esc_bits, n_boot, seed=1)),
        "overkill_dppm_opt": (ovr_bits.mean() * to_dppm,) + tuple(x * to_dppm for x in bootstrap_mean_ci(ovr_bits, n_boot, seed=2)),
        "escape_dies_opt": (esc_die.sum(),) + tuple(x * n_dies for x in bootstrap_mean_ci(esc_die, n_boot, seed=3)),
        "overkill_dies_opt": (ovr_die.sum(),) + tuple(x * n_dies for x in bootstrap_mean_ci(ovr_die, n_boot, seed=4)),
    }
    return out


def build():
    import pandas as pd

    REPORT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    ms = multi_seed_study()
    ms.to_csv(REPORT / "e2_multiseed_runs.csv", index=False)
    boot = canonical_bootstrap()

    def ms_ci(col):
        return ci95(ms[col])

    metrics = []
    for label, col in [
        ("escape_dppm @ cost-opt", "escape_dppm_opt"),
        ("overkill_dppm @ cost-opt", "overkill_dppm_opt"),
        ("escape_dies @ gb0", "escape_dies_gb0"),
        ("cost reduction %", "cost_reduction_pct"),
        ("cost-opt guardband (V)", "cost_opt_gb"),
    ]:
        lo, hi = ms_ci(col)
        metrics.append({"metric": label, "multiseed_median": round(float(ms[col].median()), 3),
                        "multiseed_ci_lo": round(lo, 3), "multiseed_ci_hi": round(hi, 3)})
    msdf = pd.DataFrame(metrics)
    msdf.to_csv(REPORT / "e2_uncertainty.csv", index=False)

    save_figure(ms, boot)
    write_summary(ms, msdf, boot)

    gb_counts = ms["cost_opt_gb"].value_counts().sort_index()
    print("=== e2_uncertainty done ===")
    print(f"cost-opt guardband across {len(ms)} seeds:\n{gb_counts.to_string()}")
    print(f"escape_dppm@opt: multi-seed 95% CI {ms_ci('escape_dppm_opt')}, "
          f"bootstrap(seed{CANON_SEED}) point {boot['escape_dppm_opt'][0]:.1f} "
          f"CI [{boot['escape_dppm_opt'][1]:.1f}, {boot['escape_dppm_opt'][2]:.1f}]")
    print(f"cost reduction %: multi-seed 95% CI {ms_ci('cost_reduction_pct')}")


def save_figure(ms, boot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    gb_counts = ms["cost_opt_gb"].value_counts().sort_index()
    ax.bar([f"{g:.2f}" for g in gb_counts.index], gb_counts.values, color="#1e88e5", alpha=0.85)
    ax.set_xlabel("cost-optimal guardband (V)")
    ax.set_ylabel(f"# of {len(ms)} seeds")
    ax.set_title("Operating-point stability across seeds (synthetic)")
    ax.grid(True, axis="y", alpha=0.3)

    labels = ["escape DPPM\n@cost-opt", "overkill DPPM\n@cost-opt"]
    pts = [boot["escape_dppm_opt"][0], boot["overkill_dppm_opt"][0]]
    los = [boot["escape_dppm_opt"][1], boot["overkill_dppm_opt"][1]]
    his = [boot["escape_dppm_opt"][2], boot["overkill_dppm_opt"][2]]
    yerr = [[p - lo for p, lo in zip(pts, los)], [hi - p for p, hi in zip(pts, his)]]
    x = np.arange(len(labels))
    ax2.errorbar(x, pts, yerr=yerr, fmt="o", color="crimson", capsize=6, markersize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("DPPM (bit-level)")
    ax2.set_title(f"Bootstrap 95% CI @cost-opt (seed={CANON_SEED}, {N_BOOT} resamples)")
    ax2.grid(True, axis="y", alpha=0.3)
    for xi, p in zip(x, pts):
        ax2.annotate(f"{p:.0f}", (xi, p), textcoords="offset points", xytext=(10, 0), fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG / "e2_uncertainty.png", dpi=150)
    plt.close(fig)


def write_summary(ms, msdf, boot):
    n = len(ms)
    gb_counts = ms["cost_opt_gb"].value_counts().sort_index()
    gb_line = ", ".join(f"{g:.2f}V×{int(cnt)}" for g, cnt in gb_counts.items())

    rows = "\n".join(
        f"| {r.metric} | {r.multiseed_median} | [{r.multiseed_ci_lo}, {r.multiseed_ci_hi}] |"
        for r in msdf.itertuples())

    md = f"""# E2 — operating point·escape/overkill 의 불확실성 (95% CI)

> **합성 데이터.** single-seed point estimate 가 "운 좋은 한 점"이 아님을 보이기 위해
> (1) 다중 seed({n}개, population variability) + (2) bootstrap({N_BOOT} resamples, sampling)
> 두 불확실성을 분리해 95% CI 를 붙였다.

## 1. cost-optimal guardband 의 안정성 (다중 seed)

{n}개 seed 의 cost-opt guardband 분포: **{gb_line}**.
→ operating point 가 한 seed 의 우연이 아니라 **재현적으로 같은 영역**에 형성됨을 보인다.

## 2. 주요 지표의 다중 seed 95% CI

| 지표 | median | 95% CI (다중 seed) |
|---|---|---|
{rows}

## 3. cost-opt 에서의 bootstrap 95% CI (canonical seed={CANON_SEED}, {N_BOOT} resamples)

| 지표 | point | bootstrap 95% CI |
|---|---|---|
| escape DPPM @cost-opt | {boot['escape_dppm_opt'][0]:.1f} | [{boot['escape_dppm_opt'][1]:.1f}, {boot['escape_dppm_opt'][2]:.1f}] |
| overkill DPPM @cost-opt | {boot['overkill_dppm_opt'][0]:.1f} | [{boot['overkill_dppm_opt'][1]:.1f}, {boot['overkill_dppm_opt'][2]:.1f}] |
| escape dies @cost-opt | {boot['escape_dies_opt'][0]:.0f} | [{boot['escape_dies_opt'][1]:.1f}, {boot['escape_dies_opt'][2]:.1f}] |
| overkill dies @cost-opt | {boot['overkill_dies_opt'][0]:.0f} | [{boot['overkill_dies_opt'][1]:.1f}, {boot['overkill_dies_opt'][2]:.1f}] |

- 대표 point(seed={CANON_SEED}, run_fbm): escape **{boot['seed7_escape_dppm_opt']:.1f} DPPM**, 미검 die **0**. 위 bootstrap CI 가 그 point 를 둘러싼다.
- bit-level escape DPPM 은 소수의 누설 셀에 좌우되어 CI 가 상대적으로 넓다 → **재측정/노이즈 저감의 가치**(robustness 와 일관).

## 4. 정직성 플래그 — 대표 seed 는 분포의 우호적 가장자리에 있다

주요 수치는 canonical seed={CANON_SEED} 값이다: escape@cost-opt **{boot['seed7_escape_dppm_opt']:.1f} DPPM**,
미검 die@gb0 **{boot['seed7_escape_dies_gb0']}**, 비용절감 **{boot['seed7_cost_reduction_pct']:.1f}%**.
{n}개 seed 분포와 비교하면 seed{CANON_SEED} 는 escape·절감 모두 **상단(favorable edge)** 에 위치한다:

| 지표 | seed{CANON_SEED}(대표) | 다중seed median | 95% CI | 위치 |
|---|---|---|---|---|
| escape@cost-opt (DPPM) | {boot['seed7_escape_dppm_opt']:.1f} | {ms['escape_dppm_opt'].median():.1f} | [{ci95(ms['escape_dppm_opt'])[0]:.1f}, {ci95(ms['escape_dppm_opt'])[1]:.1f}] | 상단↑ |
| 미검 die@gb0 | {boot['seed7_escape_dies_gb0']} | {ms['escape_dies_gb0'].median():.0f} | [{ci95(ms['escape_dies_gb0'])[0]:.1f}, {ci95(ms['escape_dies_gb0'])[1]:.1f}] | 상단↑ |
| 비용절감 (%) | {boot['seed7_cost_reduction_pct']:.1f} | {ms['cost_reduction_pct'].median():.1f} | [{ci95(ms['cost_reduction_pct'])[0]:.1f}, {ci95(ms['cost_reduction_pct'])[1]:.1f}] | 상단↑ |

→ **권장:** 보고할 때 "−80%"를 단정하기보다 **"약 −{ms['cost_reduction_pct'].median():.0f}% (20-seed 95% CI −{ci95(ms['cost_reduction_pct'])[0]:.0f}~−{ci95(ms['cost_reduction_pct'])[1]:.0f}%), 대표 seed −{boot['seed7_cost_reduction_pct']:.0f}%"** 처럼 CI 와 함께 제시하면
정직하고 방어적이다. (수치를 **바꾸는 게 아니라 맥락을 추가**하는 것 — seed{CANON_SEED} 값은 그대로 재현된다.)

## 산출물

- `reports/e2_uncertainty.csv`(다중 seed CI), `reports/e2_multiseed_runs.csv`(seed별 raw),
  `figures/e2_uncertainty.png`(operating-point 안정성 + bootstrap CI).

## 금지선

- CI 는 **합성 생성·재표집의 통계 변동**일 뿐 실제 실측 변동이 아니다. 절대 DPPM 은 illustrative.
"""
    (REPORT / "e2_uncertainty_summary.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    build()
