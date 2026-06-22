"""
robustness.py — (C) robustness 민감도 분석.

cost-optimal guardband 가 다음에 따라 어떻게 이동하는지 곡선+표로:
  1) 측정노이즈 σ          (robustness_noise.csv)
  2) defect rate           (robustness_defect.csv)
  3) escape:overkill 비용비 (robustness_costratio.csv)
  4) test 온도 corner       (robustness_temperature.csv)  <- 물리 모델(B)의 의미 입증
     : test corner 가 field(85°C)에서 멀어질수록(cooler) escape floor·필요 guardband 가 커진다.

합성 데이터. 비용은 illustrative. seed 고정 재현.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import fbm_core as c

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports"
FIG = REPORT / "figures"

N = 400
SEED = 21
GB_GRID = np.round(np.arange(0.0, 0.121, 0.01), 3)
RATIO_GRID = [(30, 1), (10, 1), (3, 1), (1, 1), (1, 3), (1, 10)]


def gen_population(seed: int, defect_scale: float = 1.0):
    rng = np.random.default_rng(seed)
    m0s, leaks, true_bad = [], [], []
    for _ in range(N):
        dt = rng.choice(c.DIE_TYPES, p=c.DIE_TYPE_P)
        m0, lk, _ = c.make_die(dt, rng, defect_scale=defect_scale)
        m0s.append(m0)
        leaks.append(lk)
        true_bad.append(c.repair_disposition((c.margin_at(m0, lk, c.T_FIELD) < 0.0))["disposition"] == "scrap")
    return m0s, leaks, np.array(true_bad)


def eval_pop(m0s, leaks, true_bad, sigma, T_test, noise_seed=999):
    """Return per-guardband dict list with die/bit escape & overkill."""
    nrng = np.random.default_rng(noise_seed)
    m_tests = [c.margin_at(m0s[i], leaks[i], T_test) + nrng.normal(0, sigma, m0s[i].shape) for i in range(N)]
    m_fields = [c.margin_at(m0s[i], leaks[i], c.T_FIELD) for i in range(N)]
    total_cells = N * c.N_CELLS
    rows = []
    for gb in GB_GRID:
        eb = ob = esc_d = ovr_d = flagged = 0
        for i in range(N):
            det = m_tests[i] < gb
            tf = m_fields[i] < 0.0
            eb += int((tf & ~det).sum())
            ob += int((~tf & det).sum())
            disp = c.repair_disposition(det)["disposition"]
            if disp != "ship":
                flagged += 1
            if true_bad[i] and disp != "scrap":
                esc_d += 1
            if (not true_bad[i]) and disp == "scrap":
                ovr_d += 1
        rows.append({"guardband": float(gb), "escape_dies": esc_d, "overkill_dies": ovr_d,
                     "flagged": flagged, "escape_dppm": round(eb / total_cells * 1e6, 2),
                     "overkill_dppm": round(ob / total_cells * 1e6, 2)})
    return pd.DataFrame(rows)


def cost_opt(df):
    cost = c.fbm_total_cost(df["escape_dies"], df["overkill_dies"], df["flagged"], df["escape_dppm"])
    i = int(cost.idxmin())
    return df.iloc[i], float(cost.min())


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    # base population reused for noise / temperature / cost-ratio sweeps
    base_m0, base_leak, base_bad = gen_population(SEED, defect_scale=1.0)

    # 1) noise sigma sweep
    noise_rows = []
    for sigma in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
        df = eval_pop(base_m0, base_leak, base_bad, sigma, c.T_TEST_DEFAULT)
        opt, cmin = cost_opt(df)
        noise_rows.append({"sigma_meas": sigma, "cost_opt_guardband": opt["guardband"],
                           "escape_dppm_at_opt": opt["escape_dppm"], "overkill_dppm_at_opt": opt["overkill_dppm"],
                           "escape_dies_at_opt": int(opt["escape_dies"]), "min_cost": round(cmin, 1)})
    noise_df = pd.DataFrame(noise_rows)
    noise_df.to_csv(REPORT / "robustness_noise.csv", index=False)

    # 2) defect rate sweep (regenerate per scale)
    defect_rows = []
    for scale in [0.5, 1.0, 1.5, 2.0, 3.0]:
        m0s, leaks, bad = gen_population(SEED + 1, defect_scale=scale)
        df = eval_pop(m0s, leaks, bad, c.SIGMA_MEAS_DEFAULT, c.T_TEST_DEFAULT)
        opt, cmin = cost_opt(df)
        defect_rows.append({"defect_scale": scale, "true_bad_rate": round(bad.mean(), 3),
                            "cost_opt_guardband": opt["guardband"],
                            "escape_dppm_at_opt": opt["escape_dppm"], "overkill_dppm_at_opt": opt["overkill_dppm"],
                            "min_cost": round(cmin, 1)})
    defect_df = pd.DataFrame(defect_rows)
    defect_df.to_csv(REPORT / "robustness_defect.csv", index=False)

    # 3) cost ratio sweep (base pop, continuous bit-level DPPM weighting)
    df_base = eval_pop(base_m0, base_leak, base_bad, c.SIGMA_MEAS_DEFAULT, c.T_TEST_DEFAULT)
    ratio_rows = []
    for e, o in RATIO_GRID:
        cost = df_base["escape_dppm"] * e + df_base["overkill_dppm"] * o
        opt = df_base.iloc[int(cost.idxmin())]
        ratio_rows.append({"escape_weight": e, "overkill_weight": o,
                           "cost_opt_guardband": opt["guardband"],
                           "escape_dppm_at_opt": opt["escape_dppm"],
                           "overkill_dppm_at_opt": opt["overkill_dppm"]})
    ratio_df = pd.DataFrame(ratio_rows)
    ratio_df.to_csv(REPORT / "robustness_costratio.csv", index=False)

    # 4) temperature corner sweep (physical meaning of guardband)
    temp_rows = []
    for tc in [60, 65, 70, 75, 80, 85]:
        T = 273.15 + tc
        df = eval_pop(base_m0, base_leak, base_bad, c.SIGMA_MEAS_DEFAULT, T)
        opt, cmin = cost_opt(df)
        gap = c.arrhenius(c.T_FIELD) / c.arrhenius(T)
        temp_rows.append({"test_corner_C": tc, "arrhenius_gap_to_field": round(gap, 2),
                          "escape_dppm_gb0": float(df.iloc[0]["escape_dppm"]),
                          "cost_opt_guardband": opt["guardband"],
                          "escape_dppm_at_opt": opt["escape_dppm"],
                          "overkill_dppm_at_opt": opt["overkill_dppm"], "min_cost": round(cmin, 1)})
    temp_df = pd.DataFrame(temp_rows)
    temp_df.to_csv(REPORT / "robustness_temperature.csv", index=False)

    save_figs(noise_df, defect_df, temp_df)
    write_summary(noise_df, defect_df, ratio_df, temp_df)

    print("=== robustness (C) done ===")
    print("NOISE:\n", noise_df.to_string(index=False))
    print("\nDEFECT:\n", defect_df.to_string(index=False))
    print("\nCOST RATIO:\n", ratio_df.to_string(index=False))
    print("\nTEMPERATURE:\n", temp_df.to_string(index=False))


def save_figs(noise_df, defect_df, temp_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axes[0].plot(noise_df["sigma_meas"], noise_df["cost_opt_guardband"], "o-", color="purple")
    axes[0].set_xlabel("measurement noise sigma (V)"); axes[0].set_ylabel("cost-opt guardband (V)")
    axes[0].set_title("noise -> guardband"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(defect_df["true_bad_rate"], defect_df["cost_opt_guardband"], "s-", color="teal")
    axes[1].set_xlabel("true bad-die rate"); axes[1].set_ylabel("cost-opt guardband (V)")
    axes[1].set_title("defect rate -> guardband"); axes[1].grid(True, alpha=0.3)

    axes[2].plot(temp_df["test_corner_C"], temp_df["escape_dppm_gb0"], "o--", color="crimson", label="escape DPPM @ gb=0")
    axes[2].plot(temp_df["test_corner_C"], temp_df["escape_dppm_at_opt"], "o-", color="darkred", label="escape DPPM @ cost-opt")
    ax2b = axes[2].twinx()
    ax2b.plot(temp_df["test_corner_C"], temp_df["cost_opt_guardband"], "^-", color="navy", label="cost-opt guardband")
    axes[2].set_xlabel("test corner (C)"); axes[2].set_ylabel("escape DPPM")
    ax2b.set_ylabel("cost-opt guardband (V)")
    axes[2].set_title("test corner -> escape / guardband"); axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", fontsize=8); ax2b.legend(loc="center right", fontsize=8)

    fig.suptitle("Robustness sensitivity (synthetic FBM)")
    fig.tight_layout()
    fig.savefig(FIG / "robustness_sensitivity.png", dpi=150)
    plt.close(fig)


def write_summary(noise_df, defect_df, ratio_df, temp_df):
    def tbl(df, cols):
        head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols)
        body = "\n".join("| " + " | ".join(str(r[col]) for col in cols) + " |" for _, r in df.iterrows())
        return head + "\n" + body

    md = f"""# Robustness 민감도 분석 (C)

> 합성 데이터, 비용 illustrative, seed={SEED}, dies/point={N}. 곡선: `figures/robustness_sensitivity.png`.

## 1. 측정노이즈 σ → cost-optimal guardband
측정 불확실성이 커질수록 같은 guardband 의 overkill 이 급증해, escape 를 줄이는 능력이 떨어진다
(escape DPPM@opt 이 σ와 함께 상승). 즉 노이즈는 escape/overkill 프런티어 전체를 악화시키며,
guardband 만으로는 한계가 있어 **계측 반복(재측정)·노이즈 저감**이 선행돼야 한다.

{tbl(noise_df, list(noise_df.columns))}

## 2. defect rate → cost-optimal guardband
불량률이 오르면 escape 위험이 커져 guardband 가 (대체로) 강화되는 경향.

{tbl(defect_df, list(defect_df.columns))}

## 3. escape:overkill 비용비 → cost-optimal guardband
operating point 는 통계가 아니라 **비용 비대칭**이 결정한다.

{tbl(ratio_df, list(ratio_df.columns))}

## 4. test 온도 corner → escape / guardband  (물리 모델 B 의 의미)
test corner 가 field(85°C)에서 멀어질수록(cooler) Arrhenius 갭이 커져 **retention escape floor 가 상승**하고
같은 비용을 맞추려면 guardband 를 키워야 한다. 즉 **hot-corner(또는 가속 retention) 테스트**가
guardband 보다 근본적인 escape 저감 레버임을 보여준다.

{tbl(temp_df, list(temp_df.columns))}

## 핵심 메시지
- guardband·operating point 는 고정 상수가 아니라 **측정노이즈·불량률·비용비·테스트 온도 코너**의 함수다.
- 특히 retention escape 는 guardband 만으로 한계가 있고, **test 온도를 field 에 가깝게** 가져가는 것이
  escape floor 자체를 낮춘다(4번). 이는 물리 margin 모델(Arrhenius)에서 자연히 도출된다.
"""
    (REPORT / "robustness_summary.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
