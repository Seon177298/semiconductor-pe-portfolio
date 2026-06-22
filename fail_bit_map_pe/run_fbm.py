"""
run_fbm.py — v2 메인 파이프라인.

물리적 margin 모델(B) + redundancy repair(D) 로:
  1) ML 학습용 데이터셋 생성 (generative latent label, pooled fail map, engineered features) -> dataset.npz
  2) guardband sweep: bit/die 단위 escape·overkill, ship/repair/scrap disposition, yield
  3) 비용 환산(코어 가정)으로 cost-optimal guardband 선정, guardband 전/후 표
  4) figures

make_population() / sweep_guardbands() 는 재사용 가능하게 분리되어 있다 (e2_uncertainty.py 가
같은 sweep 을 여러 seed 로 돌려 95% CI 를 구할 때 재사용한다).

합성 데이터. 실제 fab 데이터 아님. seed 고정 재현.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import fbm_core as c

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports"
FIG = REPORT / "figures"

N_DIES = 1500
SEED = 7
POOL = 4                          # 128 -> 32 average pooling for CNN input
GB_GRID = np.round(np.arange(0.0, 0.131, 0.01), 3)
GB_BEFORE = 0.00


def pool_map(fail: np.ndarray) -> np.ndarray:
    g = c.GRID // POOL
    return fail.reshape(g, POOL, g, POOL).mean(axis=(1, 3)).astype(np.float32)


def make_population(rng: np.random.Generator, n_dies: int = N_DIES):
    """Generate a die population's physical margins at the test/field corners.
    Returns (m_tests, m_fields, die_types). One measurement realisation (noise
    baked in) per die at the test corner; field-worst margin is the ground truth."""
    m_tests, m_fields, die_types = [], [], []
    for _ in range(n_dies):
        dt = rng.choice(c.DIE_TYPES, p=c.DIE_TYPE_P)
        m0, lk, _ = c.make_die(dt, rng)
        m_test = c.margin_at(m0, lk, c.T_TEST_DEFAULT) + rng.normal(0.0, c.SIGMA_MEAS_DEFAULT, m0.shape)
        m_field = c.margin_at(m0, lk, c.T_FIELD)
        m_tests.append(m_test.astype(np.float32))
        m_fields.append(m_field.astype(np.float32))
        die_types.append(dt)
    return m_tests, m_fields, die_types


def sweep_guardbands(m_tests, m_fields, die_types, gb_grid=GB_GRID) -> pd.DataFrame:
    """For each guardband: bit-level escape/overkill DPPM, die-level escape/overkill,
    ship/repair/scrap disposition, yield, and total cost. Pure function of a population."""
    n_dies = len(m_tests)
    total_cells = n_dies * c.N_CELLS
    # ground-truth disposition (field worst corner) is guardband-independent
    true_disp = [c.repair_disposition(mf < 0.0)["disposition"] for mf in m_fields]
    true_bad = np.array([d == "scrap" for d in true_disp])

    rows = []
    for gb in gb_grid:
        eb = ob = 0
        ship = repair = scrap = 0
        esc_d = ovr_d = 0
        for i in range(n_dies):
            det = m_tests[i] < gb
            tf = m_fields[i] < 0.0
            eb += int((tf & ~det).sum())
            ob += int((~tf & det).sum())
            disp = c.repair_disposition(det)["disposition"]
            if disp == "ship":
                ship += 1
            elif disp == "repair":
                repair += 1
            else:
                scrap += 1
            if true_bad[i] and disp != "scrap":
                esc_d += 1
            if (not true_bad[i]) and disp == "scrap":
                ovr_d += 1
        flagged = repair + scrap
        escape_dppm = round(eb / total_cells * 1e6, 2)
        overkill_dppm = round(ob / total_cells * 1e6, 2)
        cost = c.fbm_total_cost(esc_d, ovr_d, flagged, escape_dppm)
        rows.append({
            "guardband": round(float(gb), 3),
            "escape_bits": eb, "overkill_bits": ob,
            "escape_dppm": escape_dppm, "overkill_dppm": overkill_dppm,
            "escape_dies": esc_d, "overkill_dies": ovr_d,
            "ship": ship, "repair": repair, "scrap": scrap,
            "yield": round((ship + repair) / n_dies, 4),
            "true_bad_dies": int(true_bad.sum()),
            "total_cost": round(cost, 2),
        })
    return pd.DataFrame(rows)


def build():
    REPORT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    m_tests, m_fields, die_types = make_population(rng, N_DIES)

    # ---- ML dataset from the detected map at the test corner, guardband 0 ----
    X_feat, maps, y, rule_pred = [], [], [], []
    for i in range(N_DIES):
        det0 = m_tests[i] < GB_BEFORE
        X_feat.append(c.extract_features(det0))
        maps.append(pool_map(det0))
        y.append(c.DIE_TYPE_TO_LABEL[die_types[i]])
        rule_pred.append(c.rule_classify(det0))

    X_feat = np.array(X_feat, dtype=np.float32)
    maps = np.array(maps, dtype=np.float32)
    y_idx = np.array([c.LABELS.index(v) for v in y], dtype=np.int64)
    rule_idx = np.array([c.LABELS.index(v) for v in rule_pred], dtype=np.int64)
    np.savez_compressed(REPORT / "fbm_dataset.npz",
                        X_feat=X_feat, maps=maps, y=y_idx, rule_pred=rule_idx,
                        labels=np.array(c.LABELS), die_types=np.array(die_types),
                        feature_names=np.array(c.FEATURE_NAMES))

    # ---- guardband sweep with physical model + repair disposition ----
    sweep = sweep_guardbands(m_tests, m_fields, die_types, GB_GRID)
    sweep.to_csv(REPORT / "fbm_v2_gb_sweep.csv", index=False)

    cost_gb = float(sweep.loc[sweep["total_cost"].idxmin(), "guardband"])
    stat_gb = float(sweep.loc[(sweep["escape_dies"] + sweep["overkill_dies"]).idxmin(), "guardband"])
    before = sweep[sweep["guardband"] == GB_BEFORE].iloc[0]
    after = sweep[sweep["guardband"] == cost_gb].iloc[0]

    save_figures(sweep, maps, die_types, cost_gb)
    write_summary(sweep, before, after, stat_gb, cost_gb)

    print("=== run_fbm (v2) done ===")
    print(f"dataset: {N_DIES} dies, class counts:",
          {lab: int((y_idx == i).sum()) for i, lab in enumerate(c.LABELS)})
    print(f"statistical-opt guardband={stat_gb}  cost-opt guardband={cost_gb}")
    print(f"BEFORE gb0 : escape {before.escape_dppm} DPPM / esc_die {before.escape_dies}, "
          f"overkill {before.overkill_dppm} DPPM / ovr_die {before.overkill_dies}, yield {before['yield']:.3f}, cost {before.total_cost}")
    print(f"AFTER gb{cost_gb}: escape {after.escape_dppm} DPPM / esc_die {after.escape_dies}, "
          f"overkill {after.overkill_dppm} DPPM / ovr_die {after.overkill_dies}, yield {after['yield']:.3f}, cost {after.total_cost}")


def save_figures(sweep, maps, die_types, cost_gb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweep["guardband"], sweep["escape_dppm"], "o-", color="crimson", label="escape DPPM (missed)")
    ax.plot(sweep["guardband"], sweep["overkill_dppm"], "x--", color="navy", label="overkill DPPM (false reject)")
    ax.axvline(cost_gb, color="green", ls=":", label=f"cost-opt gb={cost_gb}")
    ax.set_xlabel("Guardband (V, margin units)")
    ax.set_ylabel("DPPM (bit-level)")
    ax.set_yscale("log")
    ax.set_title(f"Physical FBM: guardband vs escape/overkill "
                 f"(synthetic; test {c.T_TEST_DEFAULT-273.15:.0f}C vs field {c.T_FIELD-273.15:.0f}C)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fbm_v2_guardband_curve.png", dpi=150)
    plt.close(fig)

    # yield / disposition stack
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(sweep["guardband"],
                 sweep["ship"], sweep["repair"], sweep["scrap"],
                 labels=["ship", "repair", "scrap"],
                 colors=["#4caf50", "#ffb300", "#e53935"], alpha=0.85)
    ax.axvline(cost_gb, color="black", ls=":", label=f"cost-opt gb={cost_gb}")
    ax.set_xlabel("Guardband (V)")
    ax.set_ylabel("dies")
    ax.set_title("Disposition (ship/repair/scrap) vs guardband (synthetic)")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(FIG / "fbm_v2_disposition.png", dpi=150)
    plt.close(fig)


def write_summary(sweep, before, after, stat_gb, cost_gb):
    md = f"""# FBM v2 — 물리 margin 모델 + repair disposition 요약

> **합성 데이터.** 실제 fab/계측 데이터 아님. seed={SEED}, dies={N_DIES}.
> 물리 모델: read margin(t,T)=margin0 − leak·t·A(T), Arrhenius Ea={c.EA_EV}eV,
> test corner {c.T_TEST_DEFAULT-273.15:.0f}°C vs field worst {c.T_FIELD-273.15:.0f}°C
> (A_test={c.arrhenius(c.T_TEST_DEFAULT):.1f}, A_field={c.arrhenius(c.T_FIELD):.1f}, 갭 {c.arrhenius(c.T_FIELD)/c.arrhenius(c.T_TEST_DEFAULT):.1f}×).
> 측정노이즈 1σ={c.SIGMA_MEAS_DEFAULT}V. 재현: `python run_fbm.py`. 불확실성(95% CI): `reports/e2_uncertainty_summary.md`.

## guardband 의 물리적 의미

guardband 는 단순 임의 마진이 아니라 **(1) 측정 불확실성(σ={c.SIGMA_MEAS_DEFAULT}V) + (2) test({c.T_TEST_DEFAULT-273.15:.0f}°C)↔field({c.T_FIELD-273.15:.0f}°C) 온도/retention 코너 갭**을 흡수한다.
retention-weak(누설) 셀은 benign test corner 에서는 통과하지만 hot field corner 에서 fail → guardband 로 screen.
구조성 fail(row/col/cluster/edge)은 hard fail 이라 코너 무관하게 잡힌다.

## guardband 전/후 (escape=미검, overkill=과검)

| | guardband(V) | escape DPPM | overkill DPPM | escape dies | overkill dies | ship/repair/scrap | yield |
|---|---|---|---|---|---|---|---|
| **전 (no guardband)** | {before.guardband} | {before.escape_dppm} | {before.overkill_dppm} | {before.escape_dies} | {before.overkill_dies} | {before.ship}/{before.repair}/{before.scrap} | {before['yield']:.3f} |
| **후 (cost-opt)** | {after.guardband} | {after.escape_dppm} | {after.overkill_dppm} | {after.escape_dies} | {after.overkill_dies} | {after.ship}/{after.repair}/{after.scrap} | {after['yield']:.3f} |

- 통계 최적 guardband(오분류 die 최소) = {stat_gb} ; **비용 최소 guardband = {cost_gb}** (`fbm_core.fbm_total_cost`).
- 전 구간: `reports/fbm_v2_gb_sweep.csv`. disposition stack: `figures/fbm_v2_disposition.png`.

## redundancy repair (D)

spare row {c.SPARE_ROWS} / col {c.SPARE_COLS} / single-bit budget {c.SINGLE_BIT_BUDGET}.
disposition 3-way: **ship**(무결/budget 내) / **repair**(spare 로 복구 후 출하) / **scrap**(redundancy 초과·cluster·edge).
yield = (ship+repair)/total. escape die = field-worst 기준 scrap 대상인데 출하된 die.
"""
    (REPORT / "fbm_v2_summary.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    build()
