"""
fbm_core.py — v2 공유 코어: 물리적 margin 모델(B) + redundancy repair(D)
              + feature 추출 + rule 분류 + die 생성(generative latent label).

합성(synthetic) 데이터다. 실제 fab/계측 데이터가 아니다.

물리 모델(B) — cell read-margin 을 Vt/retention/temperature 로 모델링:
  margin(t, T) = margin0 - leak * t * A(T)
    - margin0 : t=0 read sensing margin (V). 정상셀 Gaussian, 구조 fail 은 음수(hard).
    - leak    : retention 누설율 (V per unit time @ Tref). 정상셀≈0, retention-weak 셀은 tail.
    - A(T)    : Arrhenius 온도 가속 = exp((Ea/k)(1/Tref - 1/T))  (Ea=0.6 eV)
  test corner  : (T_test, t_spec) 에서 측정 + 측정노이즈  -> 우리가 보는 값
  field worst  : (T_field, t_spec) 에서의 진짜 margin       -> 출하 후 실패 여부의 ground truth
  => guardband 는 "측정 불확실성 + (test↔field) 온도/retention 코너 갭" 을 흡수하는 물리량.

generative latent label (circularity 차단용): die 를 만들 때 주입한 패턴이 곧 정답 라벨.
  rule 분류기와 ML 분류기 모두 이 latent 을 기준으로 평가한다(둘 다 같은 truth 로 채점).
"""
from __future__ import annotations

import numpy as np

# ---- geometry ----
GRID = 128
N_CELLS = GRID * GRID

# ---- physics ----
EA_EV = 0.6                 # activation energy (eV) for retention leakage
KB = 8.617e-5              # Boltzmann (eV/K)
TREF = 298.15             # reference 25 C
T_TEST_DEFAULT = 353.15   # test/screen corner 80 C (hot retention screen)
T_FIELD = 358.15          # field worst corner 85 C
T_SPEC = 1.0              # normalized retention/refresh time the part must hold

MARGIN0_MEAN = 0.20       # healthy cell read margin at t=0 (V)
MARGIN0_SD = 0.04
SIGMA_MEAS_DEFAULT = 0.02  # tester measurement noise 1 sigma (V)

# ---- supply-voltage (Vdd) operating axis for shmoo / operating-window (E3) ----
VDD_NOM = 1.10            # nominal supply (V); shmoo sweeps a band around this
GAMMA_VDD = 0.5          # read-margin gain per supply volt (dV_margin / dVdd)

# ---- redundancy / repair (D) ----
SPARE_ROWS = 4
SPARE_COLS = 4
SINGLE_BIT_BUDGET = 150
ECC_FREE_BITS = 16        # on-chip ECC absorbs a few bits with no extra repair/test insertion

# ---- cost assumptions (illustrative, USD-equivalent relative units) ----
COST_ESCAPE_DIE = 80.0   # shipping a true-bad die (field return / customer risk)
COST_SCRAP_DIE = 6.0     # good die wrongly scrapped (yield loss)
COST_RETEST_DIE = 1.5    # extra test/repair insertion for each flagged die (TAT)
COST_PER_DPPM = 0.5      # field-quality penalty per outgoing escape DPPM


def fbm_total_cost(escape_dies, overkill_dies, flagged_dies, escape_dppm):
    return (escape_dies * COST_ESCAPE_DIE + overkill_dies * COST_SCRAP_DIE
            + flagged_dies * COST_RETEST_DIE + escape_dppm * COST_PER_DPPM)

# ---- classification thresholds (rule-based) ----
LINE_FAIL_FRAC = 0.50
EDGE_WIDTH = 4
EDGE_RING_FILL = 0.35
EDGE_MIN_FAILS = 120
CLUSTER_MIN = 30

LABELS = ["PASS", "SINGLE_BIT", "ROW", "COLUMN", "CLUSTER", "EDGE"]
LABEL_TO_BIN = {"PASS": "BIN1", "SINGLE_BIT": "BIN2", "ROW": "BIN3",
                "COLUMN": "BIN4", "CLUSTER": "BIN5", "EDGE": "BIN6"}

DIE_TYPES = ["PASS", "MARGINAL", "SINGLE_HEAVY", "ROW", "COLUMN", "CLUSTER", "EDGE"]
DIE_TYPE_P = [0.30, 0.18, 0.10, 0.12, 0.12, 0.09, 0.09]
DIE_TYPE_TO_LABEL = {
    "PASS": "PASS", "MARGINAL": "SINGLE_BIT", "SINGLE_HEAVY": "SINGLE_BIT",
    "ROW": "ROW", "COLUMN": "COLUMN", "CLUSTER": "CLUSTER", "EDGE": "EDGE",
}


def arrhenius(T: float) -> float:
    return float(np.exp((EA_EV / KB) * (1.0 / TREF - 1.0 / T)))


# --------------------------------------------------------------------------------------
# Die generation: returns (margin0, leak, die_type). All physical.
# defect_scale multiplies the count of injected weak/structural fails (for defect-rate sweep).
# --------------------------------------------------------------------------------------
def make_die(die_type: str, rng: np.random.Generator, defect_scale: float = 1.0):
    margin0 = rng.normal(MARGIN0_MEAN, MARGIN0_SD, size=(GRID, GRID))
    leak = np.zeros((GRID, GRID), dtype=float)  # healthy cells: ~no leakage

    # intrinsic retention-weak cells on every non-PASS die (corner-sensitive single bits)
    if die_type != "PASS":
        n = int(rng.binomial(N_CELLS, 0.0008 * defect_scale))
        if n:
            idx = rng.choice(N_CELLS, size=n, replace=False)
            leak.flat[idx] = rng.uniform(0.003, 0.009, n)

    if die_type == "MARGINAL":
        # retention-marginal cells; count straddles the repair budget
        n = int(rng.integers(60, 221) * defect_scale)
        idx = rng.choice(N_CELLS, size=min(n, N_CELLS), replace=False)
        leak.flat[idx] = rng.uniform(0.003, 0.008, len(idx))

    elif die_type == "SINGLE_HEAVY":
        n = int(rng.integers(40, 120) * defect_scale)
        idx = rng.choice(N_CELLS, size=min(n, N_CELLS), replace=False)
        margin0.flat[idx] = rng.normal(-0.12, 0.03, len(idx))  # clearly-failing isolated bits (hard)

    elif die_type == "ROW":
        n_bad = rng.integers(1, 7)
        rows = rng.choice(GRID, size=n_bad, replace=False)
        for r in rows:
            frac = rng.uniform(0.45, 1.0)            # partial/weak lines -> borderline for rules
            sel = rng.choice(GRID, size=int(GRID * frac), replace=False)
            margin0[r, sel] = rng.normal(-0.13, 0.04, len(sel))

    elif die_type == "COLUMN":
        n_bad = rng.integers(1, 7)
        cols = rng.choice(GRID, size=n_bad, replace=False)
        for cc in cols:
            frac = rng.uniform(0.45, 1.0)
            sel = rng.choice(GRID, size=int(GRID * frac), replace=False)
            margin0[sel, cc] = rng.normal(-0.13, 0.04, len(sel))

    elif die_type == "CLUSTER":
        cy, cx = rng.integers(20, GRID - 20, 2)
        radius = rng.integers(6, 16)
        yy, xx = np.mgrid[0:GRID, 0:GRID]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        margin0[mask] = rng.normal(-0.14, 0.04, mask.sum())

    elif die_type == "EDGE":
        w = rng.integers(2, 5)
        ring = np.zeros((GRID, GRID), dtype=bool)
        ring[:w, :] = ring[-w:, :] = ring[:, :w] = ring[:, -w:] = True
        ring &= rng.random((GRID, GRID)) < rng.uniform(0.6, 1.0)
        margin0[ring] = rng.normal(-0.13, 0.04, ring.sum())

    return margin0, leak, die_type


def margin_at(margin0, leak, T: float, t: float = T_SPEC,
              Vdd: float = VDD_NOM) -> np.ndarray:
    """Physical read margin at corner (T, t, Vdd).

    margin = margin0 + GAMMA_VDD*(Vdd - VDD_NOM) - leak*t*A(T)
    At the default Vdd=VDD_NOM the supply term is zero, so existing callers
    (run_fbm, robustness) are unchanged. Raising Vdd lifts read margin
    (drive/sensing), which recovers parametrically-marginal cells but does
    nothing for hard structural fails (margin0 << 0) — the shmoo signature.
    """
    return margin0 + GAMMA_VDD * (Vdd - VDD_NOM) - leak * t * arrhenius(T)


def fail_maps(margin0, leak, rng: np.random.Generator,
              T_test: float = T_TEST_DEFAULT, sigma_meas: float = SIGMA_MEAS_DEFAULT,
              guardband: float = 0.0):
    """Return (det_fail @ test corner with noise & guardband, true_fail @ field worst corner)."""
    m_test = margin_at(margin0, leak, T_test) + rng.normal(0.0, sigma_meas, size=margin0.shape)
    m_field = margin_at(margin0, leak, T_FIELD)
    det_fail = m_test < guardband
    true_fail = m_field < 0.0
    return det_fail, true_fail, m_test, m_field


# --------------------------------------------------------------------------------------
# Connected components (4-neighbour)
# --------------------------------------------------------------------------------------
_STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def cluster_sizes(mask: np.ndarray):
    """Sizes of 4-connected components, descending. Uses scipy.ndimage for speed."""
    from scipy import ndimage
    if not mask.any():
        return []
    labels, n = ndimage.label(mask, structure=_STRUCT4)
    if n == 0:
        return []
    counts = np.bincount(labels.ravel())[1:]  # drop background label 0
    return sorted(counts.tolist(), reverse=True)


# --------------------------------------------------------------------------------------
# Structural analysis shared by rule classifier + repair
# --------------------------------------------------------------------------------------
def analyze(fail: np.ndarray) -> dict:
    n_fail = int(fail.sum())
    w = EDGE_WIDTH
    ring = np.zeros_like(fail, dtype=bool)
    ring[:w, :] = ring[-w:, :] = ring[:, :w] = ring[:, -w:] = True
    ring_fails = int((fail & ring).sum())
    ring_fill = ring_fails / int(ring.sum())
    is_edge = (n_fail >= EDGE_MIN_FAILS and ring_fill >= EDGE_RING_FILL)

    work = fail.copy()
    if is_edge:
        work[ring] = False
    row_counts = work.sum(axis=1)
    col_counts = work.sum(axis=0)
    bad_rows = np.where(row_counts >= LINE_FAIL_FRAC * GRID)[0]
    bad_cols = np.where(col_counts >= LINE_FAIL_FRAC * GRID)[0]

    residual = work.copy()
    residual[bad_rows, :] = False
    residual[:, bad_cols] = False
    res_total = int(residual.sum())
    sizes = cluster_sizes(residual) if res_total else []
    max_clu = sizes[0] if sizes else 0

    return {
        "n_fail": n_fail, "ring_fill": ring_fill, "ring_fails": ring_fails,
        "is_edge": is_edge, "bad_rows": bad_rows, "bad_cols": bad_cols,
        "residual_singles": res_total, "max_cluster": max_clu,
        "cluster_sizes": sizes, "row_counts": row_counts, "col_counts": col_counts,
    }


def rule_classify(fail: np.ndarray) -> str:
    a = analyze(fail)
    if a["is_edge"]:
        return "EDGE"
    if a["max_cluster"] >= CLUSTER_MIN:
        return "CLUSTER"
    if len(a["bad_rows"]) >= 1 and len(a["bad_rows"]) >= len(a["bad_cols"]):
        return "ROW"
    if len(a["bad_cols"]) >= 1:
        return "COLUMN"
    if a["residual_singles"] > 0:
        return "SINGLE_BIT"
    return "PASS"


# --------------------------------------------------------------------------------------
# Repair / disposition (D) — spare row/col + single-bit budget; cluster/edge unrepairable
# --------------------------------------------------------------------------------------
def repair_disposition(fail: np.ndarray) -> dict:
    a = analyze(fail)
    n_bad_rows, n_bad_cols = len(a["bad_rows"]), len(a["bad_cols"])
    singles = a["residual_singles"]

    if a["n_fail"] == 0:
        return {"disposition": "ship", "bin": "BIN1", "reason": "no fail",
                "rows_used": 0, "cols_used": 0, "singles_repaired": 0}

    # 2D gross defects: redundancy cannot fix
    if a["is_edge"]:
        return {"disposition": "scrap", "bin": "BIN6", "reason": "edge ring (2D)",
                "rows_used": 0, "cols_used": 0, "singles_repaired": 0}
    if a["max_cluster"] >= CLUSTER_MIN:
        return {"disposition": "scrap", "bin": "BIN5", "reason": "cluster (2D)",
                "rows_used": 0, "cols_used": 0, "singles_repaired": 0}

    # line + single-bit repair within budget
    if n_bad_rows > SPARE_ROWS or n_bad_cols > SPARE_COLS:
        bin_ = "BIN3" if n_bad_rows >= n_bad_cols else "BIN4"
        return {"disposition": "scrap", "bin": bin_, "reason": "line repair over redundancy",
                "rows_used": n_bad_rows, "cols_used": n_bad_cols, "singles_repaired": 0}
    if singles > SINGLE_BIT_BUDGET:
        return {"disposition": "scrap", "bin": "BIN2", "reason": "single-bit over budget",
                "rows_used": n_bad_rows, "cols_used": n_bad_cols, "singles_repaired": 0}

    # repairable
    if n_bad_rows or n_bad_cols:
        bin_ = "BIN3" if n_bad_rows >= n_bad_cols else "BIN4"
        disp, reason = "repair", "spare row/col allocation"
    elif singles > ECC_FREE_BITS:
        bin_, disp, reason = "BIN2", "repair", "single-bit repair (> ECC-free)"
    else:
        # zero fails or only a few bits absorbed by on-chip ECC -> ship, no repair insertion
        bin_, disp, reason = ("BIN1" if singles == 0 else "BIN2"), "ship", "within ECC-free budget"
    return {"disposition": disp, "bin": bin_, "reason": reason,
            "rows_used": n_bad_rows, "cols_used": n_bad_cols, "singles_repaired": singles}


# --------------------------------------------------------------------------------------
# Feature extraction for ML classifier (A)
# --------------------------------------------------------------------------------------
FEATURE_NAMES = [
    "n_fail", "fail_rate", "n_bad_rows", "n_bad_cols",
    "max_row_frac", "max_col_frac", "row_count_std", "col_count_std",
    "largest_cluster", "second_cluster", "n_big_clusters",
    "ring_fill", "frac_in_ring", "n_isolated_bits", "largest_aspect",
]


def extract_features(fail: np.ndarray) -> list:
    a = analyze(fail)
    n_fail = a["n_fail"]
    rc, cc = a["row_counts"], a["col_counts"]
    sizes = a["cluster_sizes"]
    largest = sizes[0] if sizes else 0
    second = sizes[1] if len(sizes) > 1 else 0
    n_big = int(sum(1 for s in sizes if s >= 5))

    # isolated bits: fails with no 4-neighbour fail
    nb = np.zeros_like(fail, dtype=int)
    nb[1:, :] += fail[:-1, :]; nb[:-1, :] += fail[1:, :]
    nb[:, 1:] += fail[:, :-1]; nb[:, :-1] += fail[:, 1:]
    n_isolated = int((fail & (nb == 0)).sum())

    # aspect ratio of the largest residual component bounding box
    aspect = 1.0
    if largest > 0:
        w = EDGE_WIDTH
        ring = np.zeros_like(fail, dtype=bool)
        ring[:w, :] = ring[-w:, :] = ring[:, :w] = ring[:, -w:] = True
        work = fail.copy()
        if a["is_edge"]:
            work[ring] = False
        work[a["bad_rows"], :] = False
        work[:, a["bad_cols"]] = False
        ys, xs = np.where(work)
        if len(ys):
            hh = ys.max() - ys.min() + 1
            ww = xs.max() - xs.min() + 1
            aspect = max(hh, ww) / max(1, min(hh, ww))

    return [
        n_fail, n_fail / N_CELLS, len(a["bad_rows"]), len(a["bad_cols"]),
        rc.max() / GRID, cc.max() / GRID, float(rc.std()), float(cc.std()),
        largest, second, n_big, a["ring_fill"], a["ring_fails"] / max(1, n_fail),
        n_isolated, aspect,
    ]
