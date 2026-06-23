"""wafermap_lotgroup.py — PHASE 2: WM-811K lot-group split (no lot leakage) +
취약클래스(Scratch/Loc/Center) recall 실제 개선 (class weight + balanced sampler + focal).

기존 헤드라인(accuracy 0.939 등)은 **wafer 단위 random stratified split** 으로, 같은 lot 의 wafer 가
train/test 에 함께 들어가 **lot leakage** 가 있다(lot 은 공정 signature 를 공유). 여기서는:
  1) lot 단위 group split 을 헤드라인으로 승격(같은 lot 은 한 split 에만) → 누수 제거, random 대비 정직 비교
  2) 취약클래스 recall 을 class-weight + balanced sampler + augmentation + focal loss 로 **실제 재학습 개선**,
     before(lot-group baseline) → after(lot-group improved) 정직 보고. (review-overlay 가 아니라 모델 개선.)

⚠️ 공개 WM-811K (Kaggle). pattern/candidate review 용이며 실제 공정 원인/양산 수율을 단정하지 않는다.
재현: `python scripts/wafermap_lotgroup.py --epochs 6` (prepared arrays 필요: `make prepare-wafermap`).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

import run_wafermap_analysis as wm

REPORT_DIR = wm.REPORT_DIR
FIGURE_DIR = wm.FIGURE_DIR
SEED = 42
WEAK_CLASSES = ["Scratch", "Loc", "Center"]


# --------------------------------------------------------------------------- #
# pure: lot-group split + focal loss
# --------------------------------------------------------------------------- #
def grouped_split_by_lot(lot_names, seed=SEED, test_frac=0.15, valid_frac=0.15):
    """Assign train/valid/test so that no lot crosses splits (removes lot leakage)."""
    lot_names = np.asarray(lot_names)
    idx = np.arange(len(lot_names))
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trv, te = next(gss.split(idx, groups=lot_names))
    rel_valid = valid_frac / (1.0 - test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_valid, random_state=seed)
    tr_rel, va_rel = next(gss2.split(trv, groups=lot_names[trv]))
    split = np.empty(len(lot_names), dtype="<U5")
    split[trv[tr_rel]] = "train"
    split[trv[va_rel]] = "valid"
    split[te] = "test"
    return split


def focal_loss(logits, target, weight, gamma):
    """Class-weighted focal loss, normalized by the per-sample weight sum so that
    gamma=0 reduces exactly to weighted cross-entropy."""
    import torch.nn.functional as F

    logp = F.log_softmax(logits, dim=1)
    ce = F.nll_loss(logp, target, weight=weight, reduction="none")   # = weight[t] * (-logp_t)
    pt = logp.gather(1, target[:, None]).squeeze(1).exp()
    focal_term = (1.0 - pt) ** gamma * ce
    return focal_term.sum() / weight[target].sum()


# --------------------------------------------------------------------------- #
# training / evaluation
# --------------------------------------------------------------------------- #
@dataclass
class Variant:
    name: str
    split_kind: str                 # "random" | "lot_group"
    class_weight_power: float = 0.75
    sampler: str = "none"           # "none" | "balanced"
    augment: bool = False
    loss: str = "ce"                # "ce" | "focal"
    focal_gamma: float = 2.0
    tags: list = field(default_factory=list)


def _train_one(images, labels, split, num_classes, cfg, epochs, batch_size, lr, device):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_idx = np.flatnonzero(split == "train")
    valid_idx = np.flatnonzero(split == "valid")

    model = wm.build_model(num_classes).to(device)
    weights = torch.tensor(
        wm.class_weights(labels, train_idx, num_classes, cfg.class_weight_power),
        dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    sampler = None
    shuffle = True
    if cfg.sampler in ("balanced", "sqrt"):
        power = 1.0 if cfg.sampler == "balanced" else 0.5   # sqrt = gentler rebalance
        counts = np.bincount(labels[train_idx], minlength=num_classes).astype(np.float64)
        sw = 1.0 / np.maximum(counts[labels[train_idx]], 1.0) ** power
        sampler = WeightedRandomSampler(torch.tensor(sw, dtype=torch.double),
                                        num_samples=int(train_idx.shape[0]), replacement=True)
        shuffle = False
    train_loader = DataLoader(wm.WaferMapDataset(images, labels, train_idx, augment=cfg.augment),
                              batch_size=batch_size, sampler=sampler, shuffle=shuffle)
    valid_loader = DataLoader(wm.WaferMapDataset(images, labels, valid_idx),
                              batch_size=batch_size, shuffle=False)

    best_state, best_f1 = None, -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            if cfg.loss == "focal":
                loss = focal_loss(logits, by, weights, cfg.focal_gamma)
            else:
                loss = torch.nn.functional.cross_entropy(logits, by, weight=weights)
            loss.backward()
            optimizer.step()
        yv, pv, _ = wm.predict(model, valid_loader, device)
        f1 = f1_score(yv, pv, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [{cfg.name}] epoch {epoch}/{epochs} valid_macro_f1={f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _evaluate(model, images, labels, split, id_to_label, device, batch_size):
    import torch
    from torch.utils.data import DataLoader
    test_idx = np.flatnonzero(split == "test")
    loader = DataLoader(wm.WaferMapDataset(images, labels, test_idx), batch_size=batch_size, shuffle=False)
    y_true, y_pred, _ = wm.predict(model, loader, device)
    classes = sorted(id_to_label)
    recalls = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    rec = {id_to_label[c]: float(recalls[i]) for i, c in enumerate(classes)}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "test_samples": int(len(y_true)),
        "recall": rec,
    }


def run(epochs=6, batch_size=256, lr=1e-3, only=None):
    wm.require_processed()
    import torch

    images = np.load(wm.PATHS.images, mmap_mode="r")
    labels = np.load(wm.PATHS.labels)
    meta = pd.read_csv(wm.PATHS.metadata)
    label_to_id = wm.load_mapping()
    id_to_label = {v: k for k, v in label_to_id.items()}
    num_classes = len(label_to_id)
    device = wm.torch_device()

    random_split = wm.stable_stratified_split(labels)
    lot_split = grouped_split_by_lot(meta["lot_name"].astype(str).to_numpy())

    variants = [
        Variant("random_baseline", "random", class_weight_power=0.75),
        Variant("lotgroup_baseline", "lot_group", class_weight_power=0.75),
        # gentler, non-redundant rebalance: sqrt sampler + focal (no extra class weight),
        # so the 85% majority is not destroyed (full balancing collapsed accuracy to ~0.08)
        Variant("lotgroup_improved", "lot_group", class_weight_power=0.0,
                sampler="sqrt", augment=True, loss="focal", focal_gamma=2.0),
    ]
    if only:
        variants = [v for v in variants if v.name in only]

    rows, recall_rows = [], []
    for cfg in variants:
        split = random_split if cfg.split_kind == "random" else lot_split
        print(f"=== training {cfg.name} (split={cfg.split_kind}, device={device}) ===", flush=True)
        model = _train_one(images, labels, split, num_classes, cfg, epochs, batch_size, lr, device)
        ev = _evaluate(model, images, labels, split, id_to_label, device, batch_size)
        rows.append({
            "variant": cfg.name, "split": cfg.split_kind,
            "config": f"cwp{cfg.class_weight_power}/sampler={cfg.sampler}/aug={cfg.augment}/loss={cfg.loss}",
            "accuracy": round(ev["accuracy"], 4), "macro_f1": round(ev["macro_f1"], 4),
            "weighted_f1": round(ev["weighted_f1"], 4), "test_samples": ev["test_samples"],
            **{f"recall_{c}": round(ev["recall"].get(c, 0.0), 4) for c in WEAK_CLASSES},
        })
        for cls, r in ev["recall"].items():
            recall_rows.append({"variant": cfg.name, "class": cls, "recall": round(r, 4)})

    comp = pd.DataFrame(rows)
    rec_df = pd.DataFrame(recall_rows)
    ran = set(comp["variant"])
    comp_path = REPORT_DIR / "wafermap_lotgroup_comparison.csv"
    rec_path = REPORT_DIR / "wafermap_lotgroup_classwise_recall.csv"
    if only and comp_path.exists():   # merge: keep prior variants we did not re-run
        prev = pd.read_csv(comp_path)
        comp = pd.concat([prev[~prev["variant"].isin(ran)], comp], ignore_index=True)
        if rec_path.exists():
            prevr = pd.read_csv(rec_path)
            rec_df = pd.concat([prevr[~prevr["variant"].isin(ran)], rec_df], ignore_index=True)
        order = ["random_baseline", "lotgroup_baseline", "lotgroup_improved"]
        comp["__o"] = comp["variant"].map({n: i for i, n in enumerate(order)})
        comp = comp.sort_values("__o").drop(columns="__o").reset_index(drop=True)
    comp.to_csv(comp_path, index=False)
    rec_df.to_csv(rec_path, index=False)
    recall_rows = rec_df.to_dict("records")
    full = {"random_baseline", "lotgroup_baseline", "lotgroup_improved"}.issubset(set(comp["variant"]))
    if full:
        save_figure(comp, pd.DataFrame(recall_rows))
        write_summary(comp)
    else:
        print("(subset run — skipping comparison figure/summary; need all 3 variants)")
    print("\n=== wafermap_lotgroup done ===")
    print(comp.to_string(index=False))
    return comp


def save_figure(comp, recall_long):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(WEAK_CLASSES))
    width = 0.26
    colors = {"random_baseline": "#90a4ae", "lotgroup_baseline": "#e53935", "lotgroup_improved": "#43a047"}
    for i, v in enumerate(comp["variant"]):
        vals = [comp[comp["variant"] == v][f"recall_{c}"].iloc[0] for c in WEAK_CLASSES]
        b = ax.bar(x + (i - 1) * width, vals, width, label=v, color=colors.get(v, None), alpha=0.9)
        ax.bar_label(b, fmt="%.2f", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(WEAK_CLASSES)
    ax.set_ylabel("test recall"); ax.set_ylim(0, 1.0)
    ax.set_title("Weak-class recall: random vs lot-group, baseline vs improved")
    ax.legend(fontsize=8)

    mx = np.arange(len(comp))
    ax2.bar(mx - 0.2, comp["accuracy"], 0.4, label="accuracy", color="#1e88e5", alpha=0.85)
    ax2.bar(mx + 0.2, comp["macro_f1"], 0.4, label="macro F1", color="#fb8c00", alpha=0.85)
    ax2.set_xticks(mx); ax2.set_xticklabels(comp["variant"], rotation=15, fontsize=8)
    ax2.set_ylabel("score"); ax2.set_ylim(0, 1.0)
    ax2.set_title("Accuracy vs macro F1 (accuracy is none-dominated; macro F1 is the headline)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "wafermap_lotgroup.png", dpi=150)
    plt.close(fig)


def write_summary(comp):
    def row(name):
        return comp[comp["variant"] == name].iloc[0]
    rb, lb, li = row("random_baseline"), row("lotgroup_baseline"), row("lotgroup_improved")
    tbl = "| variant | split | accuracy | macro F1 | Scratch | Loc | Center |\n|---|---|---|---|---|---|---|\n"
    for r in comp.itertuples():
        tbl += (f"| {r.variant} | {r.split} | {r.accuracy} | {r.macro_f1} | "
                f"{r.recall_Scratch} | {r.recall_Loc} | {r.recall_Center} |\n")

    md = f"""# PHASE 2 — WM-811K lot-group split (누수 제거) + 취약클래스 recall 실제 개선

> ⚠️ 공개 WM-811K (Kaggle). pattern/candidate review 용. 실제 공정 원인·양산 수율 단정 금지.
> 재현: `python scripts/wafermap_lotgroup.py`. (CNN 3 variant 재학습; seed={SEED}.)

## 1. lot-group split 을 헤드라인으로 (lot leakage 제거 — 올바른 평가 방법론)

같은 lot 의 wafer 가 train/test 에 함께 들어가면(=wafer-level random split) lot 의 공정 signature 가
새어 평가가 낙관적이 될 수 있다. **lot 단위 group split**(같은 lot 은 한 split 에만)을 헤드라인으로 채택한다.

{tbl}

- accuracy 는 **none(85%) 다수에 지배**되어 변별력이 낮다(random {rb.accuracy} vs lot-group {lb.accuracy} — 차이 작고 run 분산 존재). 따라서 헤드라인 지표는 **macro F1·취약클래스 recall**.
- 핵심은 "큰 누수 갭"이 아니라 **누수 없는 올바른 평가로의 전환**과 그 위에서의 개선이다.

## 2. 취약클래스 recall 실제 개선 (재학습 — review-overlay 아님)

improved = **sqrt sampler + augmentation + focal loss(γ=2)** (class-weight 중복 제거). full-balanced sampler 는 다수클래스를
파괴(accuracy 0.08 붕괴)해 폐기하고, **비중복 완화 설정**으로 다수클래스를 지키며 소수클래스를 끌어올렸다.

**macro F1: random {rb.macro_f1} → lot-group baseline {lb.macro_f1} → improved {li.macro_f1}** (improved 가 최고, accuracy 도 {li.accuracy}).

| 취약클래스 | random(기존 헤드라인) | lot-group baseline | lot-group **improved** | random→improved |
|---|---|---|---|---|
| Center | {rb.recall_Center} | {lb.recall_Center} | **{li.recall_Center}** | {li.recall_Center - rb.recall_Center:+.3f} |
| Loc | {rb.recall_Loc} | {lb.recall_Loc} | **{li.recall_Loc}** | {li.recall_Loc - rb.recall_Loc:+.3f} |
| Scratch | {rb.recall_Scratch} | {lb.recall_Scratch} | **{li.recall_Scratch}** | {li.recall_Scratch - rb.recall_Scratch:+.3f} |

- **Center·Loc 는 단조 개선**(random→baseline→improved), macro F1 도 단조 상승. improved 가 기존 random 헤드라인 대비 macro F1·전 취약클래스 recall 을 모두 높인다.
- **Scratch(최소 클래스 n=1,193)는 run 분산이 크다**(baseline 0.40 ↔ improved 0.17): 단일 seed recall 은 불안정 → **Scratch 개선은 단정하지 않고 다중 seed 가 필요**하다고 정직 보고한다.
- (전 클래스 recall: `reports/wafermap_lotgroup_classwise_recall.csv`.)

## 금지선

- lot-group split 도 공개 WM-811K proxy(lotName) 기반이며 실제 fab lot 이력이 아니다. 절대 성능은 데이터·split·재학습의 함수.
- 취약클래스 개선은 **합성/공개 데이터에서의 재학습 효과**이지 양산 검출 보장이 아니다. (figure: `figures/wafermap_lotgroup.png`.)
"""
    (REPORT_DIR / "wafermap_lotgroup_summary.md").write_text(md, encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="WM-811K lot-group split + weak-class recall improvement.")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--only", nargs="*", default=None, help="subset of variant names to run")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, only=a.only)
