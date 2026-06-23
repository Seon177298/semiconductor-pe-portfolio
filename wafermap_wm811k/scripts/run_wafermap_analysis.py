from __future__ import annotations

import argparse
import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = ROOT / "figures"
CACHE_ROOT = Path.home() / ".cache" / "semiconductor_fdc_yield_analysis" / "wm811k"
RAW_DIR = CACHE_ROOT / "raw"
PROCESSED_DIR = CACHE_ROOT / "processed"
MODEL_DIR = CACHE_ROOT / "models"
RAW_FILE = RAW_DIR / "LSWMD.pkl"
IMAGE_SIZE = 64
RANDOM_SEED = 42
EXPECTED_TOTAL_RANGE = (800_000, 820_000)
EXPECTED_LABELED_RANGE = (160_000, 185_000)
CANONICAL_LABELS = {
    "center": "Center",
    "donut": "Donut",
    "edge-loc": "Edge-Loc",
    "edge-ring": "Edge-Ring",
    "loc": "Loc",
    "near-full": "Near-full",
    "random": "Random",
    "scratch": "Scratch",
    "none": "none",
}
MINORITY_REVIEW_CLASSES = ["Donut", "Random", "Scratch", "Near-full", "Center", "Loc", "Edge-Loc", "Edge-Ring"]
WEAK_REVIEW_CLASSES = ["Scratch", "Loc", "Center"]


@dataclass(frozen=True)
class ProcessedPaths:
    images: Path = PROCESSED_DIR / "images_uint8_64.npy"
    labels: Path = PROCESSED_DIR / "labels_int.npy"
    source_indices: Path = PROCESSED_DIR / "source_indices.npy"
    split: Path = PROCESSED_DIR / "split.npy"
    metadata: Path = PROCESSED_DIR / "metadata.csv"
    label_mapping: Path = PROCESSED_DIR / "label_mapping.json"
    manifest: Path = PROCESSED_DIR / "manifest.json"


PATHS = ProcessedPaths()


def normalize_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        return normalize_scalar(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        return normalize_scalar(value[0])
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text in {"", "[]", "nan", "None"}:
        return ""
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            return normalize_scalar(parsed)
        except (ValueError, SyntaxError):
            return text.strip("'\" ")
    return text.strip("'\" ")


def normalize_failure_label(value: Any) -> str:
    raw = normalize_scalar(value)
    if not raw:
        return ""
    key = raw.lower().replace("_", "-")
    return CANONICAL_LABELS.get(key, raw)


def normalize_train_test(value: Any) -> str:
    raw = normalize_scalar(value)
    if not raw:
        return "unassigned"
    key = raw.lower()
    if key.startswith("train"):
        return "original_train"
    if key.startswith("test"):
        return "original_test"
    return raw


def require_raw_file() -> Path:
    if RAW_FILE.exists():
        return RAW_FILE
    write_missing_cache_report()
    raise SystemExit(
        "WM-811K cache is missing. Run `make download-wm811k` after adding ~/.kaggle/kaggle.json. "
        f"Expected file: {RAW_FILE}"
    )


def require_processed() -> None:
    missing = [path for path in PATHS.__dict__.values() if isinstance(path, Path) and not path.exists()]
    if missing:
        raise SystemExit(
            "Processed WM-811K arrays are missing. Run `make prepare-wafermap` first. "
            + "Missing: "
            + ", ".join(str(path) for path in missing)
        )


def resize_nearest(image: Any, size: int = IMAGE_SIZE) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D wafer map, got shape {arr.shape}.")
    height, width = arr.shape
    if height == 0 or width == 0:
        raise ValueError("Empty wafer map.")
    y_idx = np.floor(np.linspace(0, height - 1, size)).astype(np.int64)
    x_idx = np.floor(np.linspace(0, width - 1, size)).astype(np.int64)
    return arr[np.ix_(y_idx, x_idx)].astype(np.uint8, copy=False)


def markdown_table(df: pd.DataFrame, limit: int = 30) -> str:
    view = df.head(limit).copy()
    columns = list(view.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in view.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_missing_cache_report() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "scope": "cache_status",
                "class": "not_installed",
                "count": 0,
                "share": 0.0,
                "note": f"Expected raw file outside Google Drive at {RAW_FILE}.",
            }
        ]
    ).to_csv(REPORT_DIR / "wafermap_class_distribution.csv", index=False)
    (REPORT_DIR / "wafermap_model_report.md").write_text(
        f"""# WM-811K Wafer Map Report

기준일: 2026-06-02

## Status

WM-811K is not cached locally yet.

Expected raw file:

```text
{RAW_FILE}
```

Download command:

```bash
make download-wm811k
```

The downloader requires `~/.kaggle/kaggle.json` and stores the raw pickle outside this Google Drive workspace.

## Boundary

Wafer-map labels support pattern/candidate review. They do not confirm a physical process cause, tool condition, or company-line result.
""",
        encoding="utf-8",
    )


def distribution_rows(scope: str, labels: pd.Series) -> list[dict[str, Any]]:
    counts = labels.value_counts(dropna=False)
    total = int(counts.sum())
    rows: list[dict[str, Any]] = []
    for label, count in counts.items():
        rows.append({"scope": scope, "class": str(label), "count": int(count), "share": int(count) / total})
    return rows


def stable_stratified_split(label_ids: np.ndarray) -> np.ndarray:
    indices = np.arange(label_ids.shape[0])
    train_valid, test = train_test_split(
        indices,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=label_ids,
    )
    train, valid = train_test_split(
        train_valid,
        test_size=0.1764705882,
        random_state=RANDOM_SEED,
        stratify=label_ids[train_valid],
    )
    split = np.full(label_ids.shape[0], "train", dtype="<U5")
    split[valid] = "valid"
    split[test] = "test"
    return split


def prepare() -> None:
    raw_path = require_raw_file()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(raw_path)
    total_rows = len(df)
    if not (EXPECTED_TOTAL_RANGE[0] <= total_rows <= EXPECTED_TOTAL_RANGE[1]):
        raise ValueError(f"Expected roughly 811k WM-811K rows, found {total_rows:,}.")
    for column in ["waferMap", "failureType"]:
        if column not in df.columns:
            raise ValueError(f"WM-811K column `{column}` was not found.")

    failure_labels = df["failureType"].map(normalize_failure_label)
    full_labels = failure_labels.where(failure_labels != "", "unlabeled")
    labeled_mask = failure_labels != ""
    labeled_count = int(labeled_mask.sum())
    if not (EXPECTED_LABELED_RANGE[0] <= labeled_count <= EXPECTED_LABELED_RANGE[1]):
        raise ValueError(f"Expected roughly 172k labeled WM-811K rows, found {labeled_count:,}.")

    labeled = df.loc[labeled_mask].copy()
    labels = failure_labels.loc[labeled_mask].reset_index(drop=True)
    source_indices = labeled.index.to_numpy(dtype=np.int64)
    classes = sorted(labels.unique(), key=lambda item: (item == "none", item))
    label_to_id = {label: idx for idx, label in enumerate(classes)}
    label_ids = labels.map(label_to_id).to_numpy(dtype=np.int64)
    split = stable_stratified_split(label_ids)

    images = np.empty((labeled_count, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for out_idx, wafer_map in enumerate(labeled["waferMap"].to_numpy()):
        images[out_idx] = resize_nearest(wafer_map)

    np.save(PATHS.images, images)
    np.save(PATHS.labels, label_ids)
    np.save(PATHS.source_indices, source_indices)
    np.save(PATHS.split, split)
    PATHS.label_mapping.write_text(json.dumps(label_to_id, indent=2, ensure_ascii=False), encoding="utf-8")

    original_split_col = next((col for col in ["trainTestLabel", "trianTestLabel"] if col in df.columns), None)
    train_test = (
        df.loc[labeled_mask, original_split_col].map(normalize_train_test).reset_index(drop=True)
        if original_split_col
        else pd.Series(["unassigned"] * labeled_count)
    )
    lot_name = (
        labeled["lotName"].map(normalize_scalar).reset_index(drop=True)
        if "lotName" in labeled.columns
        else pd.Series(["unassigned"] * labeled_count)
    )
    wafer_index = (
        pd.to_numeric(labeled["waferIndex"], errors="coerce").reset_index(drop=True)
        if "waferIndex" in labeled.columns
        else pd.Series([np.nan] * labeled_count)
    )
    die_size = (
        pd.to_numeric(labeled["dieSize"], errors="coerce").reset_index(drop=True)
        if "dieSize" in labeled.columns
        else pd.Series([np.nan] * labeled_count)
    )
    metadata = pd.DataFrame(
        {
            "processed_index": np.arange(labeled_count),
            "source_index": source_indices,
            "lot_name": lot_name,
            "wafer_index": wafer_index,
            "die_size": die_size,
            "label": labels,
            "label_id": label_ids,
            "split": split,
            "original_train_test_label": train_test,
        }
    )
    metadata.to_csv(PATHS.metadata, index=False)

    rows: list[dict[str, Any]] = []
    rows.extend(distribution_rows("full_dataset", full_labels))
    rows.extend(distribution_rows("labeled_subset", labels))
    for split_name in ["train", "valid", "test"]:
        rows.extend(distribution_rows(f"{split_name}_split", labels[split == split_name]))
    distribution = pd.DataFrame(rows)
    distribution.to_csv(REPORT_DIR / "wafermap_class_distribution.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    distribution.to_csv(PROCESSED_DIR / "class_distribution.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    manifest = {
        "raw_file": str(raw_path),
        "cache_root": str(CACHE_ROOT),
        "total_rows": total_rows,
        "labeled_rows": labeled_count,
        "image_size": IMAGE_SIZE,
        "classes": classes,
        "original_split_column": original_split_col or "unassigned",
        "metadata_schema_version": 2,
        "raw_proxy_columns": ["lot_name", "wafer_index", "die_size"],
        "random_seed": RANDOM_SEED,
        "split_counts": metadata["split"].value_counts().to_dict(),
    }
    PATHS.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report()
    print(f"Prepared {labeled_count:,} labeled WM-811K wafer maps under {PROCESSED_DIR}.")


class WaferMapDataset:
    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: np.ndarray, augment: bool = False) -> None:
        self.images = images
        self.labels = labels
        self.indices = indices
        self.augment = augment

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> tuple[Any, Any]:
        import torch

        idx = int(self.indices[item])
        image = np.array(self.images[idx], dtype=np.float32, copy=True) / 2.0
        if self.augment:
            if np.random.random() < 0.5:
                image = np.flip(image, axis=0)
            if np.random.random() < 0.5:
                image = np.flip(image, axis=1)
            rotations = int(np.random.randint(0, 4))
            if rotations:
                image = np.rot90(image, rotations)
            image = np.ascontiguousarray(image)
        image = torch.from_numpy(image[None, :, :])
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return image, label


def build_model(num_classes: int) -> Any:
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(64, num_classes),
    )


def torch_device() -> Any:
    import torch

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_mapping() -> dict[str, int]:
    return json.loads(PATHS.label_mapping.read_text(encoding="utf-8"))


def class_weights(label_ids: np.ndarray, train_indices: np.ndarray, num_classes: int, power: float) -> np.ndarray:
    counts = np.bincount(label_ids[train_indices], minlength=num_classes).astype(np.float32)
    weights = (counts.sum() / np.maximum(counts, 1.0)) ** power
    return weights / weights.mean()


def train(args: argparse.Namespace) -> None:
    require_processed()
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    label_to_id = load_mapping()
    images = np.load(PATHS.images, mmap_mode="r")
    labels = np.load(PATHS.labels)
    split = np.load(PATHS.split)
    train_indices = np.flatnonzero(split == "train")
    valid_indices = np.flatnonzero(split == "valid")
    device = torch_device()

    model = build_model(len(label_to_id)).to(device)
    weights = torch.tensor(
        class_weights(labels, train_indices, len(label_to_id), args.class_weight_power),
        dtype=torch.float32,
        device=device,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_counts = np.bincount(labels[train_indices], minlength=len(label_to_id)).astype(np.float64)
    sampler = None
    shuffle = True
    if args.sampler != "none":
        sampler_power = 0.5 if args.sampler == "sqrt" else 1.0
        sample_weights = 1.0 / (np.maximum(train_counts[labels[train_indices]], 1.0) ** sampler_power)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=int(train_indices.shape[0]),
            replacement=True,
        )
        shuffle = False
    train_loader = DataLoader(
        WaferMapDataset(images, labels, train_indices, augment=args.augment),
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle,
    )
    valid_loader = DataLoader(WaferMapDataset(images, labels, valid_indices), batch_size=args.batch_size, shuffle=False)

    best_macro_f1 = -1.0
    history: list[dict[str, Any]] = []
    best_path = MODEL_DIR / "wafermap_cnn_best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_images)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(batch_labels.shape[0])
            total_count += int(batch_labels.shape[0])

        metrics = predict_metrics(model, valid_loader, device)
        row = {
            "epoch": epoch,
            "device": str(device),
            "train_loss": total_loss / max(total_count, 1),
            **{f"valid_{key}": value for key, value in metrics.items()},
        }
        history.append(row)
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label_to_id": label_to_id,
                    "image_size": IMAGE_SIZE,
                    "epoch": epoch,
                    "valid_macro_f1": best_macro_f1,
                },
                best_path,
            )
        print(
            f"epoch={epoch} device={device} loss={row['train_loss']:.4f} "
            f"valid_macro_f1={metrics['macro_f1']:.4f} valid_weighted_f1={metrics['weighted_f1']:.4f}"
            ,
            flush=True,
        )

    pd.DataFrame(history).to_csv(REPORT_DIR / "wafermap_training_history.csv", index=False)
    evaluate(args)


def predict_metrics(model: Any, loader: Any, device: Any) -> dict[str, float]:
    y_true, y_pred, _ = predict(model, loader, device)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def predict(model: Any, loader: Any, device: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []
    with torch.no_grad():
        for batch_images, batch_labels in loader:
            logits = model(batch_images.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_true.append(batch_labels.numpy())
            y_pred.append(probs.argmax(axis=1))
            y_prob.append(probs.max(axis=1))
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_prob)


def evaluate(args: argparse.Namespace) -> None:
    require_processed()
    import torch
    from torch.utils.data import DataLoader

    model_path = MODEL_DIR / "wafermap_cnn_best.pt"
    if not model_path.exists():
        raise SystemExit("Trained model is missing. Run `make train-wafermap` first.")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    label_to_id = checkpoint["label_to_id"]
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    images = np.load(PATHS.images, mmap_mode="r")
    labels = np.load(PATHS.labels)
    split = np.load(PATHS.split)
    test_indices = np.flatnonzero(split == "test")
    device = torch_device()
    model = build_model(len(label_to_id)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loader = DataLoader(WaferMapDataset(images, labels, test_indices), batch_size=args.batch_size, shuffle=False)
    y_true, y_pred, y_prob = predict(model, loader, device)

    write_metrics(y_true, y_pred, y_prob, id_to_label, test_indices)
    write_failure_gallery(images, labels, test_indices, y_true, y_pred, y_prob, id_to_label)
    write_saliency_review(model, images, labels, test_indices, y_true, y_pred, id_to_label, device)
    write_report()
    print("WM-811K evaluation and report generation complete.")


def write_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    id_to_label: dict[int, str],
    test_indices: np.ndarray,
) -> None:
    labels_sorted = sorted(id_to_label)
    metric_rows = [
        {"metric": "accuracy", "value": accuracy_score(y_true, y_pred)},
        {"metric": "macro_f1", "value": f1_score(y_true, y_pred, average="macro", zero_division=0)},
        {"metric": "weighted_f1", "value": f1_score(y_true, y_pred, average="weighted", zero_division=0)},
        {"metric": "test_samples", "value": len(y_true)},
        {"metric": "mean_prediction_confidence", "value": float(np.mean(y_prob))},
    ]
    pd.DataFrame(metric_rows).to_csv(REPORT_DIR / "wafermap_metrics.csv", index=False)

    recalls = recall_score(y_true, y_pred, labels=labels_sorted, average=None, zero_division=0)
    support = np.bincount(y_true, minlength=len(labels_sorted))
    pd.DataFrame(
        [
            {"class": id_to_label[idx], "recall": float(recalls[pos]), "support": int(support[idx])}
            for pos, idx in enumerate(labels_sorted)
        ]
    ).to_csv(REPORT_DIR / "wafermap_classwise_recall.csv", index=False)

    matrix = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    matrix_df = pd.DataFrame(matrix, index=[id_to_label[idx] for idx in labels_sorted], columns=[id_to_label[idx] for idx in labels_sorted])
    matrix_df.to_csv(REPORT_DIR / "wafermap_confusion_matrix.csv")

    pred_rows = pd.DataFrame(
        {
            "processed_index": test_indices,
            "actual": [id_to_label[int(item)] for item in y_true],
            "predicted": [id_to_label[int(item)] for item in y_pred],
            "confidence": y_prob,
            "correct": y_true == y_pred,
        }
    )
    pred_rows.to_csv(REPORT_DIR / "wafermap_test_predictions.csv", index=False)


def plot_wafer_grid(rows: list[dict[str, Any]], images: np.ndarray, path: Path, title: str) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = min(4, len(rows))
    rows_count = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 2.4, rows_count * 2.4), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, row in zip(axes.ravel(), rows):
        axis.imshow(images[int(row["processed_index"])], cmap="viridis", vmin=0, vmax=2)
        axis.set_title(f"{row['actual']} -> {row['predicted']}", fontsize=8)
        axis.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_failure_gallery(
    images: np.ndarray,
    labels: np.ndarray,
    test_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    id_to_label: dict[int, str],
) -> None:
    rows: list[dict[str, Any]] = []
    for class_name in MINORITY_REVIEW_CLASSES:
        class_id = next((idx for idx, label in id_to_label.items() if label == class_name), None)
        if class_id is None:
            continue
        mask = (y_true == class_id) & (y_pred != y_true)
        candidates = np.flatnonzero(mask)
        for local_idx in candidates[:8]:
            rows.append(
                {
                    "processed_index": int(test_indices[local_idx]),
                    "actual": id_to_label[int(y_true[local_idx])],
                    "predicted": id_to_label[int(y_pred[local_idx])],
                    "confidence": float(y_prob[local_idx]),
                    "review_note": "minority-class pattern/candidate review; physical cause is not established",
                }
            )
    gallery = pd.DataFrame(rows)
    gallery.to_csv(REPORT_DIR / "wafermap_failure_gallery.csv", index=False)
    plot_wafer_grid(rows[:16], images, FIGURE_DIR / "wafermap_failure_gallery.png", "WM-811K Minority-Class Failures")


def write_saliency_review(
    model: Any,
    images: np.ndarray,
    labels: np.ndarray,
    test_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    id_to_label: dict[int, str],
    device: Any,
) -> None:
    import torch
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected: list[int] = []
    for class_name in MINORITY_REVIEW_CLASSES:
        class_id = next((idx for idx, label in id_to_label.items() if label == class_name), None)
        if class_id is None:
            continue
        failed = np.flatnonzero((y_true == class_id) & (y_pred != y_true))
        correct = np.flatnonzero((y_true == class_id) & (y_pred == y_true))
        if len(failed):
            selected.append(int(failed[0]))
        if len(correct):
            selected.append(int(correct[0]))
        if len(selected) >= 10:
            break
    if not selected:
        return

    model.eval()
    rows: list[dict[str, Any]] = []
    cols = 2
    fig, axes = plt.subplots(len(selected), cols, figsize=(5.0, max(2.4, len(selected) * 2.0)), squeeze=False)
    for row_idx, local_idx in enumerate(selected):
        processed_index = int(test_indices[local_idx])
        image = np.array(images[processed_index], dtype=np.float32) / 2.0
        tensor = torch.tensor(image[None, None, :, :], dtype=torch.float32, device=device, requires_grad=True)
        logits = model(tensor)
        pred_id = int(logits.argmax(dim=1).detach().cpu().item())
        score = logits[0, pred_id]
        model.zero_grad(set_to_none=True)
        score.backward()
        saliency = tensor.grad.detach().abs().cpu().numpy()[0, 0]
        rows.append(
            {
                "processed_index": processed_index,
                "actual": id_to_label[int(y_true[local_idx])],
                "predicted": id_to_label[pred_id],
                "correct": bool(y_true[local_idx] == y_pred[local_idx]),
                "saliency_mean": float(saliency.mean()),
                "saliency_max": float(saliency.max()),
                "review_note": "input-gradient saliency for pattern/candidate review only",
            }
        )
        axes[row_idx, 0].imshow(image, cmap="viridis", vmin=0, vmax=1)
        axes[row_idx, 0].set_title(f"{rows[-1]['actual']} -> {rows[-1]['predicted']}", fontsize=8)
        axes[row_idx, 1].imshow(saliency, cmap="magma")
        axes[row_idx, 1].set_title("input-gradient saliency", fontsize=8)
        for axis in axes[row_idx]:
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "wafermap_saliency_review.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(REPORT_DIR / "wafermap_saliency_review.csv", index=False)


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def write_experiment_comparison() -> pd.DataFrame | None:
    experiments = {
        "baseline_weighted_loss": "sqrt class weight",
        "balanced_sampler": "balanced sampler + augmentation",
        "class_weight_power_1": "strong class weight",
        "class_weight_power_075": "selected class weight",
    }
    rows: list[dict[str, Any]] = []
    for suffix, strategy in experiments.items():
        metric_path = REPORT_DIR / f"wafermap_metrics_{suffix}.csv"
        recall_path = REPORT_DIR / f"wafermap_classwise_recall_{suffix}.csv"
        if not metric_path.exists() or not recall_path.exists():
            continue
        metrics = pd.read_csv(metric_path)
        recall = pd.read_csv(recall_path)
        metric_map = dict(zip(metrics["metric"], metrics["value"]))
        recall_map = dict(zip(recall["class"], recall["recall"]))
        rows.append(
            {
                "experiment": suffix,
                "strategy": strategy,
                "accuracy": float(metric_map.get("accuracy", 0.0)),
                "macro_f1": float(metric_map.get("macro_f1", 0.0)),
                "weighted_f1": float(metric_map.get("weighted_f1", 0.0)),
                "scratch_recall": float(recall_map.get("Scratch", 0.0)),
                "center_recall": float(recall_map.get("Center", 0.0)),
                "loc_recall": float(recall_map.get("Loc", 0.0)),
                "none_recall": float(recall_map.get("none", 0.0)),
            }
        )
    if not rows:
        return None
    comparison = pd.DataFrame(rows).sort_values(["macro_f1", "weighted_f1"], ascending=False)
    comparison.to_csv(REPORT_DIR / "wafermap_experiment_comparison.csv", index=False)
    return comparison


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def evaluate_review_policy(predictions: pd.DataFrame, policy: str) -> pd.Series:
    confidence = pd.to_numeric(predictions["confidence"], errors="coerce").fillna(0.0)
    predicted = predictions["predicted"].astype(str)
    if policy == "model_only_baseline":
        return pd.Series([False] * len(predictions), index=predictions.index)
    if policy == "weak_pred_or_low_confidence_0_55":
        return predicted.isin(WEAK_REVIEW_CLASSES) | (confidence < 0.55)
    if policy == "weak_pred_or_low_confidence_0_70":
        return predicted.isin(WEAK_REVIEW_CLASSES) | (confidence < 0.70)
    if policy == "non_none_or_low_confidence_0_70":
        return (predicted != "none") | (confidence < 0.70)
    raise ValueError(f"Unknown review policy: {policy}")


def class_capture_rate(predictions: pd.DataFrame, flagged: pd.Series, class_name: str) -> float:
    mask = predictions["actual"].astype(str) == class_name
    return safe_rate(int(flagged[mask].sum()), int(mask.sum()))


def write_imbalance_response_pack(response: pd.DataFrame, recall: pd.DataFrame | None) -> None:
    weak_lines = "Class-wise recall is not available."
    if recall is not None and not recall.empty:
        recall_map = dict(zip(recall["class"], recall["recall"]))
        support_map = dict(zip(recall["class"], recall["support"]))
        weak_lines = "\n".join(
            [
                f"- {name}: recall `{float(recall_map.get(name, 0.0)):.3f}`, support `{int(support_map.get(name, 0))}`"
                for name in WEAK_REVIEW_CLASSES
            ]
        )
    (REPORT_DIR / "Wafermap_Imbalance_Response_Pack.md").write_text(
        f"""# Wafermap Imbalance Response Pack

Date: 2026-06-05

## Purpose

This pack closes the first WM-811K weakness: high overall accuracy hides weak minority-class recall. The response is a manual review overlay, not a retrained model improvement.

## Weak Classes

{weak_lines}

## Review Policy Comparison

{markdown_table(response, limit=20)}

## How To Explain This In Interview

- I did not present accuracy as the main result.
- Scratch, Loc, and Center were treated as weak classes because their recall was visibly lower than the majority class.
- The practical response was to add a manual review overlay based on weak predicted classes and low confidence, then measure how many actual weak-class wafers the queue would capture.
- This is not a retrained model improvement. It is an operations review policy that protects against missed minority defects until more label review, metrology, or retraining evidence exists.

## Next Training Step

The next model-side step is to run real training variants: class-weight power, balanced sampler, and augmentation. Until those experiments are generated, this pack should be described as a review-policy response, not as improved classifier performance.
""",
        encoding="utf-8",
    )


def write_imbalance_response(recall: pd.DataFrame | None) -> pd.DataFrame | None:
    prediction_path = REPORT_DIR / "wafermap_test_predictions.csv"
    if not prediction_path.exists():
        return None
    predictions = pd.read_csv(prediction_path)
    if predictions.empty or not {"actual", "predicted", "confidence"}.issubset(predictions.columns):
        return None
    actual = predictions["actual"].astype(str)
    minority_mask = actual.isin(WEAK_REVIEW_CLASSES)
    none_mask = actual == "none"
    policies = [
        ("model_only_baseline", "model output only"),
        ("weak_pred_or_low_confidence_0_55", "manual review overlay"),
        ("weak_pred_or_low_confidence_0_70", "manual review overlay"),
        ("non_none_or_low_confidence_0_70", "broad manual review overlay"),
    ]
    rows: list[dict[str, Any]] = []
    for policy, policy_type in policies:
        flagged = evaluate_review_policy(predictions, policy)
        rows.append(
            {
                "policy": policy,
                "policy_type": policy_type,
                "flagged_count": int(flagged.sum()),
                "review_rate": safe_rate(int(flagged.sum()), len(predictions)),
                "minority_capture_rate": safe_rate(int(flagged[minority_mask].sum()), int(minority_mask.sum())),
                "scratch_capture_rate": class_capture_rate(predictions, flagged, "Scratch"),
                "loc_capture_rate": class_capture_rate(predictions, flagged, "Loc"),
                "center_capture_rate": class_capture_rate(predictions, flagged, "Center"),
                "none_review_rate": safe_rate(int(flagged[none_mask].sum()), int(none_mask.sum())),
                "interview_safe_claim": (
                    "manual review overlay quantifies how many weak-class wafers would be routed to review; "
                    "not a retrained model improvement"
                    if policy != "model_only_baseline"
                    else "baseline classifier behavior before review overlay"
                ),
            }
        )
    response = pd.DataFrame(rows)
    response.to_csv(REPORT_DIR / "wafermap_imbalance_response.csv", index=False)
    write_imbalance_response_pack(response, recall)
    return response


def write_report() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    distribution = read_csv_if_exists(REPORT_DIR / "wafermap_class_distribution.csv")
    cached_distribution = PROCESSED_DIR / "class_distribution.csv"
    if (
        (distribution is None or "scope" not in distribution.columns or "not_installed" in set(distribution.get("class", [])))
        and cached_distribution.exists()
        and PATHS.manifest.exists()
    ):
        distribution = pd.read_csv(cached_distribution)
        distribution.to_csv(REPORT_DIR / "wafermap_class_distribution.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    metrics = read_csv_if_exists(REPORT_DIR / "wafermap_metrics.csv")
    recall = read_csv_if_exists(REPORT_DIR / "wafermap_classwise_recall.csv")
    gallery = read_csv_if_exists(REPORT_DIR / "wafermap_failure_gallery.csv")
    saliency = read_csv_if_exists(REPORT_DIR / "wafermap_saliency_review.csv")
    experiment_comparison = write_experiment_comparison()
    imbalance_response = write_imbalance_response(recall)
    manifest = json.loads(PATHS.manifest.read_text(encoding="utf-8")) if PATHS.manifest.exists() else {}

    if distribution is None:
        write_missing_cache_report()
        return
    if "scope" not in distribution.columns:
        write_missing_cache_report()
        distribution = pd.read_csv(REPORT_DIR / "wafermap_class_distribution.csv")

    labeled_dist = distribution[distribution["scope"] == "labeled_subset"].copy()
    split_dist = distribution[distribution["scope"].str.endswith("_split")].copy()
    metric_table = markdown_table(metrics) if metrics is not None else "Training has not been run yet."
    recall_table = markdown_table(recall.sort_values("recall") if recall is not None else pd.DataFrame())
    gallery_table = markdown_table(gallery) if gallery is not None and not gallery.empty else "No minority-class failure gallery has been generated yet."
    saliency_table = markdown_table(saliency) if saliency is not None and not saliency.empty else "No saliency review has been generated yet."
    imbalance_response_table = (
        markdown_table(imbalance_response, limit=20)
        if imbalance_response is not None and not imbalance_response.empty
        else "No imbalance response policy has been generated yet."
    )

    (REPORT_DIR / "wafermap_model_report.md").write_text(
        f"""# WM-811K Wafer Map Report

기준일: 2026-06-02

## Data Scale

- Raw cache: `{manifest.get('raw_file', RAW_FILE)}`
- Processed cache: `{PROCESSED_DIR}`
- Total WM-811K rows: {manifest.get('total_rows', 'not prepared')}
- Labeled rows used for CNN: {manifest.get('labeled_rows', 'not prepared')}
- Image preprocessing: nearest-neighbor resize to `{IMAGE_SIZE}x{IMAGE_SIZE}`
- Split: fixed random seed `{RANDOM_SEED}`, stratified train/valid/test

## Labeled Class Distribution

{markdown_table(labeled_dist)}

## Split Distribution

{markdown_table(split_dist, limit=40)}

## CNN Baseline Metrics

{metric_table}

Accuracy is reported only as a secondary indicator. The primary review metrics are macro F1, weighted F1, class-wise recall, confusion matrix, and minority-class failures.

## Imbalance Response Evidence

{imbalance_response_table}

This table is a review-policy response to class imbalance, not a retrained model comparison. It quantifies how weak predicted classes and low-confidence wafers would be routed to manual review.

## Retraining Experiment Comparison

{markdown_table(experiment_comparison) if experiment_comparison is not None and not experiment_comparison.empty else "No retrained imbalance experiment comparison has been generated yet. The current completed mitigation is the review-policy overlay above."}

## Class-Wise Recall

{recall_table}

## Minority-Class Failure Gallery

{gallery_table}

Figure, when generated: `figures/wafermap_failure_gallery.png`

## Saliency Review

{saliency_table}

Figure, when generated: `figures/wafermap_saliency_review.png`

## Boundary

This report uses wafer-map outputs for pattern/candidate review. It does not claim company data, FAB operation, condition optimization, established physical cause, or quantified yield benefit.
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, train, evaluate, and report WM-811K wafer-map analysis.")
    parser.add_argument("command", nargs="?", default="report", choices=["prepare", "train", "evaluate", "report", "run"])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--class-weight-power", type=float, default=0.75)
    parser.add_argument("--sampler", choices=["none", "sqrt", "balanced"], default="none")
    parser.add_argument("--augment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "report":
        write_report()
    elif args.command == "run":
        prepare()
        train(args)


if __name__ == "__main__":
    main()
