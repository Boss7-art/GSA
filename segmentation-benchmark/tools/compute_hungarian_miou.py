"""Compute semantic mIoU after Hungarian matching prediction ids to GT ids."""

from __future__ import annotations

from workspace_paths import record_path

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("prepared_photo_masks"),
        help="Dataset bundle created by prepare_photo_masks_dataset.py.",
    )
    parser.add_argument(
        "--prediction-subdir",
        default="emsanet_sunrgbd_pre",
        help="Subdirectory under prepared-root/predictions containing semantic_instance_npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for JSON/CSV metrics.",
    )
    parser.add_argument(
        "--ignore-gt-id",
        type=int,
        default=0,
        help="GT id to ignore when computing the matching and mIoU.",
    )
    parser.add_argument(
        "--include-absent-gt",
        action="store_true",
        help="Include GT classes from classes.json even if absent in the evaluated pixels.",
    )
    parser.add_argument(
        "--sample-list",
        type=Path,
        default=None,
        help="Optional text file with sample ids to evaluate.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_sample_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def npz_semantic(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return np.asarray(data["semantic"])


def class_name_lookup(path: Path | None) -> dict[int, str]:
    if path is None or not path.exists():
        return {}
    classes = read_json(path)
    return {int(item["id"]): str(item["name"]) for item in classes}


def get_prediction_class_names(prepared_root: Path, prediction_subdir: str) -> dict[int, str]:
    candidates = [
        prepared_root / "predictions" / prediction_subdir / "metadata" / "model_classes.json",
        prepared_root / "predictions" / prediction_subdir / "metadata" / "sunrgbd_classes.json",
        prepared_root / "predictions" / prediction_subdir / "metadata" / "nyuv2_classes.json",
        prepared_root / "predictions" / f"{prediction_subdir}_raw" / "metadata" / "sunrgbd_classes.json",
        prepared_root / "predictions" / f"{prediction_subdir}_raw" / "metadata" / "nyuv2_classes.json",
        prepared_root / "predictions" / f"{prediction_subdir}_raw" / "metadata" / "model_classes.json",
        prepared_root / "predictions" / "emsanet_sunrgbd_pre_raw" / "metadata" / "sunrgbd_classes.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return class_name_lookup(candidate)
    return {}


def compute_confusion(
    prepared_root: Path,
    prediction_subdir: str,
    ignore_gt_id: int,
    sample_ids_filter: set[str] | None = None,
) -> tuple[np.ndarray, list[int], list[int], list[str]]:
    manifest = read_json(prepared_root / "metadata" / "manifest.json")
    gt_root = prepared_root / "gt" / "semantic_instance_npz"
    pred_root = prepared_root / "predictions" / prediction_subdir / "semantic_instance_npz"

    from collections import Counter

    counts = Counter()
    sample_ids = []
    for sample in manifest["samples"]:
        name = sample["sample_id"]
        if sample_ids_filter is not None and name not in sample_ids_filter:
            continue
        gt = npz_semantic(gt_root / f"{name}.npz")
        pred = npz_semantic(pred_root / f"{name}.npz")
        if gt.shape != pred.shape:
            raise ValueError(f"Mask dimensions differ for {name}")
        stride = int(pred.max()) + 1
        for row in range(0, gt.shape[0], 128):
            a = gt[row:row + 128]
            b = pred[row:row + 128]
            valid = a != ignore_gt_id
            codes = a[valid].astype(np.int64) * stride + b[valid]
            bins = np.bincount(codes)
            for idx in np.flatnonzero(bins):
                counts[divmod(int(idx), stride)] += int(bins[idx])
        sample_ids.append(name)
    if not counts:
        raise ValueError("No labelled pixels in selected samples")
    gt_ids = sorted({g for g, _ in counts})
    pred_ids = sorted({p for _, p in counts})
    confusion = np.array([[counts[g, p] for p in pred_ids] for g in gt_ids], dtype=np.int64)
    return confusion, gt_ids, pred_ids, sample_ids


def iou_matrix_from_confusion(confusion: np.ndarray) -> np.ndarray:
    row_sum = confusion.sum(axis=1, keepdims=True)
    col_sum = confusion.sum(axis=0, keepdims=True)
    union = row_sum + col_sum - confusion
    return np.divide(
        confusion,
        union,
        out=np.zeros(confusion.shape, dtype=np.float64),
        where=union != 0,
    )


def hungarian_match(iou_matrix: np.ndarray) -> dict[int, int | None]:
    n_rows, n_cols = iou_matrix.shape
    size = max(n_rows, n_cols)
    padded = np.zeros((size, size), dtype=np.float64)
    padded[:n_rows, :n_cols] = iou_matrix
    rows, cols = linear_sum_assignment(-padded)
    assignment: dict[int, int | None] = {}
    for row, col in zip(rows, cols):
        if row < n_rows:
            assignment[int(row)] = int(col) if col < n_cols else None
    return assignment


def save_matrix_csv(
    path: Path,
    matrix: np.ndarray,
    row_ids: list[int],
    col_ids: list[int],
    row_names: dict[int, str],
    col_names: dict[int, str],
    value_format: str | None = None,
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        header = ["gt_id", "gt_name"]
        header.extend(f"{cid}:{col_names.get(cid, str(cid))}" for cid in col_ids)
        writer.writerow(header)
        for rid, row in zip(row_ids, matrix):
            values = []
            for value in row:
                if value_format is None:
                    values.append(int(value))
                else:
                    values.append(value_format.format(float(value)))
            writer.writerow([rid, row_names.get(rid, str(rid)), *values])


def main() -> int:
    args = parse_args()
    prepared_root = args.prepared_root.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            prepared_root
            / "evaluation"
            / f"{args.prediction_subdir}_hungarian_miou"
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_class_names = class_name_lookup(prepared_root / "metadata" / "classes.json")
    pred_class_names = get_prediction_class_names(prepared_root, args.prediction_subdir)
    sample_ids_filter = read_sample_ids(args.sample_list)

    confusion, gt_ids, pred_ids, sample_ids = compute_confusion(
        prepared_root=prepared_root,
        prediction_subdir=args.prediction_subdir,
        ignore_gt_id=args.ignore_gt_id,
        sample_ids_filter=sample_ids_filter,
    )

    if args.include_absent_gt:
        configured_gt_ids = [
            int(item["id"])
            for item in read_json(prepared_root / "metadata" / "classes.json")
            if int(item["id"]) != args.ignore_gt_id
        ]
        missing = [gt_id for gt_id in configured_gt_ids if gt_id not in gt_ids]
        if missing:
            confusion = np.vstack([confusion, np.zeros((len(missing), confusion.shape[1]), dtype=confusion.dtype)])
            gt_ids = gt_ids + missing

    iou_matrix = iou_matrix_from_confusion(confusion)
    assignment = hungarian_match(iou_matrix)

    per_class_rows: list[dict[str, Any]] = []
    matched_ious = []
    matched_intersection = 0
    matched_union = 0
    for row_idx, gt_id in enumerate(gt_ids):
        col_idx = assignment.get(row_idx)
        pred_id = pred_ids[col_idx] if col_idx is not None else None
        iou = float(iou_matrix[row_idx, col_idx]) if col_idx is not None else 0.0
        intersection = int(confusion[row_idx, col_idx]) if col_idx is not None else 0
        union = int(confusion[row_idx, :].sum() + confusion[:, col_idx].sum() - intersection) if col_idx is not None else int(confusion[row_idx, :].sum())
        matched_ious.append(iou)
        matched_intersection += intersection
        matched_union += union
        per_class_rows.append(
            {
                "gt_id": int(gt_id),
                "gt_name": gt_class_names.get(gt_id, str(gt_id)),
                "matched_pred_id": int(pred_id) if pred_id is not None else None,
                "matched_pred_name": pred_class_names.get(pred_id, str(pred_id)) if pred_id is not None else None,
                "iou": iou,
                "intersection_pixels": intersection,
                "union_pixels": union,
                "gt_pixels": int(confusion[row_idx, :].sum()),
                "pred_pixels_on_valid_gt": int(confusion[:, col_idx].sum()) if col_idx is not None else 0,
            }
        )

    total_valid_pixels = int(confusion.sum())
    matched_pixel_accuracy = matched_intersection / total_valid_pixels if total_valid_pixels else 0.0
    matched_frequency_weighted_iou = matched_intersection / matched_union if matched_union else 0.0
    miou = float(np.mean(matched_ious)) if matched_ious else 0.0

    summary = {
        "prepared_root": record_path(prepared_root),
        "prediction_subdir": args.prediction_subdir,
        "num_samples": len(sample_ids),
        "sample_list": record_path(args.sample_list) if args.sample_list else None,
        "sample_ids": sample_ids,
        "ignore_gt_id": args.ignore_gt_id,
        "gt_ids": gt_ids,
        "pred_ids": pred_ids,
        "mIoU_hungarian": miou,
        "matched_pixel_accuracy": matched_pixel_accuracy,
        "matched_frequency_weighted_iou": matched_frequency_weighted_iou,
        "total_valid_pixels": total_valid_pixels,
        "total_ignored_gt_pixels": None,
        "per_class": per_class_rows,
    }

    with (output_dir / "per_class_iou.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_class_rows)

    save_matrix_csv(
        output_dir / "confusion_matrix.csv",
        confusion,
        gt_ids,
        pred_ids,
        gt_class_names,
        pred_class_names,
        value_format=None,
    )
    save_matrix_csv(
        output_dir / "iou_matrix.csv",
        iou_matrix,
        gt_ids,
        pred_ids,
        gt_class_names,
        pred_class_names,
        value_format="{:.8f}",
    )
    write_json(output_dir / "summary.json", summary)

    print(f"Samples: {len(sample_ids)}")
    print(f"Valid GT pixels: {total_valid_pixels}")
    print(f"Hungarian mIoU: {miou:.6f}")
    print(f"Matched pixel accuracy: {matched_pixel_accuracy:.6f}")
    print(f"Saved metrics to: {output_dir}")
    for row in per_class_rows:
        print(
            f"{row['gt_name']} -> {row['matched_pred_name']} "
            f"(pred id {row['matched_pred_id']}): IoU={row['iou']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
