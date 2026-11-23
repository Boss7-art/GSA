"""Fine-tune PGDENet/BBSNet on the prepared A-H photo-mask dataset."""

from __future__ import annotations

from workspace_paths import record_path, resolve_reference

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


RGB_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
RGB_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
DEPTH_MEAN = 19025.149172172904
DEPTH_STD = 9880.916283772584
IGNORE_INDEX = 255


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgdenet-root", type=Path, default=Path("external/pgdenet"))
    parser.add_argument("--prepared-root", type=Path, default=Path("prepared_photo_masks"))
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=Path("prepared_photo_masks/emsanet_samples/depth_anything_v2_small_sunrgbd_stats_inverted"),
    )
    parser.add_argument("--output-subdir", default="pgdenet_ah_finetune_depthanything_inverted")
    parser.add_argument("--checkpoint-out", type=Path, default=Path("model/pgdenet.safetensors"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--input-height", type=int, default=256)
    parser.add_argument("--input-width", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--class-weight-power", type=float, default=0.75)
    parser.add_argument("--max-class-weight", type=float, default=20.0)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--aux-weight", type=float, default=0.25)
    parser.add_argument("--augment", action="store_true", help="Use train-only random flip/scale/color augmentation.")
    parser.add_argument("--rare-oversample", type=float, default=1.0, help="Repeat rare-class-rich train samples by this factor.")
    parser.add_argument("--tta-scales", default="1.0", help="Comma-separated inference scales relative to input size.")
    parser.add_argument("--tta-flip", action="store_true", help="Use horizontal flip test-time augmentation.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-list", type=Path, default=None, help="Text file with sample ids used for training.")
    parser.add_argument("--eval-list", type=Path, default=None, help="Text file with sample ids exported after training.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_sample_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def load_image(path: Path, mode: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), mode)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def resize_hw(array: np.ndarray, height: int, width: int, interpolation: int) -> np.ndarray:
    return cv2.resize(array, (width, height), interpolation=interpolation)


def save_id_png(path: Path, array: np.ndarray) -> None:
    array = np.asarray(array)
    if array.max(initial=0) <= np.iinfo(np.uint8).max:
        Image.fromarray(array.astype(np.uint8)).save(path)
    else:
        Image.fromarray(array.astype(np.uint16)).save(path)


def make_sample(sample: dict[str, Any], prepared_root: Path, depth_dir: Path, height: int, width: int) -> tuple[dict[str, np.ndarray], np.ndarray, tuple[int, int]]:
    rgb = load_image(resolve_reference(sample["rgb_path"])).astype(np.float32)
    depth = load_image(depth_dir / f"{sample['sample_id']}.png").astype(np.float32)
    gt = load_image(prepared_root / "gt" / "semantic_id" / f"{sample['sample_id']}.png").astype(np.uint8)
    original_shape = gt.shape[:2]

    rgb = resize_hw(rgb, height, width, cv2.INTER_LINEAR)
    depth = resize_hw(depth, height, width, cv2.INTER_NEAREST)
    gt = resize_hw(gt, height, width, cv2.INTER_NEAREST)

    rgb = ((rgb - RGB_MEAN) / RGB_STD).transpose(2, 0, 1)
    depth = ((depth - DEPTH_MEAN) / DEPTH_STD)[None, :, :]
    target = gt.astype(np.int64) - 1
    target[gt == 0] = IGNORE_INDEX
    return (
        {
            "rgb": rgb[None, :, :, :].astype(np.float32),
            "depth": depth[None, :, :, :].astype(np.float32),
        },
        target[None, :, :],
        original_shape,
    )


def make_train_sample(
    sample: dict[str, Any],
    prepared_root: Path,
    depth_dir: Path,
    height: int,
    width: int,
    augment: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray, tuple[int, int]]:
    if not augment:
        return make_sample(sample, prepared_root, depth_dir, height, width)

    rgb = load_image(resolve_reference(sample["rgb_path"])).astype(np.float32)
    depth = load_image(depth_dir / f"{sample['sample_id']}.png").astype(np.float32)
    gt = load_image(prepared_root / "gt" / "semantic_id" / f"{sample['sample_id']}.png").astype(np.uint8)
    original_shape = gt.shape[:2]

    if random.random() < 0.5:
        rgb = np.ascontiguousarray(rgb[:, ::-1])
        depth = np.ascontiguousarray(depth[:, ::-1])
        gt = np.ascontiguousarray(gt[:, ::-1])

    scale = random.uniform(0.75, 1.35)
    scaled_h = max(height, int(round(rgb.shape[0] * scale)))
    scaled_w = max(width, int(round(rgb.shape[1] * scale)))
    rgb = resize_hw(rgb, scaled_h, scaled_w, cv2.INTER_LINEAR)
    depth = resize_hw(depth, scaled_h, scaled_w, cv2.INTER_NEAREST)
    gt = resize_hw(gt, scaled_h, scaled_w, cv2.INTER_NEAREST)

    y0 = random.randint(0, max(0, scaled_h - height))
    x0 = random.randint(0, max(0, scaled_w - width))
    rgb = rgb[y0:y0 + height, x0:x0 + width]
    depth = depth[y0:y0 + height, x0:x0 + width]
    gt = gt[y0:y0 + height, x0:x0 + width]

    gain = random.uniform(0.85, 1.15)
    bias = random.uniform(-10.0, 10.0)
    rgb = np.clip(rgb * gain + bias, 0, 255)

    rgb = ((rgb - RGB_MEAN) / RGB_STD).transpose(2, 0, 1)
    depth = ((depth - DEPTH_MEAN) / DEPTH_STD)[None, :, :]
    target = gt.astype(np.int64) - 1
    target[gt == 0] = IGNORE_INDEX
    return (
        {
            "rgb": rgb[None, :, :, :].astype(np.float32),
            "depth": depth[None, :, :, :].astype(np.float32),
        },
        target[None, :, :],
        original_shape,
    )


def class_weights(samples: list[dict[str, Any]], prepared_root: Path, power: float, max_weight: float) -> np.ndarray:
    counts = np.zeros(8, dtype=np.float64)
    for sample in samples:
        gt = load_image(prepared_root / "gt" / "semantic_id" / f"{sample['sample_id']}.png").astype(np.uint8)
        for class_id in range(1, 9):
            counts[class_id - 1] += np.count_nonzero(gt == class_id)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / np.power(counts / counts.sum(), power)
    weights = weights / weights.mean()
    return np.clip(weights, 0.25, max_weight).astype(np.float32)


def build_epoch_order(samples: list[dict[str, Any]], prepared_root: Path, rare_oversample: float) -> list[int]:
    order = list(range(len(samples)))
    if rare_oversample <= 1:
        return order
    for idx, sample in enumerate(samples):
        gt = load_image(prepared_root / "gt" / "semantic_id" / f"{sample['sample_id']}.png").astype(np.uint8)
        rare_pixels = sum(np.count_nonzero(gt == class_id) for class_id in (5, 6, 7, 8))
        valid_pixels = max(1, int(np.count_nonzero(gt != 0)))
        rare_ratio = rare_pixels / valid_pixels
        repeats = int(round(rare_oversample * min(4.0, 1.0 + 20.0 * rare_ratio)))
        order.extend([idx] * max(0, repeats - 1))
    return order


def set_batchnorm_eval(model: Any, torch_module: Any) -> None:
    for module in model.modules():
        if isinstance(module, torch_module.nn.modules.batchnorm._BatchNorm):
            module.eval()


def pgdenet_outputs_to_list(outputs: Any) -> list[Any]:
    if isinstance(outputs, (tuple, list)):
        return list(outputs)
    return [outputs]


def soft_dice_loss(logits: Any, target: Any, ignore_index: int, eps: float = 1e-6) -> Any:
    import torch
    import torch.nn.functional as F

    valid = target != ignore_index
    if not bool(valid.any()):
        return logits.sum() * 0.0
    n_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    safe_target = target.clone()
    safe_target[~valid] = 0
    one_hot = F.one_hot(safe_target, num_classes=n_classes).permute(0, 3, 1, 2).to(probs.dtype)
    valid_f = valid.unsqueeze(1).to(probs.dtype)
    probs = probs * valid_f
    one_hot = one_hot * valid_f
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    denominator = probs.sum(dims) + one_hot.sum(dims)
    present = one_hot.sum(dims) > 0
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice[present].mean()


def patch_torchvision_resnet34(torch_module: Any, use_pretrained: bool) -> None:
    import torchvision.models as models

    original_resnet34 = models.resnet34
    cache_path = Path(torch_module.hub.get_dir()) / "checkpoints" / "resnet34-333f7ec4.pth"
    if use_pretrained and not cache_path.exists():
        return

    def local_resnet34(*args: Any, **kwargs: Any) -> Any:
        requested_pretrained = bool(kwargs.pop("pretrained", False))
        weights = kwargs.pop("weights", None)
        model = original_resnet34(*args, weights=None, **kwargs)
        if use_pretrained and (requested_pretrained or weights is not None):
            try:
                state_dict = torch_module.load(cache_path, map_location="cpu", weights_only=False)
            except TypeError:
                state_dict = torch_module.load(cache_path, map_location="cpu")
            model.load_state_dict(state_dict)
        return model

    models.resnet34 = local_resnet34


def parse_tta_scales(value: str) -> list[float]:
    scales = [float(item.strip()) for item in value.split(",") if item.strip()]
    return scales or [1.0]


def predict_tta(model: Any, batch_np: dict[str, np.ndarray], original_shape: tuple[int, int], base_hw: tuple[int, int], scales: list[float], use_flip: bool, torch_module: Any, device: Any) -> Any:
    import torch.nn.functional as F

    logits_sum = None
    n_preds = 0
    base_h, base_w = base_hw
    for scale in scales:
        height = max(32, int(round(base_h * scale / 32.0)) * 32)
        width = max(32, int(round(base_w * scale / 32.0)) * 32)
        rgb = torch_module.from_numpy(batch_np["rgb"]).to(device)
        depth = torch_module.from_numpy(batch_np["depth"]).to(device)
        if (height, width) != tuple(rgb.shape[-2:]):
            rgb_s = F.interpolate(rgb, size=(height, width), mode="bilinear", align_corners=False)
            depth_s = F.interpolate(depth, size=(height, width), mode="nearest")
        else:
            rgb_s, depth_s = rgb, depth
        for flipped in ([False, True] if use_flip else [False]):
            rgb_in, depth_in = rgb_s, depth_s
            if flipped:
                rgb_in = torch_module.flip(rgb_in, dims=(3,))
                depth_in = torch_module.flip(depth_in, dims=(3,))
            logits = model(rgb_in, depth_in)
            if flipped:
                logits = torch_module.flip(logits, dims=(3,))
            logits = F.interpolate(logits, size=original_shape, mode="bilinear", align_corners=False)
            logits_sum = logits if logits_sum is None else logits_sum + logits
            n_preds += 1
    return logits_sum / max(1, n_preds)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    sys.path.insert(0, str(args.pgdenet_root.resolve()))

    import torch
    import torch.nn.functional as F
    patch_torchvision_resnet34(torch, use_pretrained=args.resume is None)
    from toolbox.models.BBSnetmodel.BBSnet import BBSNet

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this torch install has no CUDA support.")
    device = torch.device(device_name)

    prepared_root = args.prepared_root.resolve()
    depth_dir = args.depth_dir.resolve()
    output_root = prepared_root / "predictions" / args.output_subdir
    semantic_dir = output_root / "semantic_id"
    instance_dir = output_root / "instance_id"
    npz_dir = output_root / "semantic_instance_npz"
    meta_dir = output_root / "metadata"
    for directory in (semantic_dir, instance_dir, npz_dir, meta_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = read_json(prepared_root / "metadata" / "manifest.json")
    all_samples = manifest["samples"][: args.limit]
    train_ids = read_sample_ids(args.train_list)
    eval_ids = read_sample_ids(args.eval_list)
    train_samples = [sample for sample in all_samples if train_ids is None or sample["sample_id"] in train_ids]
    eval_samples = [sample for sample in all_samples if eval_ids is None or sample["sample_id"] in eval_ids]
    if not train_samples:
        raise RuntimeError("No samples selected.")
    if not eval_samples:
        raise RuntimeError("No eval samples selected.")

    model = BBSNet(n_class=8)
    if args.resume is not None:
        from safetensors.torch import load_file
        model.load_state_dict(load_file(str(args.resume), device="cpu"))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    weights = torch.from_numpy(class_weights(train_samples, prepared_root, args.class_weight_power, args.max_class_weight)).to(device)
    print(f"class weights: {[round(float(v), 3) for v in weights.cpu().numpy().tolist()]}", flush=True)

    print(
        f"Training PGDENet/BBSNet on {len(train_samples)} train samples for {args.epochs} epochs "
        f"at {args.input_height}x{args.input_width} on {device_name}; exporting {len(eval_samples)} eval samples.",
        flush=True,
    )
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        set_batchnorm_eval(model, torch)
        order = build_epoch_order(train_samples, prepared_root, args.rare_oversample)
        random.shuffle(order)
        total_loss = 0.0
        valid_steps = 0
        for sample_idx in order:
            batch_np, target_np, _ = make_train_sample(
                train_samples[sample_idx],
                prepared_root,
                depth_dir,
                args.input_height,
                args.input_width,
                args.augment,
            )
            rgb = torch.from_numpy(batch_np["rgb"]).to(device)
            depth = torch.from_numpy(batch_np["depth"]).to(device)
            target = torch.from_numpy(target_np).to(device)
            if not bool((target != IGNORE_INDEX).any()):
                continue
            optimizer.zero_grad(set_to_none=True)
            outputs = pgdenet_outputs_to_list(model(rgb, depth))
            loss = 0.0
            aux_weight = args.aux_weight
            for out_idx, logits in enumerate(outputs):
                if logits.shape[-2:] != target.shape[-2:]:
                    logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
                weight = 1.0 if out_idx == len(outputs) - 1 else aux_weight
                ce = F.cross_entropy(logits, target, weight=weights, ignore_index=IGNORE_INDEX)
                dice = soft_dice_loss(logits, target, IGNORE_INDEX)
                loss = loss + weight * (ce + args.dice_weight * dice)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}.")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            valid_steps += 1
        if valid_steps == 0:
            raise RuntimeError(f"No valid training crops at epoch {epoch}.")
        epoch_loss = total_loss / valid_steps
        best_loss = min(best_loss, epoch_loss)
        print(f"epoch {epoch:03d}/{args.epochs:03d}: loss={epoch_loss:.6f}", flush=True)

    if args.epochs > 0:
        from safetensors.torch import save_file
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
                  str(args.checkpoint_out))

    class_table = [{"id": 0, "name": "background", "isthing": False}]
    class_table.extend({"id": i, "name": chr(64 + i), "isthing": True} for i in range(1, 9))
    write_json(meta_dir / "model_classes.json", class_table)
    model.eval()
    tta_scales = parse_tta_scales(args.tta_scales)
    with torch.no_grad():
        for index, sample in enumerate(eval_samples, start=1):
            sample_id = sample["sample_id"]
            semantic_path = semantic_dir / f"{sample_id}.png"
            instance_path = instance_dir / f"{sample_id}.png"
            npz_path = npz_dir / f"{sample_id}.npz"
            if not args.overwrite and any(path.exists() for path in (semantic_path, instance_path, npz_path)):
                raise FileExistsError(f"Outputs already exist for {sample_id}. Use --overwrite.")
            batch_np, _, original_shape = make_sample(sample, prepared_root, depth_dir, args.input_height, args.input_width)
            logits = predict_tta(
                model,
                batch_np,
                original_shape,
                (args.input_height, args.input_width),
                tta_scales,
                args.tta_flip,
                torch,
                device,
            )
            semantic = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint16) + 1
            instance = np.zeros(original_shape, dtype=np.uint16)
            save_id_png(semantic_path, semantic)
            save_id_png(instance_path, instance)
            np.savez_compressed(
                npz_path,
                semantic=semantic,
                instance=instance,
                instance_semantic_ids=np.zeros(1, dtype=np.uint16),
                sample_id=np.asarray(sample_id),
            )
            print(f"[{index:03d}/{len(eval_samples):03d}] {sample_id}: semantic ids {np.unique(semantic).tolist()}", flush=True)

    print(f"Saved PGDENet predictions to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
