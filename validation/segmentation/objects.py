import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from src.utils import save_mask, save_json as write_json
from common import sha256, weights

def grid_starts(length: int, side: int, overlap: int) -> list[int]:
    if length <= side:
        return [0]
    starts = list(range(0, length - side + 1, side - overlap))
    if starts[-1] != length - side:
        starts.append(length - side)
    return starts


def make_views(size: tuple[int, int], settings: dict) -> list[dict]:
    width, height = size
    views = []
    if settings["whole_image_pass"]:
        views.append({"id": "whole", "rect": [0, 0, width, height]})
    side = settings["tile_size"]
    for y in grid_starts(height, side, settings["tile_overlap"]):
        for x in grid_starts(width, side, settings["tile_overlap"]):
            rect = [x, y, min(x + side, width), min(y + side, height)]
            if views and rect == views[0]["rect"]:
                continue
            views.append({"id": f"tile_{len(views):03d}", "rect": rect})
    return views


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    lo = np.maximum(box[:2], boxes[:, :2])
    hi = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.prod(np.maximum(hi - lo, 0), axis=1)
    area = np.prod(np.maximum(box[2:] - box[:2], 0))
    areas = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    return intersection / np.maximum(area + areas - intersection, 1e-9)


def classwise_nms(candidates: list[dict], threshold: float) -> list[int]:
    keep = []
    for category in dict.fromkeys(item["category"] for item in candidates):
        indices = [i for i, item in enumerate(candidates) if item["category"] == category]
        indices.sort(key=lambda i: (-candidates[i]["detection_score"], i))
        while indices:
            index = indices.pop(0)
            keep.append(index)
            if indices:
                overlaps = box_iou(
                    np.array(candidates[index]["box_global"]),
                    np.array([candidates[i]["box_global"] for i in indices]),
                )
                indices = [i for i, overlap in zip(indices, overlaps) if overlap <= threshold]
    return sorted(keep)


def load_rgb(path: Path) -> tuple[Image.Image, dict]:
    with Image.open(path) as opened:
        info = {"raw_size_wh": list(opened.size), "exif_orientation": opened.getexif().get(274, 1)}
        rgb = ImageOps.exif_transpose(opened).convert("RGB")
    info["upright_size_wh"] = list(rgb.size)
    return rgb, info

def main():
    parser = argparse.ArgumentParser(description="Segment vegetation, light fixtures and advertising signs.")
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("objects.json"))
    parser.add_argument("--detector", type=Path, default=weights() / "dino_tiny")
    parser.add_argument("--segmenter", type=Path, default=weights() / "sam21_small")
    args = parser.parse_args()
    import torch
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if not 0 <= settings["tile_overlap"] < settings["tile_size"]:
        raise ValueError("Tile overlap must be smaller than the tile side.")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError("Use a fresh output directory; existing runs are preserved.")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = []
    for path in args.images:
        rgb, frame = load_rgb(path)
        samples.append({"path": path, "rgb": rgb, "frame": frame, "views": make_views(rgb.size, settings)})
    if len({s["path"].stem.casefold() for s in samples}) != len(samples):
        raise ValueError("Input photographs must have distinct names.")
    manifest = {"settings": settings, "annotations_used": False,
                "models": [{"id": settings[key], "sha256": sha256(path / "model.safetensors")}
                           for key, path in (("detector_id", args.detector), ("segmenter_id", args.segmenter))],
                "samples": [{"path": s["path"].name, "sha256": sha256(s["path"]), **s["frame"]} for s in samples]}
    write_json(args.output / "manifest.json", manifest)
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

    detector_processor = AutoProcessor.from_pretrained(str(args.detector), local_files_only=True)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(
        str(args.detector), local_files_only=True, disable_custom_kernels=True,
    ).eval().to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for sample in samples:
        candidates = []
        rgb = sample["rgb"]
        folder = args.output / sample["path"].stem
        folder.mkdir(exist_ok=True)
        sample["folder"] = folder
        for view_number, view in enumerate(sample["views"], 1):
            patch = rgb.crop(view["rect"])
            x0, y0 = view["rect"][:2]
            for category, prompt in settings["prompts"].items():
                inputs = detector_processor(images=patch, text=prompt, return_tensors="pt").to(device)
                with torch.inference_mode():
                    outputs = detector(**inputs)
                result = detector_processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=settings["detection_threshold"],
                    text_threshold=settings["text_threshold"], target_sizes=[(patch.height, patch.width)],
                )[0]
                labels = result.get("text_labels", result.get("labels"))
                for box, score, phrase in zip(result["boxes"].cpu().tolist(), result["scores"].cpu().tolist(), labels):
                    box = [max(0.0, min(float(box[0]), patch.width)), max(0.0, min(float(box[1]), patch.height)),
                           max(0.0, min(float(box[2]), patch.width)), max(0.0, min(float(box[3]), patch.height))]
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    candidates.append({"id": len(candidates), "category": category, "phrase": str(phrase),
                                       "detection_score": float(score), "view_id": view["id"], "box_local": box,
                                       "box_global": [box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0]})
                del outputs, inputs
            print(f"DINO {sample['path'].stem} view {view_number}/{len(sample['views'])}: {len(candidates)} raw boxes", flush=True)
        keep = classwise_nms(candidates, settings["classwise_box_nms_iou"])
        write_json(folder / "detections_before_nms.json", candidates)
        sample["candidates"] = [candidates[i] for i in keep]
        write_json(folder / "detections_after_nms.json", sample["candidates"])
    del detector, detector_processor
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    print("Detection complete. Loading SAM 2.1 Small.", flush=True)
    sam_processor = Sam2Processor.from_pretrained(str(args.segmenter), local_files_only=True)
    sam = Sam2Model.from_pretrained(str(args.segmenter), local_files_only=True).eval().to(device)
    summaries = []
    for sample in samples:
        rgb, folder = sample["rgb"], sample["folder"]
        unions = {category: np.zeros((rgb.height, rgb.width), bool) for category in settings["prompts"]}
        by_view = defaultdict(list)
        for item in sample["candidates"]:
            by_view[item["view_id"]].append(item)
        instance_folder = folder / "raw_instance_masks"
        instance_folder.mkdir(exist_ok=True)
        completed = 0
        for view in sample["views"]:
            items = by_view[view["id"]]
            if not items:
                continue
            patch = rgb.crop(view["rect"])
            x0, y0 = view["rect"][:2]
            embeddings = None
            for offset in range(0, len(items), settings["sam_box_batch"]):
                chunk = items[offset:offset + settings["sam_box_batch"]]
                inputs = sam_processor(images=patch, input_boxes=[[i["box_local"] for i in chunk]], return_tensors="pt").to(device)
                if embeddings is not None:
                    inputs.pop("pixel_values")
                with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    outputs = sam(**inputs, image_embeddings=embeddings, multimask_output=settings["sam_multimask"])
                embeddings = outputs.image_embeddings
                scores = outputs.iou_scores[0].float().cpu()
                best = scores.argmax(-1)
                low_res = outputs.pred_masks[0].float().cpu()
                chosen = low_res[torch.arange(len(chunk)), best].unsqueeze(0).unsqueeze(2)
                resized = sam_processor.post_process_masks(chosen, inputs["original_sizes"].cpu(), mask_threshold=settings["sam_mask_threshold"])[0]
                for index, item in enumerate(chunk):
                    mask = resized[index, 0].numpy().astype(bool)
                    area = int(mask.sum())
                    item["sam_predicted_iou"] = float(scores[index, best[index]])
                    item["sam_selected_mask"] = int(best[index])
                    item["sam_mask_pixels"] = area
                    item["sam_view_rect"] = view["rect"]
                    item["status"] = "predicted" if area else "empty_sam_prediction"
                    if area:
                        unions[item["category"]][y0:y0 + patch.height, x0:x0 + patch.width] |= mask
                        ys, xs = np.nonzero(mask)
                        rect = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                        name = f"{item['id']:05d}_{item['category']}.png"
                        Image.fromarray(mask[rect[1]:rect[3], rect[0]:rect[2]].astype(np.uint8) * 255).save(instance_folder / name)
                        item["mask_file"] = "raw_instance_masks/" + name
                        item["mask_rect_global"] = [rect[0] + x0, rect[1] + y0, rect[2] + x0, rect[3] + y0]
                completed += len(chunk)
                del outputs, inputs, low_res, chosen, resized
            del embeddings
            print(f"SAM {sample['path'].stem} {view['id']}: {completed}/{len(sample['candidates'])} candidate masks", flush=True)
        write_json(folder / "instances.json", sample["candidates"])
        for category, mask in unions.items():
            save_mask(folder / f"{category}_mask.png", mask)
        summary = {"image": sample["path"].stem, "size_wh": list(rgb.size), "view_count": len(sample["views"]), "categories": {}}
        for category, mask in unions.items():
            category_items = [i for i in sample["candidates"] if i["category"] == category]
            area = int(mask.sum())
            summary["categories"][category] = {"candidate_boxes_after_nms": len(category_items),
                "nonempty_candidate_masks": sum(i["sam_mask_pixels"] > 0 for i in category_items),
                "union_pixels": area, "union_fraction": area / mask.size,
                "status": "automatic_prediction_needs_review" if area else "no_confident_prediction"}
        write_json(folder / "summary.json", summary)
        summaries.append(summary)

    write_json(args.output / "results.json", {"samples": summaries})


if __name__ == "__main__":
    main()
