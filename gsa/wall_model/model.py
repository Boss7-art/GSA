import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.ops import nms
from slab_model.features import DenseVision, GRID


PROMPTS = (
    "shop window. glass storefront. glass wall. glass door. display window.",
    "shop entrance. open doorway. corridor opening. archway. entrance to store.",
    "opaque wall. column. pillar. solid wall panel. door frame.",
    "person. potted plant. freestanding signboard. chair. table.",
)
SEMANTICS = (
    "wall", "floor", "ceiling", "windowpane", "door", "column", "person", "glass",
    "building", "signboard", "plant", "chair", "table", "escalator", "mirror", "railing", "shelf",
)


class WallFeatures(DenseVision):
    def semantic(self, image):
        captured = []
        hook = self.seg.decode_head.classifier.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0])
        )
        try:
            inputs = self.sp(images=image, return_tensors="pt").to(self.device)
            logits = self.seg(**inputs).logits[0]
        finally:
            hook.remove()
        embedding = captured[0][0].float()
        embedding = embedding / embedding.square().mean(0, keepdim=True).add(1.e-6).sqrt()
        probabilities = logits.float().softmax(0)
        priors = torch.stack([
            probabilities[self.class_ids[label]] if label in self.class_ids else torch.zeros_like(probabilities[0])
            for label in SEMANTICS
        ])
        return torch.cat([embedding, 3 * priors]).cpu().numpy()

    def multiscale_semantics(self, image):
        global_semantics = self.semantic(image)
        details = np.zeros_like(global_semantics)
        weight = np.zeros((1, 1, GRID), np.float32)
        for left, right in ((0, 80), (48, 128)):
            start = min(image.width - 1, round(left * image.width / GRID))
            end = max(start + 1, round(right * image.width / GRID))
            block = self.semantic(image.crop((start, 0, end, image.height)))
            block = F.interpolate(
                torch.from_numpy(block)[None], size=(GRID, right - left), mode="bilinear", align_corners=False
            )[0].numpy()
            taper = np.ones(right - left, np.float32)
            if left:
                taper[:32] = np.linspace(.02, 1, 32)
            else:
                taper[-32:] = np.linspace(1, .02, 32)
            details[:, :, left:right] += block * taper
            weight[:, :, left:right] += taper
        return (.65 * torch.from_numpy(global_semantics) + .35 * torch.from_numpy(details / weight)).numpy()

    def candidates(self, image):
        layers = np.zeros((len(PROMPTS) * 2, GRID, GRID), np.float32)
        embeddings = None
        for group, prompt in enumerate(PROMPTS):
            inputs = self.dp(images=image, text=prompt, return_tensors="pt").to(self.device)
            detections = self.dino(**inputs)
            regions = self.dp.post_process_grounded_object_detection(
                detections, inputs.input_ids, threshold=.18, text_threshold=.22,
                target_sizes=[(image.height, image.width)],
            )[0]
            boxes, scores = regions["boxes"], regions["scores"]
            boxes[:, 0::2].clamp_(0, image.width)
            boxes[:, 1::2].clamp_(0, image.height)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes, scores = boxes[valid], scores[valid]
            if not len(boxes):
                continue
            for selection in nms(boxes, scores, .72)[:16].split(4):
                selected = boxes[selection].cpu()
                batch = self.mp(images=image, input_boxes=[selected.tolist()], return_tensors="pt").to(self.device)
                if embeddings is not None:
                    batch.pop("pixel_values")
                use_bfloat16 = self.device == "cuda" and torch.cuda.is_bf16_supported()
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=use_bfloat16):
                    segmentation = self.sam(**batch, image_embeddings=embeddings, multimask_output=False)
                embeddings = segmentation.image_embeddings
                proposals = segmentation.pred_masks[0, :, 0].float().cpu().numpy()
                for logits, confidence, box in zip(proposals, scores[selection].tolist(), selected.numpy()):
                    mask = cv2.resize(logits, (GRID, GRID), interpolation=cv2.INTER_LINEAR) > 0
                    layer = layers[2 * group]
                    layer[mask] = np.maximum(layer[mask], confidence)
                    normalized = box / np.array([image.width, image.height, image.width, image.height]) * GRID
                    x1, y1 = np.floor(normalized[:2]).astype(int)
                    x2, y2 = np.ceil(normalized[2:]).astype(int)
                    region = layers[2 * group + 1, y1:y2, x1:x2]
                    np.maximum(region, confidence, out=region)
        return layers

    @torch.inference_mode()
    def __call__(self, image):
        semantics = self.multiscale_semantics(image)
        candidates = self.candidates(image)
        y, x = (np.mgrid[:GRID, :GRID].astype(np.float32) + .5) / GRID
        rgb = cv2.resize(np.asarray(image), (GRID, GRID), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).astype(np.float32).transpose(2, 0, 1) / 255. - .5
        features = np.concatenate([semantics, 4 * candidates, 2 * lab, y[None], np.abs(x - .5)[None]])
        return features.astype(np.float16)


def render(image, probabilities):
    scale = min(1., 900 / max(image.size))
    size = tuple(max(1, round(k * scale)) for k in image.size)
    rgb = cv2.resize(np.asarray(image), size, interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    guide = cv2.resize(grey, (GRID, GRID), interpolation=cv2.INTER_AREA)

    def mean(values):
        return cv2.blur(values, (5, 5))

    def upsample(values):
        return cv2.resize(values, size, interpolation=cv2.INTER_LINEAR)

    variance = (mean(guide * guide) - mean(guide) ** 2).clip(0) + .006
    maps = []
    for p in probabilities:
        slope = (mean(guide * p) - mean(guide) * mean(p)) / variance
        intercept = mean(p) - slope * mean(guide)
        maps.append(.65 * (grey * upsample(mean(slope)) + upsample(mean(intercept))) + .35 * upsample(p))
    classes = np.argmax(maps, axis=0).astype(np.uint8)
    return cv2.resize(classes, image.size, interpolation=cv2.INTER_NEAREST)


def save(image, classes, output, source_hash):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(classes).save(output / "classes.png")
    for label, mask in (("wall_solid", classes == 1), ("wall_void", classes == 2)):
        Image.fromarray(mask.astype(np.uint8) * 255).save(output / (label + ".png"))

    view = image.copy()
    view.thumbnail((850, 620))
    small = cv2.resize(classes, view.size, interpolation=cv2.INTER_NEAREST_EXACT)
    rgb = np.array(view)
    for key, color in ((1, (28, 178, 185)), (2, (248, 121, 33))):
        mask = small == key
        rgb[mask] = .46 * rgb[mask] + .54 * np.array(color)
    Image.fromarray(rgb).save(output / "overlay.jpg", quality=91)

    void = int((classes == 2).sum())
    solid = int((classes == 1).sum())
    result = dict(
        source_sha256=source_hash, image_size=list(image.size), wall_void_pixels=void, wall_solid_pixels=solid,
        void_over_total=void / (void + solid) if void + solid else None,
    )
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
