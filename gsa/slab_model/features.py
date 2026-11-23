import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.ops import nms
from transformers import (AutoImageProcessor, AutoModelForSemanticSegmentation, AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor)
from model_assets import assets_root


GRID = 128


PROMPTS = [
    "sky. skylight. escalator opening. stairwell. floor void. opening to lower floor.",
    "glass roof. ceiling opening. atrium void. roof window. glass canopy.",
    "opening in floor. atrium railing. staircase. escalator. balcony opening.",
]


class DenseVision:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        assets = assets_root()
        seg = assets / "segformer_b0"
        det = assets / "dino_tiny"
        sam = assets / "sam21_small"
        self.sp = AutoImageProcessor.from_pretrained(seg, use_fast=False, local_files_only=True)
        self.seg = AutoModelForSemanticSegmentation.from_pretrained(seg, local_files_only=True).eval().to(self.device)
        self.dp = AutoProcessor.from_pretrained(det, local_files_only=True)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(
            det, disable_custom_kernels=True, local_files_only=True
        ).eval().to(self.device)
        self.mp = Sam2Processor.from_pretrained(sam, local_files_only=True)
        self.sam = Sam2Model.from_pretrained(sam, local_files_only=True).eval().to(self.device)
        self.class_ids = {str(v).lower(): int(k) for k, v in self.seg.config.id2label.items()}


class FrozenFeatures(DenseVision):
    @torch.inference_mode()
    def __call__(self, im):
        embedding = []
        hook = self.seg.decode_head.classifier.register_forward_pre_hook(
            lambda module, args: embedding.append(args[0].detach()))
        try:
            seg = self.seg(**self.sp(images=im, return_tensors="pt").to(self.device)).logits[0]
        finally:
            hook.remove()
        logits = seg.float().softmax(0)
        probs = logits[[self.class_ids["floor"], self.class_ids["ceiling"]]].cpu().numpy()
        scores = np.zeros((len(PROMPTS), GRID, GRID), dtype=np.float32)
        embeddings = None
        for group, prompt in enumerate(PROMPTS):
            inp = self.dp(images=im, text=prompt, return_tensors="pt").to(self.device)
            det = self.dino(**inp)
            out = self.dp.post_process_grounded_object_detection(
                det, inp.input_ids, threshold=.15, text_threshold=.25, target_sizes=[(im.height, im.width)])[0]
            boxes = out["boxes"]
            boxes[:, 0::2].clamp_(0, im.width)
            boxes[:, 1::2].clamp_(0, im.height)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes, conf = boxes[valid], out["scores"][valid]
            if not len(boxes):
                continue
            keep = nms(boxes, conf, .80)[:16]
            for idx in keep.split(4):
                batch = self.mp(images=im, input_boxes=[boxes[idx].cpu().tolist()], return_tensors="pt").to(self.device)
                if embeddings is not None:
                    batch.pop("pixel_values")
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
                    result = self.sam(**batch, image_embeddings=embeddings, multimask_output=False)
                embeddings = result.image_embeddings
                raw = result.pred_masks[0, :, 0].float().cpu().numpy()
                for mask, score in zip(raw, conf[idx].tolist()):
                    support = cv2.resize(mask, (GRID, GRID), interpolation=cv2.INTER_LINEAR) > 0
                    scores[group][support] = np.maximum(scores[group][support], score)
        emb = embedding[0][0].float().cpu().numpy()
        if emb.shape[1:] != (GRID, GRID):
            raise ValueError("Unexpected semantic feature grid.")
        return dict(embedding=emb.astype(np.float16), prompts=scores.astype(np.float16),
                    surfaces=probs.astype(np.float16))


def upright(path):
    with Image.open(path) as src:
        return ImageOps.exif_transpose(src).convert("RGB")
