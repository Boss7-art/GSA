from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageOps
from safetensors.torch import load_file
import torch
from groundingdino.config import GroundingDINO_SwinT_OGC
from groundingdino.datasets import transforms
from groundingdino.models import build_model
from groundingdino.util.utils import get_phrases_from_posmap
from segment_anything import SamPredictor, build_sam

from model_assets import assets_root


def colour_masks(rgb):
    red, green, blue = np.moveaxis(rgb, -1, 0)
    warm = (red > blue) & (green < red)
    cool = (blue > red) & (green < blue)
    return warm, cool, ~(warm | cool)


def area(detections, category):
    return sum(np.count_nonzero(mask) for label, mask in detections if category in label)


def colour_ratio(detections, category, colours):
    if not detections:
        return 0
    masks = [mask for label, mask in detections if category in label]
    total = sum(np.count_nonzero(mask) for mask in masks)
    if not total:
        return (0, 0, 0)
    counts = [sum(np.count_nonzero(mask & colour) for mask in masks) for colour in colours]
    return np.asarray(counts) / total


class IndicatorEngine:
    def __init__(self, device="cpu", box_threshold=0.3, text_threshold=0.25):
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        model_dir = assets_root()
        config = SimpleNamespace(**{k: v for k, v in vars(GroundingDINO_SwinT_OGC).items()
                                    if not k.startswith("_")})
        config.device = "cpu"
        config.text_encoder_type = str(model_dir / "bert-base-uncased")
        self.detector = build_model(config)
        state = load_file(str(model_dir / "detector.safetensors"))
        load = self.detector.load_state_dict(state, strict=False)
        if any(not key.startswith("bert.") for key in load.missing_keys):
            raise ValueError(f"Incomplete detector checkpoint: {load.missing_keys}")
        self.detector.eval()
        self.sam = SamPredictor(build_sam(checkpoint=str(model_dir / "sam_vit_h_4b8939.pth")).to(device).eval())
        self.transform = transforms.Compose([
            transforms.RandomResize([800], max_size=1333),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _detections(self, image, tensor, caption, threshold):
        caption = caption.strip().lower().rstrip(".") + "."
        outputs = self.detector(tensor[None], captions=[caption])
        logits = outputs["pred_logits"][0].sigmoid()
        keep = logits.max(dim=1).values > threshold
        boxes = outputs["pred_boxes"][0][keep]
        if not len(boxes):
            return []
        tokenizer = self.detector.tokenizer
        tokens = tokenizer(caption)
        labels = [get_phrases_from_posmap(scores > self.text_threshold, tokens, tokenizer)
                  for scores in logits[keep]]
        height, width = image.shape[:2]
        boxes = boxes * boxes.new_tensor((width, height, width, height))
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        if not self.sam.is_image_set:
            self.sam.set_image(image)
        boxes = self.sam.transform.apply_boxes_torch(boxes, (height, width)).to(self.sam.device)
        masks, _, _ = self.sam.predict_torch(None, None, boxes=boxes, multimask_output=False)
        return list(zip(labels, masks[:, 0].cpu().numpy()))

    @torch.inference_mode()
    def measure(self, photo):
        with Image.open(photo) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        tensor, _ = self.transform(image, None)
        rgb = np.asarray(image)
        self.sam.reset_image()
        colours = colour_masks(rgb)
        pixels = rgb.shape[0] * rgb.shape[1]

        ceiling = self._detections(rgb, tensor, "ceiling. windows in the ceiling", self.box_threshold)
        ceiling_area = area(ceiling, "ceiling")
        result = {"ceiling_colour_fractions": colour_ratio(ceiling, "ceiling", colours)}
        wall_threshold = 0.2
        while True:
            wall = self._detections(rgb, tensor, "wall", wall_threshold)
            enclosure = (area(wall, "wall") + ceiling_area) / pixels if wall else 0
            if enclosure <= 1:
                break
            wall_threshold += 0.5
        result["enclosure_degree"] = enclosure
        result["wall_colour_fractions"] = colour_ratio(wall, "wall", colours)

        objects = self._detections(rgb, tensor, "light. chair. plant. billboard", self.box_threshold)
        for label, name in (("plant", "green_view_index"), ("light", "artificial_lighting"),
                            ("chair", "seating"), ("billboard", "billboards")):
            result[name] = area(objects, label) / pixels
        floor = self._detections(rgb, tensor, "floor", self.box_threshold)
        result["floor_colour_fractions"] = colour_ratio(floor, "floor", colours)
        result["image_colour_fractions"] = np.array([np.count_nonzero(mask) for mask in colours]) / pixels
        return result
