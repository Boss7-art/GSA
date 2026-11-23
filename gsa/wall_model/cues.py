import re

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from model_assets import assets_root
from slab_model.features import GRID


GLASS = ("Find the glass panes in the vertical building or storefront facades in this photo. "
         "Include floor-to-ceiling glass, printed, frosted, tinted or mirror-like shop windows and glass doors. "
         "Enclose the entire pane out to its frames, including reflections or merchandise visible through it. "
         "Exclude atrium guardrails, freestanding display cases and floor or ceiling reflections.")
OPEN = ("Find the unglazed openings through vertical storefront walls, especially broad open shop fronts. "
        "Enclose the whole projected aperture between its jambs and under its lintel, including the entire view "
        "of shelving, customers or furnishings inside. Exclude floor cutouts, escalators, central corridors, "
        "atrium railings and opaque advertisements.")
SOLID = ("Find the opaque structural portions of vertical storefront facades facing this circulation space: "
         "piers, sign fascias, metal frames, masonry walls and fitted opaque wall panels. "
         "Exclude glass, clear doorways, railings, floor slabs, soffits and freestanding signs.")
SUFFIX = (" Return only a JSON list of bounding boxes in [x_min,y_min,x_max,y_max] coordinates "
          "normalized from 0 to 1000 in the full image. At most six boxes. Return [] if none.")


def parse_boxes(text):
    pattern = r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    boxes = []
    for groups in re.findall(pattern, text):
        box = np.clip(np.array(groups, np.float32), 0, 1000)
        if box[2] > box[0] and box[3] > box[1] and box.tolist() not in boxes:
            boxes.append(box.tolist())
    return boxes[:6]


class StorefrontVision:
    def __init__(self, device):
        self.device = device
        path = assets_root() / "qwen3_vl_2b"
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, local_files_only=True, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa",
        ).eval().to(device)

    def descriptors(self, inputs):
        embeds, intermediate = self.model.get_image_features(inputs.pixel_values, inputs.image_grid_thw)
        _, h, w = inputs.image_grid_thw[0].tolist()
        merge = self.model.config.vision_config.spatial_merge_size
        h, w = h // merge, w // merge
        layers = []
        for seed, tokens in ((419, intermediate[1]), (461, embeds[0])):
            x = tokens.float()
            x = x.sign() * x.abs().sqrt()
            x = .55 * F.normalize(x, dim=1) + .45 * F.normalize(x - x.mean(0), dim=1)
            matrix = torch.randn((x.shape[1], 96), generator=torch.Generator().manual_seed(seed)).to(x.device)
            x = (x @ matrix).T.reshape(1, 96, h, w)
            layers.append(F.interpolate(x, (GRID,GRID), mode="bilinear", align_corners=False)[0])
        return torch.cat(layers).cpu().numpy().astype(np.float16)

    @torch.inference_mode()
    def __call__(self, image):
        picture = image.copy()
        picture.thumbnail((896, 896), Image.Resampling.LANCZOS)
        proposals = []
        descriptors = None
        for question in (GLASS, OPEN, SOLID):
            message = [{"role": "user", "content": [{"type":"image","image":picture},
                        {"type":"text","text":question + SUFFIX}]}]
            inputs = self.processor.apply_chat_template(message, tokenize=True, add_generation_prompt=True,
                        return_dict=True, return_tensors="pt").to(self.device)
            if descriptors is None:
                descriptors = self.descriptors(inputs)
            ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=168)
            answer = self.processor.decode(ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            proposals.append(parse_boxes(answer))
        return descriptors, proposals
