import json
import re
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from model_assets import assets_root
from .features import GRID


QUESTION = """Look at the architecture of this shopping mall photo. Locate visible openings through a ceiling or floor slab. An opening in the ceiling includes a skylight, an atrium cut-out showing a different storey, or a portion open to the sky. An opening in the floor includes the visible void behind a balustrade where the floor is cut away and one can see a lower storey. Enclose the visible projected area of each actual opening, including the space seen through it. Look for both large openings that extend beyond the image border and smaller openings. Ordinary ceiling light fixtures, mirrors, continuous floors, patterns, shop windows, doorways in vertical walls, rails without a visible void, and decorative ceiling recesses are not openings. Use the layout of the scene. Return [] if no such opening is visible. Otherwise return a JSON array, up to six objects, with keys "role" ("ceiling" or "floor") and "bbox_2d" ([left,top,right,bottom] as integers in 0..1000 of the entire photograph). JSON only."""


def parse_boxes(raw):
    clean = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw.strip())
    items = json.loads(clean)
    if not isinstance(items, list):
        raise ValueError("Expected a list of opening candidates")
    boxes = []
    for item in items:
        role = item["role"].strip().lower()
        box = np.asarray(item["bbox_2d"], dtype=np.float32)
        if role not in ("ceiling", "floor") or box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError("Invalid opening proposal")
        box = box.clip(0, 1000)
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append({"role": role, "box": box.tolist()})
    return boxes


class SceneLocator:
    def __init__(self, device):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        source = assets_root() / "qwen3_vl_2b"
        self.processor = AutoProcessor.from_pretrained(source, local_files_only=True)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            source, dtype=dtype, attn_implementation="sdpa", local_files_only=True).eval().to(self.device)

    @torch.inference_mode()
    def presence(self, image):
        picture = image.copy()
        picture.thumbnail((672,672),Image.Resampling.LANCZOS)
        positive = {
            "floor":"A real cut-out in the walking floor is visible; we can see into a lower level beyond a floor edge or balustrade.",
            "ceiling":"A real opening through the overhead slab is visible: sky through a skylight, an upper storey through a void, or an atrium open above.",
        }
        negative = {
            "floor":"There is no clear cut-out through the floor. Continuous paving, escalator steps by themselves or a shiny floor are not evidence of a hole.",
            "ceiling":"There is no clear opening through the overhead slab. Lights, panels, patterned or recessed ceilings alone do not establish an opening.",
        }
        tokenizer = self.processor.tokenizer
        token_a,token_b = (tokenizer.encode(s,add_special_tokens=False) for s in ("A","B"))
        if len(token_a)!=1 or len(token_b)!=1:
            raise RuntimeError("Expected single-token answer choices")
        scores = {}
        for role in ("floor","ceiling"):
            differences = []
            for choices,sign in (((positive[role],negative[role]),1),((negative[role],positive[role]),-1)):
                query = ("Inspect the whole photograph. Which description is better supported? "
                         f"A: {choices[0]} B: {choices[1]} Answer only A or B.")
                messages=[{"role":"user","content":[{"type":"image","image":picture},{"type":"text","text":query}]}]
                inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                return_dict=True,return_tensors="pt").to(self.device)
                logits=self.model(**inputs,logits_to_keep=1).logits[0,-1].float()
                differences.append(sign*float(logits[token_a[0]]-logits[token_b[0]]))
            scores[role]=float(torch.sigmoid(torch.tensor(np.mean(differences))))
        return scores

    @torch.inference_mode()
    def locate(self, image):
        picture = image.copy()
        picture.thumbnail((896, 896), Image.Resampling.LANCZOS)
        messages = [{"role":"user","content":[{"type":"image","image":picture},
                                              {"type":"text","text":QUESTION}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt").to(self.device)
        for limit in (400, 650):
            ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=limit)
            raw = self.processor.decode(ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            try:
                return parse_boxes(raw)
            except (KeyError, TypeError, ValueError, AttributeError):
                if limit == 650:
                    recovered = []
                    for match in re.finditer(r"\{[^{}]+\}",raw):
                        try:
                            recovered.extend(parse_boxes("["+match.group()+"]"))
                        except (KeyError, TypeError, ValueError, AttributeError):
                            continue
                    return recovered
        raise RuntimeError("Unexpected generation state")


@torch.inference_mode()
def scene_maps(image, candidates, processor, sam, device, evidence):
    maps = np.zeros((8, GRID, GRID), np.float32)
    maps[6] = evidence["ceiling"]
    maps[7] = evidence["floor"]
    unique = {}
    for item in candidates:
        unique.setdefault((item["role"],tuple(item["box"])),item)
    candidates = list(unique.values())[:12]
    if not candidates:
        return maps
    coords = np.array([item["box"] for item in candidates], np.float32) / 1000.
    image_boxes = coords * [image.width, image.height, image.width, image.height]
    inputs = processor(images=image, input_boxes=[image_boxes.tolist()], return_tensors="pt").to(device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device=="cuda"):
        pred = sam(**inputs, multimask_output=True)
    yy, xx = (np.mgrid[:GRID,:GRID].astype("float32")+.5)/GRID
    scores = pred.iou_scores[0].float().cpu().numpy()
    masks = pred.pred_masks[0].float().cpu().numpy()
    for i,(item,(x1,y1,x2,y2)) in enumerate(zip(candidates,coords)):
        k = 0 if item["role"] == "ceiling" else 3
        nx, ny = (xx-x1)/(x2-x1), (yy-y1)/(y2-y1)
        inside = (nx>=0)&(nx<=1)&(ny>=0)&(ny<=1)
        margin = np.minimum.reduce((nx,ny,1-nx,1-ny))
        maps[k] = np.maximum(maps[k],inside)
        maps[k+1] = np.maximum(maps[k+1], np.clip(2*margin,0,1))
        best = int(scores[i].argmax())
        mask = cv2.resize(masks[i,best],(GRID,GRID),interpolation=cv2.INTER_LINEAR)>0
        maps[k+2] = np.maximum(maps[k+2],mask * np.clip(scores[i,best],0,1))
    return maps
