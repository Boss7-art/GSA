import cv2
import numpy as np
import torch

from slab_model.geometry_features import Geometry
from model_assets import assets_root
from slab_model.features import GRID


class SurfaceFeatures:
    def __init__(self, device, sam, processor):
        self.device = device
        self.geometry = Geometry(assets_root() / "moge" / "model.pt", device)
        self.processor = processor
        self.sam = sam

    @torch.inference_mode()
    def __call__(self, image, base, groups, camera_out=None):
        surfaces = np.array(base[257:259], np.float32) / 3
        geometry = self.geometry(image, surfaces, camera_out=camera_out)
        yy, xx = (np.mgrid[:GRID,:GRID].astype(np.float32) + .5) / GRID
        maps = np.zeros((3,4,GRID,GRID), np.float32)
        tagged = [(k,np.array(b)/1000) for k,boxes in enumerate(groups) for b in boxes]
        embedding = None
        for start in range(0,len(tagged),4):
            chunk = tagged[start:start+4]
            coords = [list(b * [image.width,image.height,image.width,image.height]) for _,b in chunk]
            inputs = self.processor(images=image, input_boxes=[coords], return_tensors="pt").to(self.device)
            if embedding is not None:
                inputs.pop("pixel_values")
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16,
                                enabled=self.device=="cuda" and torch.cuda.is_bf16_supported()):
                result = self.sam(**inputs, image_embeddings=embedding, multimask_output=True)
            embedding = result.image_embeddings
            scores = result.iou_scores[0].float().cpu().numpy()
            masks = result.pred_masks[0].float().cpu().numpy()
            for i,(group,box) in enumerate(chunk):
                nx = (xx-box[0]) / (box[2]-box[0])
                ny = (yy-box[1]) / (box[3]-box[1])
                margin = np.minimum.reduce([nx,ny,1-nx,1-ny])
                maps[group,0] = np.maximum(maps[group,0],margin>=0)
                maps[group,1] = np.maximum(maps[group,1],np.clip(4*margin,0,1))
                best = int(scores[i].argmax())
                logits = cv2.resize(masks[i,best],(GRID,GRID),interpolation=cv2.INTER_LINEAR)
                maps[group,2] = np.maximum(maps[group,2],(logits>0)*np.clip(scores[i,best],0,1))
                for j in range(len(scores[i])):
                    candidate = cv2.resize(masks[i,j],(GRID,GRID),interpolation=cv2.INTER_LINEAR)
                    maps[group,3] = np.maximum(maps[group,3],(candidate>0)*np.clip(scores[i,j],0,1))
        return geometry, maps.reshape(-1,GRID,GRID).astype(np.float16)
