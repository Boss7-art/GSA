import hashlib
from pathlib import Path

import numpy as np
import torch

from model_assets import assets_root
from slab_model.features import upright
from .model import WallFeatures, render, save
from .layout import features as layout_features, probabilities
from .cues import StorefrontVision
from .spatial import SurfaceFeatures
from .head import FacadeHead, combine, experts, pack


class WallPredictor:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = assets_root() / "wall" / "model.pt"
        trained = torch.load(weights, map_location="cpu", weights_only=True)
        if trained["architecture"] != "facade_v2":
            raise ValueError("Unexpected wall segmentation checkpoint architecture")
        self.head = FacadeHead().to(self.device).eval()
        self.head.load_state_dict(trained["head"])
        self.ridge = {k:v.to(self.device) for k,v in trained["ridge"].items()}
        self.rule = trained["rule"]
        with np.load(weights.with_name("layout_model.npz"), allow_pickle=False) as data:
            self.layout = {key: data[key] for key in ("center", "scale", "projection", "bias", "coef")}
        self.vision = StorefrontVision(self.device)
        self.vision.model.to("cpu")
        if self.device=="cuda":
            torch.cuda.empty_cache()
        self.base = WallFeatures(device=self.device)
        self.surface = SurfaceFeatures(self.device,sam=self.base.sam,processor=self.base.mp)

    @torch.inference_mode()
    def predict_file(self, photo, output, camera_out=None):
        source_hash = hashlib.sha256(Path(photo).read_bytes()).hexdigest()
        image = upright(photo)
        dense = (self.base.seg,self.base.dino,self.base.sam,self.surface.geometry.model)
        for model in dense:
            model.to("cpu")
        if self.device=="cuda":
            torch.cuda.empty_cache()
        self.vision.model.to(self.device)
        descriptors, boxes = self.vision(image)
        self.vision.model.to("cpu")
        if self.device=="cuda":
            torch.cuda.empty_cache()
        for model in dense:
            model.to(self.device)
        base = self.base(image)
        geometry, proposals = self.surface(image,base,boxes,camera_out=camera_out)
        features = pack(base,descriptors,geometry,proposals).to(self.device)
        result = combine(*experts(self.head,self.ridge,features),self.rule).cpu().numpy()
        features = layout_features(base, descriptors, geometry, proposals)
        result = probabilities(features, result, self.layout)
        return save(image, render(image, result), output, source_hash)
