import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from model_assets import assets_root

from .features import FrozenFeatures, upright
from .floor_aperture import refine_floor_apertures
from .floor_veto import apply_veto, feature_grid, read_head
from .geometry_features import Geometry
from .refinement import (align_boundaries, combine_maps, make_heads, probabilities,
                        context_tensor, segment_probabilities)
from .scene_cues import SceneLocator, scene_maps
from .segment import COLORS, ROLES, FusionHead, mask_classes, pack, ratios, role_map
from .storey_geometry import add_supported_openings, storey_geometry

HEAD_SHA256 = "2737124f09f05584c9d13d2ee6b8bad81cb1df16757ad726fe982f77e97d1c97"


class Predictor:

    def __init__(self, device=None):
        package = assets_root() / "slab"
        weights = package / "model.pt"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != HEAD_SHA256:
            raise ValueError("Invalid slab model checksum.")
        weight = torch.load(weights, map_location="cpu", weights_only=True)["base_model"]
        self.heads = make_heads(weight["channels"])
        self.heads.load_state_dict(weight["models"])
        self.heads.eval()
        self.policy = weight["policy"]
        prior = weight["prior_head"]
        self.prior = FusionHead(prior["model"]["reduce.0.weight"].shape[1])
        self.prior.load_state_dict(prior["model"])
        self.prior.eval()
        self.prior_threshold = prior["threshold"]
        self.backend = FrozenFeatures(device=device)
        self.geometry = Geometry(assets_root() / "moge" / "model.pt", device=device)
        self.floor_veto = read_head(package / "floor_veto.npz")
        self.locator = None

    @torch.inference_mode()
    def _roles(self, image):
        device = self.backend.device
        vision = (self.backend.seg,self.backend.dino,self.backend.sam,self.geometry.model)
        for model in vision:
            model.to("cpu")
        if device == "cuda":
            torch.cuda.empty_cache()
        if self.locator is None:
            self.locator = SceneLocator(device)
        else:
            self.locator.model.to(device)
        boxes = self.locator.locate(image)
        evidence = self.locator.presence(image)
        self.locator.model.to("cpu")
        if device == "cuda":
            torch.cuda.empty_cache()
        for model in vision:
            model.to(device)

        feat = self.backend(image)
        surfaces = feat["surfaces"].astype(np.float32)
        camera = {}
        geometry = self.geometry(image,surfaces,camera_out=camera)
        scene = scene_maps(image,boxes,self.backend.mp,self.backend.sam,device,evidence)
        inputs = context_tensor(feat,geometry,scene)
        prob = probabilities(self.heads,inputs[None])[0].numpy()
        present = bool(scene[6:,0,0].astype(np.float16).max() >= .55)
        classes = segment_probabilities(prob,present,self.policy)
        context_roles = align_boundaries(classes,surfaces,image)

        prior_inputs = torch.cat([pack(feat),torch.from_numpy(geometry.astype(np.float32))])
        prior_prob = self.prior(prior_inputs[None]).softmax(1)[0].numpy()
        prior_roles = role_map(mask_classes(prior_prob,self.prior_threshold),surfaces)
        roles = combine_maps(prior_roles,context_roles,present)
        local_planes = storey_geometry(feat,geometry,image)
        roles = add_supported_openings(roles,local_planes)
        proposed = refine_floor_apertures(roles, **camera)
        if np.any(roles != proposed):
            grid = feature_grid(feat, geometry, scene, local_planes)
            roles = apply_veto(roles, proposed, grid, self.floor_veto)
        return roles

    def predict_file(self, photo, output):
        photo = Path(photo)
        source_hash = hashlib.sha256(photo.read_bytes()).hexdigest()
        image = upright(photo)
        roles = self._roles(image)
        native = cv2.resize(roles, image.size, interpolation=cv2.INTER_NEAREST)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(native).save(output / "classes.png")
        solid = np.isin(native, [1, 3])
        opening = np.isin(native, [2, 4])
        Image.fromarray(opening.astype(np.uint8) * 255).save(output / "slab_void.png")
        for i, role in enumerate(ROLES):
            for kind, code in (("solid", 2*i+1), ("void", 2*i+2)):
                mask = (native == code).astype(np.uint8) * 255
                Image.fromarray(mask).save(output / f"{role}_{kind}.png")
        view = image.copy()
        view.thumbnail((900, 680))
        labels = cv2.resize(roles, view.size, interpolation=cv2.INTER_NEAREST)
        rgb = np.array(view)
        active = labels > 0
        rgb[active] = (.53*rgb[active] + .47*COLORS[labels[active]]).astype(np.uint8)
        Image.fromarray(rgb).save(output / "overlay.jpg", quality=88)
        result = {
            "source_sha256": source_hash,
            "image_size": list(image.size),
            "image_plane": ratios(int(solid.sum()), int(opening.sum())),
        }
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
