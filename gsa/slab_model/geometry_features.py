import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from .features import GRID


def geom_channels(depth, normals, intrinsics, surfaces):
    size = (GRID, GRID)
    z = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST_EXACT)
    n = cv2.resize(normals, size, interpolation=cv2.INTER_NEAREST_EXACT)
    valid = np.isfinite(z) & (z > 0)
    med = np.median(z[valid]) if valid.any() else 1.
    z = np.where(valid,z / med,0)
    n = np.nan_to_num(n, nan=0., posinf=0., neginf=0.)
    yy,xx = (np.mgrid[:GRID,:GRID].astype(np.float32) + .5) / GRID
    k = intrinsics
    points = np.stack([(xx-k[0,2])/k[0,0]*z,(yy-k[1,2])/k[1,1]*z,z],axis=-1)
    maps = [valid.astype(np.float32), np.log1p(z), *np.moveaxis(n,-1,0),
            np.clip(points[...,1],-2,2)]
    for i, surface in enumerate(surfaces):
        seeds = points[valid & (surface > .55)]
        residual = np.zeros_like(z)
        support = np.zeros_like(z)
        if len(seeds) >= 24:
            rng = np.random.default_rng(184+i)
            if len(seeds)>2500:
                seeds = seeds[rng.choice(len(seeds),2500,replace=False)]
            tris = seeds[rng.integers(len(seeds),size=(160,3))]
            normal = np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0])
            norm = np.linalg.norm(normal,axis=1)
            normal = normal[norm>1e-5] / norm[norm>1e-5,None]
            tris = tris[norm>1e-5]
            if len(normal):
                offset = (normal*tris[:,0]).sum(1)
                inliers = np.abs(seeds@normal.T-offset)<.035
                best = inliers.sum(0).argmax()
                if inliers[:,best].mean() >= .22:
                    signed = ((points@normal[best])-offset[best])*np.sign(offset[best])
                    residual = np.where(valid,np.clip(signed,-1,1),0)
                    support = valid & (np.abs(signed)<.06)
        maps.extend([residual,support.astype(np.float32)])
    return np.stack(maps).astype(np.float32)


class Geometry:
    def __init__(self, checkpoint, device=None):
        from moge.model.v2 import MoGeModel
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MoGeModel.from_pretrained(checkpoint).eval().to(self.device)

    @torch.inference_mode()
    def camera_input(self, image):
        im=image.copy()
        im.thumbnail((1536,1536),Image.Resampling.LANCZOS)
        rgb=torch.from_numpy(np.array(im)).permute(2,0,1).to(self.device).float()/255.
        res=self.model.infer(rgb,resolution_level=9,apply_mask=True,force_projection=True,use_fp16=self.device=="cuda")
        depth, normals, k = [res[key].float().cpu().numpy() for key in ("depth","normal","intrinsics")]
        return rgb, dict(depth=depth, normals=normals, intrinsics=k)

    @torch.inference_mode()
    def __call__(self, image, surfaces, camera_out=None):
        rgb, camera = self.camera_input(image)
        if camera_out is not None:
            camera_out.update(camera)
        depth, normals, k = (camera[key] for key in ("depth","normals","intrinsics"))
        height,width = rgb.shape[-2:]
        h,w=round(48*np.sqrt(height/width)),round(48*np.sqrt(width/height))
        enc=self.model.encoder(rgb[None],h,w)[0]
        projection=torch.randn(80,enc.shape[0],generator=torch.Generator().manual_seed(7304)).to(enc.device)
        vis=torch.einsum("oc,chw->ohw",projection,enc)
        vis=F.normalize(vis,dim=0)*np.sqrt(80)
        vis=F.interpolate(vis[None],(GRID,GRID),mode="bilinear",align_corners=False)[0].cpu().numpy()
        return np.concatenate([vis,geom_channels(depth,normals,k,surfaces)*3]).astype("float16")
