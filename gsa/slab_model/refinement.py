import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .segment import FusionHead, pack, role_map


def context_tensor(features, geometry, scene):
    scene = np.asarray(scene,dtype=np.float16).astype(np.float32)
    surfaces = features["surfaces"].astype(np.float32)
    near_floor = role_map(np.ones(scene.shape[1:],np.uint8),surfaces) == 1
    local_evidence = np.where(near_floor,scene[7],scene[6])
    evidence_maps = np.stack((local_evidence,scene[0]*scene[6],scene[3]*scene[7]))
    return torch.cat((pack(features),torch.from_numpy(geometry.astype("float32")),
                      torch.from_numpy(3*scene),torch.from_numpy(3*evidence_maps)))


class ContextHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(channels,40,1),nn.GroupNorm(5,40),nn.GELU())
        self.local = nn.Sequential(nn.Conv2d(40,32,3,padding=1),nn.GroupNorm(4,32),nn.GELU())
        self.mid = nn.Sequential(nn.Conv2d(40,32,3,padding=1),nn.GroupNorm(4,32),nn.GELU())
        self.wide = nn.Sequential(nn.Conv2d(40,24,3,padding=1),nn.GroupNorm(4,24),nn.GELU(),
                                  nn.Conv2d(24,24,3,padding=1),nn.GELU())
        self.out = nn.Sequential(nn.Conv2d(88,32,3,padding=1),nn.GroupNorm(4,32),nn.GELU(),
                                 nn.Dropout2d(.1),nn.Conv2d(32,3,1))

    def forward(self, values):
        z = self.project(values)
        mid = self.mid(F.avg_pool2d(z,4))
        wide = self.wide(F.avg_pool2d(z,12,stride=8,padding=2,count_include_pad=False))
        context = [F.interpolate(x,z.shape[-2:],mode="bilinear",align_corners=False) for x in (mid,wide)]
        return self.out(torch.cat([self.local(z),*context],dim=1))


def make_heads(channels):
    return nn.ModuleList([FusionHead(channels),ContextHead(channels)])


@torch.inference_mode()
def probabilities(models, x):
    predictions = []
    for model in models:
        direct = model(x).softmax(1)
        reflected = model(x.flip((-1,))).softmax(1).flip((-1,))
        predictions.append((direct + reflected) / 2)
    return torch.stack(predictions).mean(0)


def segment_probabilities(prob, scene_present, policy):
    opening = prob[2]
    low = policy["low"] if scene_present else policy["empty_low"]
    strong = min(.95, low + policy["core_delta"])
    weak = (opening >= low).astype("uint8")
    components, labels, stats, _ = cv2.connectedComponentsWithStats(weak,connectivity=8)
    core_counts = np.bincount(labels[opening >= strong],minlength=components)
    keep = (core_counts >= 3) & (stats[:,cv2.CC_STAT_AREA] >= 12)
    keep[0] = False
    occupied = keep[labels]
    result = (prob[1] > prob[0]).astype("uint8")
    result[occupied] = 2
    return result


def align_boundaries(classes, surfaces, image):
    coarse = role_map(classes, surfaces)
    width, height = image.size
    scale = min(1., 640 / max(width, height))
    size = max(1, round(scale * width)), max(1, round(scale * height))
    rgb = cv2.resize(np.asarray(image), size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.
    coarse_rgb = cv2.resize(rgb, (coarse.shape[1], coarse.shape[0]), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[:size[1], :size[0]].astype(np.float32)
    xx = np.clip((xx + .5) * coarse.shape[1] / size[0] - .5, 0, coarse.shape[1] - 1)
    yy = np.clip((yy + .5) * coarse.shape[0] / size[1] - .5, 0, coarse.shape[0] - 1)
    ix, iy = xx.astype(int), yy.astype(int)
    fx, fy = xx - ix, yy - iy
    votes = np.zeros((*yy.shape,5),np.float32)
    for dx, xweight in ((0,1-fx),(1,fx)):
        for dy, yweight in ((0,1-fy),(1,fy)):
            cx = np.minimum(coarse.shape[1]-1, ix+dx)
            cy = np.minimum(coarse.shape[0]-1, iy+dy)
            error = ((rgb-coarse_rgb[cy,cx])**2).sum(axis=2)
            weight = xweight*yweight*(.05 + np.exp(-error/(2*.12**2)))
            for k in range(5):
                votes[:,:,k] += weight*(coarse[cy,cx]==k)
    return votes.argmax(axis=2).astype(np.uint8)


def combine_maps(semantic_roles, context_roles, has_scene_evidence):
    if not has_scene_evidence or not np.isin(context_roles,[2,4]).any():
        return context_roles.copy()
    height,width = context_roles.shape
    semantic_roles = cv2.resize(semantic_roles,(width,height),interpolation=cv2.INTER_NEAREST)
    first = np.isin(semantic_roles,[2,4])
    second = np.isin(context_roles,[2,4])
    if not semantic_roles.any():
        return context_roles.copy()
    fractions = [first.sum()/np.count_nonzero(semantic_roles),second.sum()/np.count_nonzero(context_roles)]
    target = np.mean(fractions)
    common = first & second
    disputed = first ^ second
    if not disputed.any():
        return context_roles.copy()
    background = context_roles.copy()
    background[second] = 0
    semantic_solid = np.isin(semantic_roles,[1,3])
    background[second & semantic_solid] = semantic_roles[second & semantic_solid]
    def signed(mask):
        limit = max(2.,min(height,width)*.06)
        inner = cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
        outer = cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,5)
        return np.clip(inner-outer,-limit,limit)
    score = .65*signed(second)+.35*signed(first)
    indices = np.flatnonzero(disputed)
    order = indices[np.argsort(-score.ravel()[indices],kind="stable")]
    additions = np.concatenate(([0],np.cumsum(background.ravel()[order]==0)))
    num = int(common.sum())+np.arange(len(order)+1)
    den = np.count_nonzero((background!=0)|common)+additions
    fractions = np.divide(num,den,out=np.zeros_like(num,dtype=float),where=den>0)
    keep = int(np.argmin(np.abs(fractions-target)))
    opening = common.copy()
    opening.ravel()[order[:keep]] = True
    assign_semantic = opening & ~second
    background[assign_semantic] = semantic_roles[assign_semantic]
    background[opening & second] = context_roles[opening & second]
    return background
