import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .features import GRID


ROLES = ("floor", "ceiling")


COLORS = np.array([[0, 0, 0], [44, 167, 176], [246, 81, 65],
                   [92, 117, 223], [255, 166, 35]], dtype=np.uint8)


def pack(feat):
    yy, xx = np.mgrid[:GRID, :GRID].astype(np.float32) / (GRID - 1)
    sem = feat["embedding"].astype(np.float32)
    sem /= np.sqrt(np.mean(sem ** 2, axis=0, keepdims=True) + 1.e-6)
    priors = np.concatenate([feat["surfaces"], feat["prompts"], yy[None],
                             np.abs(xx - .5)[None], (yy - .5)[None] ** 2], axis=0)
    return torch.from_numpy(np.concatenate([sem, priors * 3]).astype(np.float32))


class FusionHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(channels, 32, 1), nn.GroupNorm(4, 32), nn.GELU())
        self.classifier = nn.Sequential(nn.Conv2d(96, 32, 3, padding=1),
                                        nn.GroupNorm(4, 32), nn.GELU(), nn.Dropout2d(.12),
                                        nn.Conv2d(32, 3, 1))

    def forward(self, x):
        z = self.reduce(x)
        context = [F.avg_pool2d(z, k, 1, k // 2, count_include_pad=False) for k in (7, 21)]
        return self.classifier(torch.cat([z, *context], dim=1))


def mask_classes(prob, threshold):
    classes = (prob[1] > prob[0]).astype(np.uint8)
    classes[prob[2] >= threshold] = 2
    return classes


def role_map(classes, surfaces):
    seeds = [(p > .40) for p in surfaces]
    yy = np.broadcast_to(np.linspace(0, 1, classes.shape[0])[:, None], classes.shape)
    costs = []
    for i, (seed, p) in enumerate(zip(seeds, surfaces)):
        if seed.sum() >= 8:
            dist = cv2.distanceTransform((~seed).astype("uint8"), cv2.DIST_L2, 5) / classes.shape[0]
        else:
            dist = np.full(classes.shape, 1.)
        costs.append(dist - .25 * p + .14 * ((1 - yy) if i == 0 else yy))
    ceil = costs[1] < costs[0]
    return np.where(classes > 0, classes + 2 * ceil, 0).astype("uint8")


def ratios(solid, opening):
    return {"solid_pixels": int(solid), "opening_pixels": int(opening),
            "void_over_total": opening / (solid + opening) if solid + opening else None,
            "void_over_solid": opening / solid if solid else None}
