import cv2
import numpy as np
from scipy.special import expit


def wall_band(base, geometry):
    floor, ceiling = base[257:259].astype(np.float32) / 3.
    valid, log_depth = geometry[80:82].astype(np.float32) / 3.
    depth = np.expm1(log_depth)
    h, w = floor.shape
    yy = np.broadcast_to(np.arange(h)[:,None], (h,w))
    cols = np.arange(w)
    floor_seed = cv2.blur(floor,(5,7)) > .45
    ceil_seed = cv2.blur(ceiling,(5,7)) > .45
    floor_y = np.where(floor_seed,yy,h).min(0)
    ceil_y = np.where(ceil_seed,yy,-1).max(0)
    active = (ceil_y >= 0) & (floor_y < h) & (floor_y > ceil_y+2)
    low = np.clip(ceil_y,0,h-1)
    high = np.clip(floor_y,0,h-1)
    t = (yy-low)/np.maximum(high-low,1)
    inside = (t>0) & (t<1) & active
    inverse = 1/np.maximum(depth,.03)
    line = inverse[low,cols] + np.clip(t,0,1)*(inverse[high,cols]-inverse[low,cols])
    valid_band = inside & (valid>.5) & (valid[low,cols]>.5) & (valid[high,cols]>.5)
    residual = np.abs(inverse-line)/np.maximum(line,.03)
    coherent = valid_band * np.exp(-np.minimum(residual,5)/.25)
    gradient = np.minimum(np.abs(np.gradient(log_depth,axis=0))/.08,1)
    connected = inside & (gradient < .5)
    return np.stack([inside, active+np.zeros_like(t), np.clip(t,-1,2)*active,
                     valid_band,coherent,connected,
                     np.clip((yy-ceil_y)/h,-1,1)*(ceil_y>=0),
                     np.clip((floor_y-yy)/h,-1,1)*(floor_y<h),gradient]).astype(np.float32)


def features(base, vision, geometry, boxes):
    rng = np.random.default_rng(19013)
    blocks = []
    for block, out in ((base[:256],48),(vision,64),(geometry[:80],28)):
        projection = rng.standard_normal((out,len(block)),dtype=np.float32)/np.sqrt(len(block))
        projected = (projection @ block.astype(np.float32).reshape(len(block),-1)).reshape(out,*base.shape[1:])
        blocks.append(projected)
    cues = np.concatenate([base[256:],geometry[80:],boxes]).astype(np.float32)
    cues = np.concatenate([cues,wall_band(base,geometry)])
    def pool(size):
        return np.moveaxis(cv2.blur(np.moveaxis(cues,0,-1),size),-1,0)
    return np.concatenate([*blocks,cues,pool((5,19)),pool((19,5))]).astype(np.float32)


def design(features, params):
    normalized = np.clip((features-params["center"])/params["scale"],-4,4)
    hidden = np.maximum(normalized @ params["projection"] + params["bias"],0)
    return np.column_stack([normalized,hidden,np.ones(len(normalized),np.float32)])


def probabilities(raw_features, facade_probabilities, params):
    channels,h,w = raw_features.shape
    rounded = raw_features.astype(np.float16).astype(np.float32)
    x = design(rounded.reshape(channels,-1).T, params)
    layout_logits = (x @ params["coef"].T).T.reshape(2,h,w)
    wall = np.clip(1-facade_probabilities[0],.0001,.9999)
    opening = np.clip(facade_probabilities[2]/np.maximum(wall,.0001),.0001,.9999)
    facade_logits = np.stack([np.log(wall/(1-wall)),np.log(opening/(1-opening))])
    wall, opening = expit(.45*facade_logits + .55*layout_logits)
    return np.stack([1-wall, wall*(1-opening),wall*opening]).astype(np.float32)
