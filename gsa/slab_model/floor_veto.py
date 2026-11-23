import cv2
import numpy as np


def feature_grid(features, geometry, scene, planes):
    embed=np.asarray(features["embedding"],np.float32)
    embed=embed/np.sqrt(np.mean(embed**2,axis=0,keepdims=True)+1e-6)
    encoded=np.asarray(geometry[:80],np.float32)
    encoded=encoded/np.sqrt(np.mean(encoded**2,axis=0,keepdims=True)+1e-6)
    surface=np.asarray(features["surfaces"],np.float32)
    geom=np.asarray(geometry[80:],np.float32)/3
    cues=np.concatenate([surface, features["prompts"],geom,planes,scene]).astype(np.float32)
    context=np.stack([cv2.GaussianBlur(a,(0,0),3.) for a in cues])
    yy,xx=np.indices(surface.shape[1:],dtype=np.float32)
    yy=(yy+.5)/surface.shape[1]
    xx=np.abs((xx+.5)/surface.shape[2]-.5)
    channels=np.concatenate([embed,encoded,cues,context,yy[None],xx[None]])
    return np.ascontiguousarray(np.nan_to_num(np.moveaxis(channels,0,-1),nan=0.,posinf=0.,neginf=0.))


def read_head(path):
    with np.load(path,allow_pickle=False) as data:
        if str(data["version"])!="floor_veto_v1":
            raise ValueError(f"Unexpected floor veto head: {path}")
        params={k:data[k] for k in ("center","spread","weight","bias")}
    expected={"center":(416,), "spread":(416,), "weight":(416,3), "bias":(3,)}
    if any(v.shape!=expected[k] or not np.isfinite(v).all() for k,v in params.items()) or np.any(params["spread"]<=0):
        raise ValueError(f"Invalid floor veto coefficients: {path}")
    return params


def apply_veto(original, proposed, grid, head):
    result=original.copy()
    selected=(original==2)&(proposed==1)
    if not selected.any():
        return result
    x=(grid-head["center"])/head["spread"]
    logits=x @ head["weight"] + head["bias"]
    probs=np.exp(logits-logits.max(axis=-1,keepdims=True))
    opaque=probs[:,:,1]/probs.sum(axis=-1)
    opaque=cv2.resize(opaque,(original.shape[1],original.shape[0]),interpolation=cv2.INTER_LINEAR)
    confident=selected&(opaque>=.75)
    if not confident.any():
        return result
    _,ids=cv2.connectedComponents(confident.astype(np.uint8),connectivity=8)
    adjacent=cv2.dilate((original==1).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    roots=np.unique(ids[confident&adjacent])
    if len(roots):
        result[np.isin(ids,roots)]=1
    return result
