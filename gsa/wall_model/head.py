import torch
from torch import nn
from torch.nn import functional as F

BASE = 286
VISION = 192
GEOMETRY = 90
PROPOSALS = 12


def pack(base, vision, geometry, proposals):
    blocks = [torch.as_tensor(a).half() for a in (base,vision,geometry,proposals)]
    if [v.shape[0] for v in blocks] != [BASE,VISION,GEOMETRY,PROPOSALS]:
        raise ValueError(f"Unexpected feature shapes: {[v.shape for v in blocks]}")
    return torch.cat(blocks)


def separate(x):
    base, vision, geometry, proposals = torch.split(x, (BASE,VISION,GEOMETRY,PROPOSALS), dim=1)
    cues = torch.cat([base[:,256:], geometry[:,80:], 2 * proposals], dim=1)
    return base[:,:256], vision, geometry[:,:80], cues


def projection(channels, out):
    return nn.Sequential(nn.Conv2d(channels,out,1),nn.GroupNorm(4,out),nn.GELU())


class FacadeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([projection(c,d) for c,d in ((256,16),(192,28),(80,16),(52,20))])
        self.local = nn.Sequential(nn.Conv2d(80,40,1),nn.GroupNorm(4,40),nn.GELU())
        self.fusion = nn.Sequential(nn.Conv2d(200,40,3,padding=1), nn.GroupNorm(4,40), nn.GELU(),
                                    nn.Dropout2d(.15), nn.Conv2d(40,2,1))

    def forward(self, x):
        local = self.local(torch.cat([m(v) for m,v in zip(self.encoders,separate(x))],1))
        context = [local, F.avg_pool2d(local,9,1,4,count_include_pad=False),
                   F.avg_pool2d(local,(3,25),1,(1,12),count_include_pad=False),
                   F.avg_pool2d(local,(25,3),1,(12,1),count_include_pad=False)]
        coarse = F.avg_pool2d(local,16,16)
        context.append(F.interpolate(coarse,local.shape[-2:],mode="bilinear",align_corners=False))
        return self.fusion(torch.cat(context,1))


def events(logits):
    wall, through = logits.sigmoid().unbind(1)
    return torch.stack([1-wall, wall*(1-through), wall*through],dim=1)


def linear_features(x):
    seg, vision, geometry, cues = separate(x)
    return torch.cat([seg,vision,geometry,cues,
                      F.avg_pool2d(cues,(5,19),1,(2,9),count_include_pad=False),
                      F.avg_pool2d(cues,(19,5),1,(9,2),count_include_pad=False)],dim=1)


@torch.no_grad()
def ridge_logits(ridge,x):
    f = linear_features(x)
    f = ((f-ridge["center"][None,:,None,None])/ridge["scale"][None,:,None,None]).clamp(-6,6)
    coef = ridge["coef"]
    return 4 * F.conv2d(f,coef[:,:-1,None,None],coef[:,-1])


@torch.no_grad()
def experts(head, ridge, x):
    x = x[None].float()
    a = (head(x)+head(x.flip((-1,))).flip((-1,)))/2
    b = ridge_logits(ridge,x)
    return a[0],b[0]


def combine(neural,linear,rule):
    w,domain,opening = rule
    bias = neural.new_tensor([domain,opening])[:,None,None]
    logits = w*neural + (1-w)*linear + bias
    return events(logits[None])[0]
