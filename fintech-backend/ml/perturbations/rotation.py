import torch
import torchvision.transforms.functional as F
from .common import set_seed

def apply_rotation(x: torch.Tensor, degrees: float, seed: int):
    if degrees == 0:
        return x

    set_seed(seed)

    batched = x.dim() == 4
    if not batched:
        x = x.unsqueeze(0)

    out = []
    for img in x:
        out.append(F.rotate(img, angle=degrees))

    out = torch.stack(out)
    return out if batched else out.squeeze(0)
