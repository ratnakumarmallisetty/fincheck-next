import torch
import torchvision.transforms.functional as F
from .common import set_seed

def apply_blur(x: torch.Tensor, sigma: float, seed: int):
    """
    x: Tensor [C,H,W] or [B,C,H,W]
    sigma: 0.0 (off) → ~2.0
    """
    if sigma <= 0:
        return x

    set_seed(seed)

    batched = x.dim() == 4
    if not batched:
        x = x.unsqueeze(0)

    out = []
    for img in x:
        out.append(F.gaussian_blur(img, kernel_size=3, sigma=sigma))

    out = torch.stack(out)
    return out if batched else out.squeeze(0)
