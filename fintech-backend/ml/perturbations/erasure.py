import torch
from .common import set_seed

def apply_erasure(x: torch.Tensor, percent: float, seed: int):
    """
    percent: 0.0 → 0.5 (fraction of image area)
    """
    if percent <= 0:
        return x

    set_seed(seed)

    batched = x.dim() == 4
    if not batched:
        x = x.unsqueeze(0)

    B, C, H, W = x.shape
    area = int(H * W * percent)
    erase_h = int((area / W) ** 0.5)
    erase_w = int((area / H) ** 0.5)

    for i in range(B):
        y = torch.randint(0, H - erase_h + 1, (1,)).item()
        x0 = torch.randint(0, W - erase_w + 1, (1,)).item()
        x[i, :, y:y+erase_h, x0:x0+erase_w] = 0.0

    return x if batched else x.squeeze(0)
