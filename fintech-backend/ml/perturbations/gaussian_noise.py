import torch
from .common import set_seed

def apply_gaussian_noise(x: torch.Tensor, sigma: float, seed: int):
    if sigma <= 0:
        return x

    set_seed(seed)
    noise = torch.randn_like(x) * sigma
    return torch.clamp(x + noise, 0.0, 1.0)
