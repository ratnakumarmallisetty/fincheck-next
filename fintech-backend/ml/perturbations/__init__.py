from .blur import apply_blur
from .rotation import apply_rotation
from .gaussian_noise import apply_gaussian_noise
from .erasure import apply_erasure

def apply_perturbations(x, config: dict):
    seed = config.get("seed", 42)

    x = apply_blur(x, config.get("blur", 0.0), seed)
    x = apply_rotation(x, config.get("rotation", 0.0), seed)
    x = apply_gaussian_noise(x, config.get("gaussian_noise", 0.0), seed)
    x = apply_erasure(x, config.get("erasure", 0.0), seed)

    return x
