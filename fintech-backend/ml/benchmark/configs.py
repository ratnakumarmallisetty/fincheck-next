from dataclasses import dataclass
from typing import Optional

@dataclass
class BenchmarkConfig:
    dataset: str               # "mnist" | "cifar"
    sample_mode: str           # "full" | "random"
    sample_size: Optional[int] # used only if random
    seed: int = 42
    apply_perturbations: bool = False
    perturbation_config: Optional[dict] = None
