import random
from torch.utils.data import Subset

def sample_dataset(dataset, mode: str, n: int | None, seed: int):
    if mode == "full":
        return dataset

    if mode != "random":
        raise ValueError("sample_mode must be 'full' or 'random'")

    if not n or n <= 0:
        raise ValueError("sample_size must be > 0 for random sampling")

    random.seed(seed)
    indices = random.sample(range(len(dataset)), n)
    return Subset(dataset, indices)
