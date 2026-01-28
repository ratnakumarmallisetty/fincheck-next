import torch
from torch.utils.data import DataLoader

from ml.perturbations import apply_perturbations
from .sampler import sample_dataset

def evaluate_model(
    model,
    dataset,
    config,
    batch_size=64,
    device="cpu"
):
    model.eval()
    model.to(device)

    dataset = sample_dataset(
        dataset,
        config.sample_mode,
        config.sample_size,
        config.seed
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if config.apply_perturbations:
                x = apply_perturbations(x, config.perturbation_config)

            logits = model(x)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total
    }
