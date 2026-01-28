from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import torch
import torch.nn as nn

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from ml.benchmark import BenchmarkConfig, evaluate_model

router = APIRouter()


# -------- Request Schema --------
class BenchmarkRequest(BaseModel):
    dataset: str                    # "mnist"
    sample_mode: str                # "full" | "random"
    sample_size: Optional[int] = None
    seed: int = 42

    apply_perturbations: bool = False
    perturbation_config: Optional[dict] = None


# -------- TEMP Model Loader --------
def load_model(dataset: str):
    if dataset == "mnist":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    else:
        raise ValueError("Only MNIST supported for now")


# -------- Dataset Loader --------
def load_dataset(dataset: str):
    if dataset == "mnist":
        return MNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )
    else:
        raise ValueError("Only MNIST supported for now")


# -------- API Endpoint --------
@router.post("/benchmark")
def run_benchmark(req: BenchmarkRequest):
    model = load_model(req.dataset)
    dataset = load_dataset(req.dataset)

    cfg = BenchmarkConfig(
        dataset=req.dataset,
        sample_mode=req.sample_mode,
        sample_size=req.sample_size,
        seed=req.seed,
        apply_perturbations=req.apply_perturbations,
        perturbation_config=req.perturbation_config
    )

    result = evaluate_model(model, dataset, cfg)

    return {
        "dataset": req.dataset,
        "metrics": result
    }
