import time
import random
import io
import torch
import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.transforms import GaussianBlur

from model_def import MNISTCNN

# ==================================================
# GLOBAL TORCH CONFIG
# ==================================================
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ==================================================
# APP SETUP
# ==================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FILES = [
    "baseline_mnist.pth",
    "kd_mnist.pth",
    "lrf_mnist.pth",
    "pruned_mnist.pth",
    "quantized_mnist.pth",
    "ws_mnist.pth",
]

# ==================================================
# SEED
# ==================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==================================================
# MODEL LOADING
# ==================================================
MNIST_MODELS = {}

@app.on_event("startup")
def load_models():
    for f in MODEL_FILES:
        model = MNISTCNN().to(DEVICE)
        model.load_state_dict(
            torch.load(MODEL_DIR / f, map_location=DEVICE),
            strict=False,
        )
        model.eval()
        MNIST_MODELS[f] = model
    print("✅ MNIST models loaded")

# ==================================================
# TRANSFORMS
# ==================================================
CLEAN = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

def NOISY(std=0.2):
    return transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.clamp(
            x + std * torch.randn_like(x), 0.0, 1.0
        )),
    ])

def BLUR():
    return transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((28, 28)),
        GaussianBlur(5, 1.0),
        transforms.ToTensor(),
    ])

def NOISY_BLUR(std=0.2):
    return transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((28, 28)),
        GaussianBlur(5, 1.0),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.clamp(
            x + std * torch.randn_like(x), 0.0, 1.0
        )),
    ])

# ==================================================
# INFERENCE (SINGLE RUN)
# ==================================================
@torch.inference_mode()
def run_batch(images):
    batch = torch.stack(images).to(DEVICE)
    out = {}

    for name, model in MNIST_MODELS.items():
        start = time.perf_counter()
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)

        out[name] = {
            "latency_ms": round((time.perf_counter() - start) * 1000 / len(batch), 3),
            "confidence_percent": round(
                probs.max(dim=1).values.mean().item() * 100, 2
            ),
            "entropy": round(
                float(-(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()), 4
            ),
            "stability": round(float(logits.std()), 4),
            "ram_mb": 0.0,
        }

    return out

# ==================================================
# INFERENCE (MULTI RUN – NOISY)
# ==================================================
def run_noisy_multi_eval(build_fn, runs=5):
    acc = {k: [] for k in MNIST_MODELS}

    for r in range(runs):
        set_seed(42 + r)
        images = build_fn()
        res = run_batch(images)

        for m, v in res.items():
            acc[m].append(v)

    final = {}
    for m, vals in acc.items():
        final[m] = {
            "latency_mean": round(np.mean([x["latency_ms"] for x in vals]), 3),
            "latency_std": round(np.std([x["latency_ms"] for x in vals]), 3),
            "confidence_mean": round(np.mean([x["confidence_percent"] for x in vals]), 2),
            "confidence_std": round(np.std([x["confidence_percent"] for x in vals]), 2),
            "entropy_mean": round(np.mean([x["entropy"] for x in vals]), 4),
            "entropy_std": round(np.std([x["entropy"] for x in vals]), 4),
            "stability_mean": round(np.mean([x["stability"] for x in vals]), 4),
            "stability_std": round(np.std([x["stability"] for x in vals]), 4),
        }

    return final

# ==================================================
# SINGLE IMAGE
# ==================================================
@app.post("/run")
async def run(image: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await image.read())).convert("L")
    return run_batch([CLEAN(img)])

# ==================================================
# DATASET
# ==================================================
@app.post("/run-dataset")
async def run_dataset(dataset_name: str = Form(...)):
    base = MNIST(root=DATA_DIR, train=False, download=True)

    if dataset_name == "MNIST_100":
        images = [CLEAN(base[i][0]) for i in range(100)]
        results = run_batch(images)

    elif dataset_name == "MNIST_500":
        images = [CLEAN(base[i][0]) for i in range(500)]
        results = run_batch(images)

    elif dataset_name == "MNIST_FULL":
        images = [CLEAN(base[i][0]) for i in range(len(base))]
        results = run_batch(images)

    elif dataset_name == "MNIST_NOISY_100":
        results = run_noisy_multi_eval(
            lambda: [NOISY()(base[i][0]) for i in range(100)]
        )

    elif dataset_name == "MNIST_NOISY_500":
        results = run_noisy_multi_eval(
            lambda: [NOISY()(base[i][0]) for i in range(500)]
        )

    elif dataset_name == "MNIST_NOISY_BLUR_100":
        results = run_noisy_multi_eval(
            lambda: [NOISY_BLUR()(base[i][0]) for i in range(100)]
        )

    elif dataset_name == "MNIST_NOISY_BLUR_500":
        results = run_noisy_multi_eval(
            lambda: [NOISY_BLUR()(base[i][0]) for i in range(500)]
        )

    else:
        raise HTTPException(400, "Unknown dataset")

    return {
        "dataset_type": dataset_name,
        "num_images": len(base),
        "models": results,
    }

# ==================================================
# OCR
# ==================================================
import pytesseract
@app.post("/verify")
async def verify(image: UploadFile = File(...), raw_text: str = Form(...)):
    img = Image.open(image.file).convert("L").resize((128, 32))

    ocr_text = pytesseract.image_to_string(
        img,
        config="--psm 10 --oem 1 -c tessedit_char_whitelist=0123456789",
    ).strip()

    errors = []

    max_len = max(len(raw_text), len(ocr_text))

    for i in range(max_len):
        typed_char = raw_text[i] if i < len(raw_text) else ""
        ocr_char = ocr_text[i] if i < len(ocr_text) else ""

        if typed_char != ocr_char:
            errors.append({
                "position": i + 1,
                "typed_char": typed_char,
                "ocr_char": ocr_char,
            })

    return {
        "verdict": "VALID_TYPED_TEXT" if not errors else "INVALID_OR_AMBIGUOUS",
        "final_output": ocr_text,
        "errors": errors,
        "why": (
            "OCR output perfectly matches typed text."
            if not errors
            else "One or more characters differ between typed text and OCR output."
        ),
    }
