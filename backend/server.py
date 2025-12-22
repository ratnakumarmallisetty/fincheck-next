import time
import io
import zipfile
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List
import pytesseract

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.transforms import GaussianBlur

from model_def import MNISTCNN

# ==================================================
# GLOBAL TORCH OPTIMIZATION (CRITICAL)
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
# MODEL LOADER (LAZY, CACHED)
# ==================================================

MNIST_MODELS = None

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_noisy_multi_eval(
    build_images_fn,
    models,
    runs: int = 5,
):
    collected = {
        name: {
            "latency": [],
            "confidence": [],
            "entropy": [],
            "stability": [],
        }
        for name in models
    }

    for run in range(runs):
        set_seed(42 + run)
        images = build_images_fn()

        results = run_batch_chunked(images, models)

        for model, m in results.items():
            collected[model]["latency"].append(m["latency_ms"])
            collected[model]["confidence"].append(m["confidence_percent"])
            collected[model]["entropy"].append(m["entropy"])
            collected[model]["stability"].append(m["stability"])

    final = {}
    for model, v in collected.items():
        final[model] = {
            "latency_mean": round(float(np.mean(v["latency"])), 3),
            "latency_std": round(float(np.std(v["latency"])), 3),

            "confidence_mean": round(float(np.mean(v["confidence"])), 2),
            "confidence_std": round(float(np.std(v["confidence"])), 2),

            "entropy_mean": round(float(np.mean(v["entropy"])), 4),
            "entropy_std": round(float(np.std(v["entropy"])), 4),

            "stability_mean": round(float(np.mean(v["stability"])), 4),
            "stability_std": round(float(np.std(v["stability"])), 4),
        }

    return final


def load_mnist_models():
    global MNIST_MODELS
    if MNIST_MODELS is None:
        print("🔵 Loading MNIST models...")
        MNIST_MODELS = {}
        for f in MODEL_FILES:
            model = MNISTCNN().to(DEVICE)
            model.load_state_dict(
                torch.load(MODEL_DIR / f, map_location=DEVICE),
                strict=False,
            )
            model.eval()
            MNIST_MODELS[f] = model
        print("✅ MNIST models loaded")
    return MNIST_MODELS

# ==================================================
# TRANSFORMS
# ==================================================

CLEAN_TRANSFORM = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

def NOISY_TRANSFORM(std=0.2):
    return transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.clamp(x + std * torch.randn_like(x), 0.0, 1.0)),
    ])

def BLUR_TRANSFORM(kernel_size=5, sigma=1.0):
    return transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((28, 28)),
        GaussianBlur(kernel_size=kernel_size, sigma=sigma),
        transforms.ToTensor(),
    ])

def NOISY_BLUR_TRANSFORM(noise_std=0.2, blur_kernel=5, blur_sigma=1.0):
    return transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((28, 28)),
        GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.clamp(x + noise_std * torch.randn_like(x), 0.0, 1.0)),
    ])

# ==================================================
# FAST BATCHED INFERENCE (CORE SPEEDUP)
# ==================================================

@torch.inference_mode()
def run_batch_chunked(
    images: List[torch.Tensor],
    models,
    chunk_size: int = 128,
):
    results = {}

    for name in models.keys():
        results[name] = {
            "latencies": [],
            "confidences": [],
            "entropies": [],
            "stabilities": [],
        }

    for i in range(0, len(images), chunk_size):
        chunk = images[i:i + chunk_size]
        batch = torch.stack(chunk).to(DEVICE)

        for name, model in models.items():
            start = time.perf_counter()
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            latency_ms = (time.perf_counter() - start) * 1000 / len(chunk)

            confidence = probs.max(dim=1).values.mean().item() * 100
            entropy = float(-(probs * torch.log(probs + 1e-8)).sum(dim=1).mean().item())
            stability = float(logits.std().item())

            results[name]["latencies"].append(latency_ms)
            results[name]["confidences"].append(confidence)
            results[name]["entropies"].append(entropy)
            results[name]["stabilities"].append(abs(stability))

    # Aggregate
    final = {}
    for name, r in results.items():
        final[name] = {
            "latency_ms": round(sum(r["latencies"]) / len(r["latencies"]), 3),
            "confidence_percent": round(sum(r["confidences"]) / len(r["confidences"]), 2),
            "entropy": round(sum(r["entropies"]) / len(r["entropies"]), 4),
            "stability": round(sum(r["stabilities"]) / len(r["stabilities"]), 4),
            "ram_mb": 0.0,
        }

    return final


# ==================================================
# HEALTH
# ==================================================

@app.get("/health")
def health():
    return {"status": "ok", "mnist_loaded": MNIST_MODELS is not None}

# ==================================================
# SINGLE IMAGE
# ==================================================

@app.post("/run")
async def run(image: UploadFile = File(...)):
    models = load_mnist_models()
    img = Image.open(image.file).convert("L")
    tensor = CLEAN_TRANSFORM(img)

    # Reuse chunked inference for consistency
    results = run_batch_chunked([tensor], models)

    return results


# ==================================================
# DATASET (FAST, BATCHED)
# ==================================================
@app.post("/run-dataset")
async def run_dataset(
    zip_file: Optional[UploadFile] = File(None),
    dataset_name: Optional[str] = Form(None),
):
    models = load_mnist_models()
    images: List[torch.Tensor] = []

    # ---------- CUSTOM ZIP ----------
    if zip_file:
        with zipfile.ZipFile(io.BytesIO(await zip_file.read())) as z:
            for name in z.namelist():
                if name.lower().endswith((".png", ".jpg", ".jpeg")):
                    with z.open(name) as f:
                        with Image.open(f) as img:
                            images.append(
                                CLEAN_TRANSFORM(img.convert("L"))
                            )
        dataset_type = "CUSTOM_ZIP"

    # ---------- PREBUILT DATASETS ----------
    elif dataset_name:
        base = MNIST(root=DATA_DIR, train=False, download=True)

        if dataset_name == "MNIST_100":
            images = [CLEAN_TRANSFORM(base[i][0]) for i in range(100)]

        elif dataset_name == "MNIST_500":
            images = [CLEAN_TRANSFORM(base[i][0]) for i in range(500)]

        elif dataset_name == "MNIST_FULL":
            images = [CLEAN_TRANSFORM(base[i][0]) for i in range(len(base))]

        elif dataset_name == "MNIST_NOISY_100":
            images = [NOISY_TRANSFORM()(base[i][0]) for i in range(100)]

        elif dataset_name == "MNIST_BLUR_100":
            images = [BLUR_TRANSFORM()(base[i][0]) for i in range(100)]

        elif dataset_name == "MNIST_NOISY_BLUR_100":
            images = [NOISY_BLUR_TRANSFORM()(base[i][0]) for i in range(100)]
        elif dataset_name == "MNIST_NOISY_500":
            images = [NOISY_TRANSFORM()(base[i][0]) for i in range(500)]

        elif dataset_name == "MNIST_NOISY_BLUR_500":
            images = [NOISY_BLUR_TRANSFORM()(base[i][0]) for i in range(500)]


        else:
            return {
                "error": f"Unknown dataset_name: {dataset_name}"
            }

        dataset_type = dataset_name

    # ---------- INVALID REQUEST ----------
    else:
        return {
            "error": "Either zip_file or dataset_name must be provided"
        }

    # ---------- SAFETY GUARD ----------
    MAX_IMAGES = 10000
    if len(images) > MAX_IMAGES:
        return {
            "error": f"Dataset too large ({len(images)} images). Max allowed is {MAX_IMAGES}."
        }

    # ---------- CHUNKED INFERENCE ----------
    batch_results = run_batch_chunked(images, models)

    return {
        "dataset_type": dataset_type,
        "num_images": len(images),
        "models": batch_results,
    }


# ==================================================
# OCR (FAST)
# ==================================================

@app.post("/verify")
async def verify(image: UploadFile = File(...), raw_text: str = Form(...)):
    try:
        img = Image.open(image.file).convert("L")
        img = img.resize((128, 32))

        ocr_text = pytesseract.image_to_string(
            img,
            config="--psm 10 --oem 1 -c tessedit_char_whitelist=0123456789",
        ).strip()

        if not ocr_text:
            return {"verdict": "INVALID_OR_AMBIGUOUS", "final_output": None, "errors": []}

        errors = []
        for i in range(max(len(raw_text), len(ocr_text))):
            if (raw_text[i:i+1] or None) != (ocr_text[i:i+1] or None):
                errors.append({"position": i + 1})

        if errors:
            return {
                "verdict": "INVALID_OR_AMBIGUOUS",
                "final_output": ocr_text,
                "errors": errors,
            }

        return {
            "verdict": "VALID_TYPED_TEXT",
            "final_output": ocr_text,
            "errors": [],
        }

    except Exception:
        return {"verdict": "ERROR", "final_output": None, "errors": []}
