import time
import io
import zipfile
import psutil
import torch
import numpy as np
from pathlib import Path
from typing import Optional
import pytesseract

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.transforms import GaussianBlur

from model_def import MNISTCNN

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
# MODEL LOADER (LAZY)
# ==================================================

MNIST_MODELS = None

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
# TRANSFORMS (CORRECT ORDER)
# ==================================================

# Clean MNIST
CLEAN_TRANSFORM = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# Gaussian Noise (sensor noise)
def NOISY_TRANSFORM(std=0.2):
    return transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: torch.clamp(
                x + std * torch.randn_like(x),
                0.0,
                1.0,
            )
        ),
    ])

# Gaussian Blur (defocus / scanning)
def BLUR_TRANSFORM(kernel_size=5, sigma=1.0):
    return transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((28, 28)),
        GaussianBlur(kernel_size=kernel_size, sigma=sigma),
        transforms.ToTensor(),
    ])

# Noise + Blur (realistic degradation)
def NOISY_BLUR_TRANSFORM(
    noise_std=0.2,
    blur_kernel=5,
    blur_sigma=1.0,
):
    return transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((28, 28)),
        GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma),
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: torch.clamp(
                x + noise_std * torch.randn_like(x),
                0.0,
                1.0,
            )
        ),
    ])

# ==================================================
# SINGLE IMAGE INFERENCE
# ==================================================

def run_single_image(img_tensor, models, process):
    results = {}

    for name, model in models.items():
        mem_before = process.memory_info().rss / 1024 / 1024

        start = time.perf_counter()
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)
        latency_ms = (time.perf_counter() - start) * 1000

        mem_after = process.memory_info().rss / 1024 / 1024

        confidence = probs.max().item() * 100
        entropy = float(-(probs * torch.log(probs + 1e-8)).sum().item())
        stability = float(logits.std().item())
        ram_mb = mem_after - mem_before

        results[name] = {
            "latency_ms": round(latency_ms, 3),
            "confidence_percent": round(confidence, 2),
            "entropy": round(entropy, 4),
            "stability": round(abs(stability), 4),
            "ram_mb": round(ram_mb, 3),
        }

    return results

# ==================================================
# AGGREGATE DATASET METRICS
# ==================================================

def aggregate_metrics(per_image_results):
    aggregated = {}

    for model, entries in per_image_results.items():
        aggregated[model] = {
            "avg_confidence": round(np.mean([e["confidence_percent"] for e in entries]), 2),
            "avg_latency_ms": round(np.mean([e["latency_ms"] for e in entries]), 3),
            "avg_entropy": round(np.mean([e["entropy"] for e in entries]), 4),
            "avg_stability": round(np.mean([e["stability"] for e in entries]), 4),
            "avg_ram_mb": round(np.mean([e["ram_mb"] for e in entries]), 3),
            "num_images": len(entries),
        }

    return aggregated

# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "mnist_loaded": MNIST_MODELS is not None,
    }

# ==================================================
# SINGLE IMAGE ENDPOINT
# ==================================================

@app.post("/run")
async def run(image: UploadFile = File(...)):
    models = load_mnist_models()
    process = psutil.Process()

    img = Image.open(image.file).convert("L")
    img_tensor = CLEAN_TRANSFORM(img).unsqueeze(0).to(DEVICE)

    return run_single_image(img_tensor, models, process)

# ==================================================
# DATASET / ROBUSTNESS ENDPOINT
# ==================================================

@app.post("/run-dataset")
async def run_dataset(
    zip_file: Optional[UploadFile] = File(None),
    dataset_name: Optional[str] = Form(None),
):
    models = load_mnist_models()
    process = psutil.Process()

    per_image_results = {name: [] for name in models.keys()}
    images = []

    # ---------------- LOAD DATASET ----------------

    if zip_file is not None:
        with zipfile.ZipFile(io.BytesIO(await zip_file.read())) as z:
            for name in z.namelist():
                if name.lower().endswith((".png", ".jpg", ".jpeg")):
                    with z.open(name) as f:
                        images.append(
                            CLEAN_TRANSFORM(Image.open(f).convert("L"))
                        )
        dataset_type = "CUSTOM_ZIP"

    elif dataset_name:

        base_dataset = MNIST(
            root=DATA_DIR,
            train=False,
            download=True,
            transform=None,
        )

        if dataset_name == "MNIST_100":
            images = [CLEAN_TRANSFORM(base_dataset[i][0]) for i in range(100)]

        elif dataset_name == "MNIST_500":
            images = [CLEAN_TRANSFORM(base_dataset[i][0]) for i in range(500)]

        elif dataset_name == "MNIST_TEST":
            images = [CLEAN_TRANSFORM(base_dataset[i][0]) for i in range(len(base_dataset))]

        elif dataset_name == "MNIST_NOISY_100":
            images = [NOISY_TRANSFORM()(base_dataset[i][0]) for i in range(100)]

        elif dataset_name == "MNIST_BLUR_100":
            images = [BLUR_TRANSFORM()(base_dataset[i][0]) for i in range(100)]

        elif dataset_name == "MNIST_NOISY_BLUR_100":
            images = [NOISY_BLUR_TRANSFORM()(base_dataset[i][0]) for i in range(100)]

        else:
            return {"error": f"Unknown dataset_name: {dataset_name}"}

        dataset_type = dataset_name

    else:
        return {"error": "Either zip_file or dataset_name must be provided"}

    # ---------------- RUN INFERENCE ----------------

    for img in images:
        img_tensor = img.unsqueeze(0).to(DEVICE)
        single_result = run_single_image(img_tensor, models, process)

        for model, metrics in single_result.items():
            per_image_results[model].append(metrics)

    aggregated = aggregate_metrics(per_image_results)

    return {
        "dataset_type": dataset_type,
        "num_images": len(images),
        "models": aggregated,
    }


# ==================================================
# OCR + CHARACTER ERROR DETECTION
# ==================================================
@app.post("/verify")
async def verify(
    image: UploadFile = File(...),
    raw_text: str = Form(...)
):
    try:
        pil_img = Image.open(image.file).convert("L")

        # OCR (digits only)
        ocr_text = pytesseract.image_to_string(
            pil_img,
            config="--psm 7 -c tessedit_char_whitelist=0123456789"
        ).strip().replace(" ", "")

        if not ocr_text:
            return {
                "verdict": "INVALID_OR_AMBIGUOUS",
                "method": "OCR",
                "final_output": None,
                "errors": [],
                "why": "OCR could not detect numeric characters."
            }

        errors = []

        len_raw = len(raw_text)
        len_ocr = len(ocr_text)
        max_len = max(len_raw, len_ocr)

        # --- CHARACTER-LEVEL COMPARISON ---
        for i in range(max_len):
            typed_char = raw_text[i] if i < len_raw else None
            ocr_char = ocr_text[i] if i < len_ocr else None

            if typed_char != ocr_char:
                if typed_char is None:
                    errors.append({
                        "position": i + 1,
                        "typed_char": None,
                        "ocr_char": ocr_char,
                        "error_type": "EXTRA_CHARACTER_IN_IMAGE",
                        "reason": "OCR detected an extra digit not present in typed input"
                    })
                elif ocr_char is None:
                    errors.append({
                        "position": i + 1,
                        "typed_char": typed_char,
                        "ocr_char": None,
                        "error_type": "MISSING_CHARACTER_IN_IMAGE",
                        "reason": "Typed digit has no corresponding OCR character"
                    })
                else:
                    errors.append({
                        "position": i + 1,
                        "typed_char": typed_char,
                        "ocr_char": ocr_char,
                        "error_type": "CHARACTER_MISMATCH",
                        "reason": "Ambiguous or misrecognized handwritten digit"
                    })

        # --- FINAL VERDICT ---
        if errors:
            return {
                "verdict": "INVALID_OR_AMBIGUOUS",
                "method": "OCR_ERROR_DETECTION",
                "final_output": ocr_text,
                "errors": errors,
                "why": "Multiple discrepancies detected between typed input and handwritten image."
            }

        return {
            "verdict": "VALID_TYPED_TEXT",
            "method": "OCR",
            "final_output": ocr_text,
            "errors": [],
            "why": "Typed numeric input validated successfully."
        }

    except Exception as e:
        print("❌ /verify failed:", str(e))
        return {
            "verdict": "ERROR",
            "method": "OCR",
            "final_output": None,
            "errors": [],
            "why": "OCR service failed on server."
        }
