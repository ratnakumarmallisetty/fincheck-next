import time
import psutil
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import pytesseract
import re
from model_def import MNISTCNN

# -------------------- App setup --------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FILES = [
    "baseline_mnist.pth",
    "kd_mnist.pth",
    "lrf_mnist.pth",
    "pruned_mnist.pth",
    "quantized_mnist.pth",
    "ws_mnist.pth",
]

# -------------------- MNIST Lazy Loader --------------------
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
                strict=False
            )
            model.eval()
            MNIST_MODELS[f] = model
        print("✅ MNIST models loaded")
    return MNIST_MODELS

# -------------------- Transform --------------------
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# -------------------- Health --------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "mnist_loaded": MNIST_MODELS is not None
    }

# ==================================================
# MNIST INFERENCE / BENCHMARK ENDPOINT
# ==================================================
@app.post("/run")
async def run(image: UploadFile = File(...)):
    print("/run called...")
    models = load_mnist_models()

    img = Image.open(image.file).convert("L")
    img = transform(img).unsqueeze(0).to(DEVICE)

    process = psutil.Process()

    results = {}

    for name, model in models.items():
        # ---------------- MEMORY BEFORE ----------------
        mem_before = process.memory_info().rss / 1024 / 1024

        # ---------------- INFERENCE ----------------
        start = time.perf_counter()
        with torch.no_grad():
            logits = model(img)
            probs = torch.softmax(logits, dim=1)
        latency_ms = (time.perf_counter() - start) * 1000

        # ---------------- MEMORY AFTER ----------------
        mem_after = process.memory_info().rss / 1024 / 1024

        # ---------------- METRICS ----------------
        confidence = probs.max().item() * 100

        # Entropy (numerically stable)
        entropy = float(
            -(probs * torch.log(probs + 1e-8)).sum().item()
        )

        # Stability proxy (logit spread)
        stability = float(logits.std().item())

        # Throughput (samples/sec)
        throughput = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        # Cold start (0 = warm, frontend still renders)
        cold_start_ms = 0.0

        results[name] = {
            "confidence": round(confidence, 2),
            "latency_ms": round(latency_ms, 2),
            "throughput": round(throughput, 2),
            "entropy": round(entropy, 4),
            "stability": round(abs(stability), 4),
            "ram_mb": round(mem_after - mem_before, 2),
            "cold_start_ms": cold_start_ms,
        }

    return results

# ==================================================
# OCR + CHARACTER ERROR DETECTION
# ==================================================
@app.post("/verify")
async def verify(
    image: UploadFile = File(...),
    raw_text: str = Form(...)
):
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
    min_len = min(len(raw_text), len(ocr_text))

    for i in range(min_len):
        if raw_text[i] != ocr_text[i]:
            errors.append({
                "position": i + 1,
                "typed_char": raw_text[i],
                "ocr_char": ocr_text[i],
                "reason": "Ambiguous character normalized by OCR"
            })

    if len(raw_text) != len(ocr_text):
        errors.append({
            "reason": "Length mismatch between typed input and OCR output"
        })

    if errors:
        return {
            "verdict": "INVALID_OR_AMBIGUOUS",
            "method": "OCR_ERROR_DETECTION",
            "final_output": ocr_text,
            "errors": errors,
            "why": "One or more characters are ambiguous or invalid."
        }

    return {
        "verdict": "VALID_TYPED_TEXT",
        "method": "OCR",
        "final_output": ocr_text,
        "errors": [],
        "why": "Typed numeric input validated successfully."
    }
