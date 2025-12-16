import time
import psutil
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

from backend.model_def import MNISTCNN
from backend import download_models

download_models.download()


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

MODELS = {}

transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

for f in MODEL_FILES:
    model = MNISTCNN().to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_DIR / f, map_location=DEVICE),
        strict=False
    )
    model.eval()
    MODELS[f] = model

@app.post("/run")
async def run(image: UploadFile = File(...)):
    img = Image.open(image.file).convert("L")
    img = transform(img).unsqueeze(0).to(DEVICE)

    process = psutil.Process()
    results = {}

    for name, model in MODELS.items():
        mem_before = process.memory_info().rss / 1024 / 1024
        start = time.perf_counter()

        with torch.no_grad():
            out = model(img)

        latency = (time.perf_counter() - start) * 1000
        mem_after = process.memory_info().rss / 1024 / 1024

        results[name] = {
            "confidence": float(out.softmax(1).max()) * 100,
            "latency_ms": round(latency, 2),
            "ram_mb": round(mem_after - mem_before, 2),
        }

    return results
