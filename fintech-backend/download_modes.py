# backend/download_models.py
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

print("📁 Model dir:", MODEL_DIR)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RELEASE_TAG = "v1-models"
BASE_URL = f"https://github.com/mukesh1352/fincheck-next/releases/download/{RELEASE_TAG}"

MODELS = [
    "baseline_mnist.pth",
    "kd_mnist.pth",
    "lrf_mnist.pth",
    "pruned_mnist.pth",
    "quantized_mnist.pth",
    "ws_mnist.pth",
]

def download():
    for name in MODELS:
        dest = MODEL_DIR / name
        if dest.exists():
            print(f"✅ {name} already exists")
            continue

        url = f"{BASE_URL}/{name}"
        print(f"⬇️ Downloading {url}")

        r = requests.get(url, stream=True)
        r.raise_for_status()

        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

    print("🎉 All models downloaded successfully")

if __name__ == "__main__":
    download()