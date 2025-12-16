import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

BASE_URL = "https://github.com/mukesh1352/fincheck-next/releases/download/models"

FILES = [
    "baseline_mnist.pth",
    "kd_mnist.pth",
    "lrf_mnist.pth",
    "pruned_mnist.pth",
    "quantized_mnist.pth",
    "ws_mnist.pth",
]

def download():
    for f in FILES:
        dest = MODEL_DIR / f
        if dest.exists():
            print(f"✅ {f} already exists")
            continue

        url = f"{BASE_URL}/{f}"
        print(f"⬇️ Downloading {f}")
        r = requests.get(url, stream=True)
        r.raise_for_status()

        with open(dest, "wb") as out:
            for chunk in r.iter_content(chunk_size=8192):
                out.write(chunk)

if __name__ == "__main__":
    download()
