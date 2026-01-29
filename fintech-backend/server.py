import cv2
import numpy as np
import torch
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torchvision.transforms as transforms
from scipy import ndimage
import io
import random
import time
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.transforms import GaussianBlur
from sklearn.metrics import confusion_matrix

from model_def import MNISTCNN

# =========================
# TORCH CONFIG
# =========================
torch.set_grad_enabled(False)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# =========================
# APP
# =========================
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

# =========================
# LOAD KD MNIST MODEL
# =========================
MODEL = None

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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================
# TRANSFORM
# =========================
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),  # already 28x28
])
#==================
# ENHANCE KEYSTROKE
#==================
def enhance_strokes(gray):
    kernel = np.ones((2, 2), np.uint8)

    # close gaps in strokes
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # thicken strokes slightly
    gray = cv2.dilate(gray, kernel, iterations=0)

    return gray

# =========================
# CLEAN IMAGE (CHEQUE SAFE)
# =========================
def clean_image(img: Image.Image):
    img = np.array(img.convert("L"))

    img = enhance_strokes(img)

    _, img = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return img


# =========================
# MNIST NORMALIZATION (CRITICAL)
# =========================
def normalize_mnist_digit(digit_img):
    """
    Convert segmented digit into MNIST-style 28x28 image
    digit_img: binary image (white digit on black background)
    """

    # 1️⃣ Crop tight bounding box
    coords = cv2.findNonZero(digit_img)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    digit_img = digit_img[y:y+h, x:x+w]

    # 2️⃣ Aspect-ratio safe resize (max side = 20)
    h, w = digit_img.shape
    scale = 20.0 / max(h, w)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    digit_img = cv2.resize(
        digit_img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # 3️⃣ Place in 28x28 canvas (centered)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_img

    # 4️⃣ Center-of-mass alignment (CRITICAL for MNIST)
    cy, cx = ndimage.center_of_mass(canvas)

    if np.isnan(cx) or np.isnan(cy):
        return None

    shift_x = int(round(14 - cx))
    shift_y = int(round(14 - cy))

    canvas = ndimage.shift(
        canvas,
        shift=(shift_y, shift_x),
        mode="constant",
        cval=0
    )

    return Image.fromarray(canvas.astype(np.uint8))


# =========================
# SEGMENT DIGITS (OPENCV)
# =========================
def segment_digits(img):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        img, connectivity=8
    )

    digits = []
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]

        if area < 80 or w < 8 or h < 15:
            continue

        digit = img[y:y+h, x:x+w]
        digits.append((x, digit))

    digits.sort(key=lambda d: d[0])
    return [d[1] for d in digits]

# =========================
# MNIST INFERENCE
# =========================
@torch.inference_mode()
def classify_digit(img):
    model = MNIST_MODELS["kd_mnist.pth"]  # ✅ explicit

    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(tensor), dim=1)[0]

    top = torch.topk(probs, 3)
    return [
        {
            "digit": int(d),
            "confidence": round(float(c * 100), 2)
        }
        for d, c in zip(top.indices.cpu(), top.values.cpu())
    ]


# =========================
# API
# =========================
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

def compute_far_frr(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    total = cm.sum()

    FARs = []
    FRRs = []

    for c in range(10):
        TP = cm[c, c]
        FP = cm[:, c].sum() - TP
        FN = cm[c, :].sum() - TP
        TN = total - TP - FP - FN

        FAR_c = FP / (FP + TN + 1e-8)
        FRR_c = FN / (FN + TP + 1e-8)

        FARs.append(FAR_c)
        FRRs.append(FRR_c)

    FAR = float(np.mean(FARs))
    FRR = float(np.mean(FRRs))

    return cm.tolist(), round(FAR, 4), round(FRR, 4)

def risk_score(FAR, FRR, alpha=0.5, beta=0.5):
    return round(alpha * FAR + beta * FRR, 4)

# ======================================================
# INFERENCE CORE
# ======================================================
@torch.inference_mode()
def run_batch(images, true_labels=None):
    batch = torch.stack(images).to(DEVICE)
    out = {}

    for name, model in MNIST_MODELS.items():
        start = time.perf_counter()
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).cpu().numpy()

        entry = {
            "latency_ms": round((time.perf_counter() - start) * 1000 / len(batch), 3),
            "confidence_percent": round(probs.max(dim=1).values.mean().item() * 100, 2),
            "entropy": round(float(-(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()), 4),
            "stability": round(float(logits.std()), 4),
            "ram_mb": 0.0,
        }

        if true_labels is not None:
            cm, FAR, FRR = compute_far_frr(true_labels, preds)
            entry["evaluation"] = {
                "confusion_matrix": cm,
                "FAR": FAR,
                "FRR": FRR,
                "risk_score": risk_score(FAR, FRR)
            }

        out[name] = entry

    return out


# ==================================================
# INFERENCE (MULTI RUN – NOISY)
# ==================================================
def run_noisy_multi_eval(build_fn, true_labels, runs=5):
    acc = {k: [] for k in MNIST_MODELS}
    all_preds = {k: [] for k in MNIST_MODELS}

    for r in range(runs):
        set_seed(42 + r)
        images = build_fn()
        res = run_batch(images)

        for m, v in res.items():
            acc[m].append(v)

        batch = torch.stack(images).to(DEVICE)
        for name, model in MNIST_MODELS.items():
            logits = model(batch)
            preds = torch.softmax(logits, dim=1).argmax(dim=1)
            all_preds[name].extend(preds.cpu().numpy())

    final = {}

    for m, vals in acc.items():
        entry = {
            "latency_mean": round(np.mean([x["latency_ms"] for x in vals]), 3),
            "latency_std": round(np.std([x["latency_ms"] for x in vals]), 3),
            "confidence_mean": round(np.mean([x["confidence_percent"] for x in vals]), 2),
            "confidence_std": round(np.std([x["confidence_percent"] for x in vals]), 2),
            "entropy_mean": round(np.mean([x["entropy"] for x in vals]), 4),
            "entropy_std": round(np.std([x["entropy"] for x in vals]), 4),
            "stability_mean": round(np.mean([x["stability"] for x in vals]), 4),
            "stability_std": round(np.std([x["stability"] for x in vals]), 4),
        }

        repeated_labels = true_labels * runs

        cm, FAR, FRR = compute_far_frr(
            repeated_labels,
            all_preds[m]
        )

        entry["evaluation"] = {
            "confusion_matrix": cm,
            "FAR": FAR,
            "FRR": FRR,
            "risk_score": risk_score(FAR, FRR)
        }

        final[m] = entry

    return final

# ==================================================
# SINGLE IMAGE
# ==================================================
@app.post("/run")
async def run(
    image: UploadFile = File(...),
    expected_digit: int = Form(...)
):
    img = Image.open(io.BytesIO(await image.read())).convert("L")

    return run_batch(
        [CLEAN(img)],
        true_labels=[expected_digit]
    )

# ==================================================
# DATASET
# ==================================================
@app.post("/run-dataset")
async def run_dataset(dataset_name: str = Form(...)):
    base = MNIST(root=DATA_DIR, train=False, download=True)

    if dataset_name == "MNIST_100":
        images = [CLEAN(base[i][0]) for i in range(100)]
        labels = [base[i][1] for i in range(100)]
        results = run_batch(images, labels)

    elif dataset_name == "MNIST_500":
        images = [CLEAN(base[i][0]) for i in range(500)]
        labels = [base[i][1] for i in range(500)]
        results = run_batch(images, labels)

    elif dataset_name == "MNIST_FULL":
        images = [CLEAN(base[i][0]) for i in range(len(base))]
        labels = [base[i][1] for i in range(len(base))]
        results = run_batch(images, labels)

    elif dataset_name == "MNIST_NOISY_100":
        labels = [base[i][1] for i in range(100)]
        results = run_noisy_multi_eval(
            lambda: [NOISY()(base[i][0]) for i in range(100)],
            true_labels=labels
        )

    elif dataset_name == "MNIST_NOISY_500":
        labels = [base[i][1] for i in range(500)]
        results = run_noisy_multi_eval(
            lambda: [NOISY()(base[i][0]) for i in range(500)],
            true_labels=labels
        )

    elif dataset_name == "MNIST_NOISY_BLUR_100":
        labels = [base[i][1] for i in range(100)]
        results = run_noisy_multi_eval(
            lambda: [NOISY_BLUR()(base[i][0]) for i in range(100)],
            true_labels=labels
        )

    elif dataset_name == "MNIST_NOISY_BLUR_500":
        labels = [base[i][1] for i in range(500)]
        results = run_noisy_multi_eval(
            lambda: [NOISY_BLUR()(base[i][0]) for i in range(500)],
            true_labels=labels
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
import base64

def encode_img(img):
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode()
@app.post("/verify-digit-only")
async def verify_digit_only(image: UploadFile = File(...)):
    try:
        raw_pil = Image.open(image.file).convert("L")
        raw_np = np.array(raw_pil)

        cleaned = clean_image(raw_pil)
        digit_imgs = segment_digits(cleaned)

        if not digit_imgs:
            return {
                "verdict": "INVALID",
                "digits": "",
                "analysis": [],
                "preview": None,
                "reason": "No digits detected"
            }

        analysis = []
        final_digits = []
        final_verdict = "VALID"

        preview_cropped = None
        preview_normalized = None

        for i, dimg in enumerate(digit_imgs):

            # Save first cropped preview
            if preview_cropped is None:
                preview_cropped = encode_img(dimg)

            mnist_img = normalize_mnist_digit(dimg)

            if mnist_img is None:
                analysis.append({
                    "position": i + 1,
                    "status": "INVALID",
                    "predicted": None,
                    "confidence": 0,
                    "possible_values": [],
                })
                final_digits.append("?")
                final_verdict = "INVALID"
                continue

            # Save first normalized preview
            if preview_normalized is None:
                preview_normalized = encode_img(np.array(mnist_img))

            preds = classify_digit(mnist_img)
            best = preds[0]

            conf = best["confidence"]

            if conf >= 90:
                status = "VALID"
            elif conf >= 70:
                status = "AMBIGUOUS"
                if final_verdict != "INVALID":
                    final_verdict = "AMBIGUOUS"
            else:
                status = "INVALID"
                final_verdict = "INVALID"

            analysis.append({
                "position": i + 1,
                "status": status,
                "predicted": str(best["digit"]),
                "confidence": conf,
                "possible_values": [str(p["digit"]) for p in preds]
            })

            final_digits.append(str(best["digit"]))

        return {
            "verdict": final_verdict,
            "digits": "".join(final_digits),
            "analysis": analysis,
            "preview": {
                "original": encode_img(raw_np),
                "cropped": preview_cropped,
                "normalized": preview_normalized
            }
        }

    except Exception as e:
        return {
            "verdict": "ERROR",
            "message": str(e)
        }
