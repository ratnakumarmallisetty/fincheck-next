# Fincheck – Confidence-Aware Cheque Digit Validation System

**Risk-Aware Handwritten Digit Verification for Financial Documents**

**Next.js (Frontend) · FastAPI + PyTorch (Backend)**

---

## 1. Overview

**Fincheck** is a **full-stack fintech verification system** designed to detect **incorrect, ambiguous, or risky handwritten digits** in financial documents such as **bank cheques**.

Unlike conventional OCR systems that *always output a digit*, **Fincheck is explicitly confidence-aware**.

> **Core Principle**  
> *In financial systems, a wrong prediction is more dangerous than no prediction.*

Instead of maximizing accuracy at all costs, Fincheck prioritizes:

- Risk minimization
- Uncertainty exposure
- Safe abstention
- Human-in-the-loop verification

---

## 2. Problem Statement

Traditional OCR and digit-recognition systems:

- Optimize for **top-1 accuracy**
- Force predictions on ambiguous inputs
- Hide uncertainty from downstream systems

In financial workflows (cheques, account numbers, amounts):

- Silent misclassification → **monetary loss**
- Ambiguous digits (`3 ↔ 5`, `1 ↔ 7`, `0 ↔ 6`) → **high risk**
- No confidence signal → **no human review**

**Fincheck addresses this gap** by explicitly modeling **confidence, entropy, and instability**, and by rejecting unsafe predictions.

---

## 3. Key Capabilities

### 3.1 Image-Only Digit Validation (No OCR Dependency)

- Accepts **handwritten digit images**
- Works with **scans, photos, noisy inputs**
- Handles **single or multi-digit sequences**
- No typed text required

---

### 3.2 Confidence-Aware Decision States

Each digit is classified into one of three states:

| State | Meaning |
|-----|--------|
| **VALID** | High confidence (≥ 90%), safe for automation |
| **AMBIGUOUS** | Medium confidence (70–90%), human review required |
| **INVALID** | Low confidence (< 70%) or out-of-distribution |

This prevents unsafe automation in financial pipelines.

---

### 3.3 Position-Level Explainability

For every digit position, Fincheck reports:

- Digit index
- Predicted digit
- Confidence score (%)
- Top-3 plausible alternatives

This enables **auditability, dispute resolution, and human verification**.

---

## 4. Why MNIST Is Used (Critical Design Choice)

MNIST is **not** used to “recognize cheques”.

It is used as a **digit shape validity prior**:

- MNIST models learn **canonical handwritten digit manifolds**
- Inputs far from this manifold → low confidence, high entropy
- Enables **rejection instead of forced prediction**

> MNIST acts as a *risk filter*, not an OCR engine.

---

## 5. System Architecture

```text
User (Browser)
   ↓
Next.js Frontend (Bun)
   ↓
FastAPI Backend
   ↓
Image Cleaning & Binarization (OpenCV)
   ↓
Digit Segmentation (Connected Components)
   ↓
MNIST Normalization (28×28 + Center-of-Mass)
   ↓
Multi-Model MNIST Inference (PyTorch)
   ↓
Confidence / Entropy / Stability Analysis
   ↓
VALID / AMBIGUOUS / INVALID Verdict
````

---

## 6. Digit Segmentation Module (Critical Component)

Segmentation is a **first-class risk control stage** in Fincheck.

A perfect classifier is useless if digits are **incorrectly segmented**.

---

### Why Segmentation Matters in Cheques

Real cheque images contain:

* Touching or overlapping digits
* Broken ink strokes
* Background noise
* Irregular spacing

Segmentation errors can cause:

* Digit merging (`11 → 1`)
* Digit splitting (`8 → 0 + 0`)
* Missing or extra digits

In finance, these are **catastrophic failures**.

---

### Segmentation Pipeline

```text
Input Image
   ↓
Grayscale Conversion
   ↓
Stroke Enhancement (Morphological Close)
   ↓
Otsu Thresholding + Inversion
   ↓
Connected Components (8-connectivity)
   ↓
Geometric Filtering (Area, Width, Height)
   ↓
Left-to-Right Ordering
```

---

### Conservative Design Choices

* Small or thin components are **discarded**
* Borderline shapes are **rejected**, not guessed
* Ordering assumes **left-to-right digits only**

> Segmentation failure is treated as a **risk signal**, not an exception.

---

### Why Not Deep Learning Segmentation?

Fincheck intentionally avoids neural segmentation because:

* Silent hallucinations are common
* Large labeled cheque datasets are required
* Failure modes are opaque
* Hard to audit

Rule-based segmentation provides **determinism, transparency, and safety**.

---

## 7. Benchmarking Philosophy

### Why Accuracy Alone Is Insufficient

In fintech:

* 99% accuracy with 1% silent errors is unacceptable
* A wrong digit is worse than a rejected digit

Therefore, Fincheck benchmarks models using **risk-oriented metrics**, not accuracy alone.

---

## 8. Benchmarked Models

| Model                 | Description         |
| --------------------- | ------------------- |
| `baseline_mnist.pth`  | Standard CNN        |
| `kd_mnist.pth`        | Knowledge-distilled |
| `lrf_mnist.pth`       | Low-rank factorized |
| `pruned_mnist.pth`    | Weight-pruned       |
| `quantized_mnist.pth` | Quantized inference |
| `ws_mnist.pth`        | Weight-shared       |

---

## 9. Benchmark Metrics Explained

### Confidence (%)

* Mean max softmax probability
* Measures **certainty**

### Entropy

* Measures **uncertainty**
* High entropy → ambiguous digit

### Stability

* Logit standard deviation
* Measures **numerical consistency**

### Latency (ms)

* Inference time per image
* Important for real-time validation

---

## 10. Benchmark Datasets

| Dataset                  | Purpose                   |
| ------------------------ | ------------------------- |
| `MNIST_100 / 500 / FULL` | Clean baseline            |
| `MNIST_NOISY_*`          | Sensor noise robustness   |
| `MNIST_NOISY_BLUR_*`     | Scan + camera degradation |

Noise simulates **real cheque acquisition conditions**.

---

## 11. `/verify-digit-only` Endpoint (Detailed)

### Purpose

Fully automated **image-only cheque digit validation**.

### Pipeline

1. Image cleaning
2. Digit segmentation
3. MNIST normalization
4. Top-3 prediction
5. Confidence-based verdict

### Example Output

```json
{
  "verdict": "AMBIGUOUS",
  "digits": "709",
  "analysis": [
    {
      "position": 3,
      "predicted": "9",
      "confidence": 72.5,
      "status": "AMBIGUOUS",
      "possible_values": [9, 3, 5]
    }
  ]
}
```

---

## 12. `/verify` Endpoint (OCR + Typed Text)

### Purpose

Detect **human data-entry errors**.

### Use Case

* User types cheque number or amount
* OCR extracts digits
* System flags mismatches

### Output

* Mismatch positions
* Typed vs OCR value
* Final verdict

---

## 13. Why Two Verification Modes Exist

| Mode                 | Use Case                 |
| -------------------- | ------------------------ |
| `/verify-digit-only` | Automated pipelines      |
| `/verify`            | Human-assisted workflows |

This mirrors **real banking systems**, not demos.

---

## 14. Tech Stack

### Frontend

* Next.js (App Router)
* TypeScript
* Tailwind CSS
* Bun

### Backend

* FastAPI
* PyTorch
* OpenCV
* NumPy / SciPy
* PIL
* Torchvision
* Tesseract (only for `/verify`)

---

## 15. Installation & Running

### Clone

```bash
git clone <YOUR_REPO_URL>
cd fincheck
```

### Backend

```bash
cd fintech-backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install scipy
python download_models.py
uvicorn server:app --port 8000
```

### Frontend

```bash
cd fintech-frontend
bun install
bun run dev
```

---

## 16. Evaluation Philosophy (Summary)

Fincheck explicitly prioritizes:

* Rejection over risky prediction
* Explainability over opacity
* Confidence over accuracy
* Auditability over convenience

This makes it suitable for **real-world financial systems**, not just ML benchmarks.

---

## 17. License

For **academic, research, and demonstration purposes only**.
```
