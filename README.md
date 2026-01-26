# Fincheck – Confidence-Aware Cheque Digit Validation System

**Risk-Aware Handwritten Digit Verification for Financial Documents**

**Next.js (Frontend) · FastAPI + PyTorch (Backend)**

---

## 1. Overview

**Fincheck** is a **full-stack fintech verification system** designed to detect **incorrect, ambiguous, or risky handwritten digits** in financial documents such as **bank cheques**.

Unlike conventional OCR systems that *always output a digit*, **Fincheck is explicitly confidence-aware**.

> **Core Principle**  
> *In financial systems, a wrong prediction is more dangerous than no prediction.*

Fincheck therefore prioritizes **abstention, uncertainty detection, and auditability** over raw accuracy.

---

## 2. Problem Statement

Traditional OCR and digit-recognition pipelines:

- Optimize for **top-1 accuracy**
- Force a prediction even for ambiguous or degraded inputs
- Hide uncertainty from downstream systems

In financial workflows (cheques, account numbers, transaction amounts):

- Silent misclassification → **monetary loss**
- Ambiguous digits (`3 ↔ 5`, `1 ↔ 7`, `0 ↔ 6`) → **high risk**
- Lack of confidence signals → **no human intervention**

**Fincheck addresses this gap** by modeling **confidence, entropy, and instability**, and by explicitly refusing unsafe predictions.

---

## 3. Key Capabilities

### 3.1 Image-Only Digit Validation (No OCR Dependency)

- Accepts **handwritten digit images**
- Works with **scans, camera photos, noisy inputs**
- Handles **single or multi-digit sequences**
- No typed text required

---

### 3.2 Confidence-Aware Decision States

Each digit is classified into one of three states:

| State | Meaning |
|-----|-------|
| **VALID** | High confidence (≥ 90%), safe for automation |
| **AMBIGUOUS** | Medium confidence (70–90%), human review required |
| **INVALID** | Low confidence (< 70%) or out-of-distribution |

This prevents unsafe downstream automation.

---

### 3.3 Position-Level Explainability

For every digit position, Fincheck reports:

- Digit index
- Predicted digit
- Confidence score (%)
- Top-3 plausible alternatives

This enables **manual verification, audit trails, and dispute resolution**.

---

## 4. Why MNIST Is Used (Critical Design Choice)

MNIST is **not** used to “recognize cheques”.

It is used as a **digit shape validity prior**:

- MNIST models learn **canonical digit manifolds**
- Inputs far from this manifold → high entropy, low confidence
- This allows **rejection instead of forced prediction**

> Fincheck uses MNIST as a *risk filter*, not an OCR engine.

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

## 6. Benchmarking Philosophy

### Why Accuracy Alone Is Insufficient

In fintech systems:

* **99% accuracy with 1% silent errors is unacceptable**
* A wrong digit is worse than a rejected digit
* Confidence calibration matters more than raw accuracy

Therefore, Fincheck benchmarks models using:

* **Confidence**
* **Entropy**
* **Stability**
* **Latency**
* **Robustness under noise**

---

## 7. Benchmarked Models

Fincheck evaluates multiple MNIST CNN variants:

| Model                 | Description               |
| --------------------- | ------------------------- |
| `baseline_mnist.pth`  | Standard CNN              |
| `kd_mnist.pth`        | Knowledge-distilled model |
| `lrf_mnist.pth`       | Low-rank factorized model |
| `pruned_mnist.pth`    | Weight-pruned model       |
| `quantized_mnist.pth` | Quantized inference       |
| `ws_mnist.pth`        | Weight-shared model       |

This allows **trade-off analysis** between speed, confidence, and robustness.

---

## 8. Benchmark Metrics Explained

### 8.1 Confidence (%)

* Mean max softmax probability
* Indicates **how sure the model is**
* Low confidence → unsafe prediction

---

### 8.2 Entropy

* Measures **prediction uncertainty**
* High entropy → ambiguous digit
* Used to trigger **AMBIGUOUS / INVALID**

---

### 8.3 Stability

* Standard deviation of logits
* Measures **numerical and prediction consistency**
* High instability → unreliable inference

---

### 8.4 Latency (ms)

* Average inference time per image
* Important for **real-time validation**

---

## 9. Benchmark Datasets

| Dataset                  | Purpose                   |
| ------------------------ | ------------------------- |
| `MNIST_100 / 500 / FULL` | Clean baseline            |
| `MNIST_NOISY_*`          | Sensor noise robustness   |
| `MNIST_NOISY_BLUR_*`     | Scan + camera degradation |

Noise and blur simulate **real cheque acquisition conditions**.

---

## 10. Example Benchmark Output

```json
{
  "kd_mnist.pth": {
    "latency_mean": 1.12,
    "confidence_mean": 97.34,
    "entropy_mean": 0.0812,
    "stability_mean": 1.4321
  }
}
```

Interpretation:

* High confidence
* Low entropy
* Stable logits
  → **Safe for financial usage**

---

## 11. `/verify-digit-only` Endpoint (Detailed)

### Purpose

Image-only cheque digit validation with **no OCR fallback**.

### Pipeline

1. Image cleaning and binarization
2. Digit segmentation
3. MNIST normalization
4. Top-3 digit prediction
5. Confidence-based decision

### Output Example

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

This endpoint is designed for **automated cheque pipelines**.

---

## 12. `/verify` Endpoint (OCR + Typed Text Validation)

### Purpose

Cross-check **user-entered digits vs OCR output**.

### Use Case

* User manually enters cheque number or amount
* OCR extracts digits from image
* System flags mismatches

### Output

* Exact mismatch positions
* Typed vs OCR value
* Final verdict

This prevents **human data entry errors**.

---

## 13. Why Two Verification Modes Exist

| Mode                 | When to Use                     |
| -------------------- | ------------------------------- |
| `/verify-digit-only` | Full automation, no typed input |
| `/verify`            | Human-assisted workflows        |

This mirrors **real banking systems**, not demos.

---

## 14. Evaluation Philosophy

Fincheck explicitly chooses:

* **Rejection over risky prediction**
* **Explainability over opacity**
* **Confidence over accuracy**
* **Auditability over convenience**

This makes it suitable for **real-world fintech workflows**, not just ML benchmarks.

---

## 15. License

For **academic, research, and demonstration purposes only**.
