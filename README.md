# Fincheck – Confidence-Aware Cheque Digit Validation System

**Next.js (Frontend) + FastAPI + PyTorch (Backend)**

---

## Overview

**Fincheck** is a **full-stack fintech verification system** designed to detect **incorrect, ambiguous, or risky handwritten digits** in financial documents such as **bank cheques**.

Unlike traditional OCR systems that always output a digit, **Fincheck focuses on risk detection** — identifying **when a digit should not be trusted**.

### Core idea

> *In financial systems, a wrong prediction is more dangerous than no prediction.*

---

## Key Features

### Image-only cheque digit verification

* Accepts **single or multiple handwritten digits**
* Works with **clean, noisy, scanned, or photographed images**
* Supports **PNG / JPG / JPEG** formats

### Confidence-aware digit validation

Each digit is classified as:

* **VALID** – high confidence
* **AMBIGUOUS** – multiple plausible digits
* **INVALID** – low confidence / unreliable

###  Position-level error reporting

* Identifies **exact digit position**
* Shows **predicted value**
* Displays **confidence percentage**
* Lists **possible alternative digits**

###  MNIST-based verification (no OCR guessing)

* Uses **pretrained MNIST CNN models (.pth)**
* Converts cheque digits → **MNIST-style 28×28 normalized format**
* Rejects digits outside the learned digit manifold

---

##  Why MNIST is Used (Important)

MNIST is **not used to recognize cheques**.

It is used as a **digit shape verifier** to determine:

* Whether a digit matches known handwritten digit distributions
* Whether a digit is ambiguous or unsafe to trust

This avoids silent failures common in OCR systems.

---

##  System Architecture

```text
User (Browser)
   ↓
Next.js Frontend (Bun)
   ↓ API Requests
FastAPI Backend
   ↓
Image Cleaning (OpenCV)
   ↓
Digit Segmentation (Connected Components)
   ↓
MNIST Normalization (28×28 + Center of Mass)
   ↓
KD-MNIST Inference (Confidence-Aware)
   ↓
VALID / AMBIGUOUS / INVALID
```

---

##  Tech Stack

### Frontend

* Next.js (App Router)
* TypeScript
* Tailwind CSS
* Bun
* Canvas-based image rendering
* Fetch API

### Backend

* FastAPI
* PyTorch
* OpenCV
* NumPy / SciPy
* PIL
* Torchvision
* Tesseract (only for `/verify`, not image-only digit validation)

---


##  Prerequisites

### Common

* Git
* Internet connection

### Frontend

* **Bun (required)**

### Backend

* **Python 3.10 – 3.12**
* pip
* (Optional) CUDA for GPU inference

---

##  Installing Bun (Frontend)

### macOS / Linux

```bash
curl -fsSL https://bun.sh/install | bash
source ~/.zshrc   # or ~/.bashrc
bun --version
```

### Windows (PowerShell)

```powershell
powershell -c "irm bun.sh/install.ps1 | iex"
```

Restart terminal and verify:

```powershell
bun --version
```

---

##  Clone Repository

```bash
git clone <YOUR_REPO_URL>
cd fincheck-next
```

---

##  Backend Setup (FastAPI + PyTorch)

### Move to backend

```bash
cd fintech-backend
```

### Create virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start backend server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Backend runs at:

```
http://127.0.0.1:8000
```

---

##  Backend API Endpoints

| Endpoint             | Method | Description                            |
| -------------------- | ------ | -------------------------------------- |
| `/verify-digit-only` | POST   | **Image-only cheque digit validation** |
| `/verify`            | POST   | OCR + typed text verification          |
| `/run`               | POST   | Single image MNIST inference           |
| `/run-dataset`       | POST   | Dataset-level MNIST evaluation         |

---

##  Frontend Setup (Next.js + Bun)

### Move to frontend

```bash
cd ../fintech-frontend
```

### Install dependencies

```bash
bun install
```

---

##  Environment Variables (Frontend)

Create `.env` file:

```bash
touch .env
```

### `.env` (DO NOT COMMIT)

```env
# Database
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>/<db>

# Authentication
BETTER_AUTH_SECRET=your_secret_here
BETTER_AUTH_URL=http://localhost:3000

# Backend
INFERENCE_API_URL=http://127.0.0.1:8000

# OAuth (optional)
GITHUB_CLIENT_ID=your_client_id
GITHUB_CLIENT_SECRET=your_client_secret
```

Add to `.gitignore`:

```gitignore
.env
.env.local
```

---

##  Run Frontend (Development)

```bash
bun run dev
```

Frontend available at:

```
http://localhost:3000
```

---

##  Example Output (Cheque Digit Validation)

```text
Verdict: AMBIGUOUS
Detected Digits: 709

Position 1
Status: VALID
Predicted: 7
Confidence: 97%

Position 2
Status: VALID
Predicted: 0
Confidence: 97.65%

Position 3
Status: AMBIGUOUS
Predicted: 9
Confidence: 72.5%
Possible values: 9, 3, 5
```

---

## Evaluation Philosophy

This system does **not maximize accuracy**.

Instead, it minimizes **financial risk** by:

* Rejecting low-confidence digits
* Highlighting ambiguous digits
* Avoiding silent misclassification


---

## 📜 License

This project is intended for **academic, research, and demonstration purposes**.
