# Fintech Full Stack Application  
(Next.js Frontend + FastAPI Inference Backend)

This repository contains a **full-stack Fintech demo system**:

- **Frontend**: Next.js (App Router) + TypeScript + Tailwind + Bun  
- **Backend**: FastAPI + PyTorch (MNIST inference & OCR verification)
---

# Prerequisites

## Common

* Git
* Internet connection (for dependencies)

## Frontend

* **Bun** (required)

## Backend

* **Python 3.10 – 3.12**
* `pip`
* (Optional) CUDA for GPU inference

---

#  Installing Bun (Frontend)

## macOS (Intel / Apple Silicon)

```bash
curl -fsSL https://bun.sh/install | bash
source ~/.zshrc
bun --version
```

---

## Linux

```bash
curl -fsSL https://bun.sh/install | bash
source ~/.bashrc
bun --version
```

---

## Windows

### PowerShell (Recommended)

```powershell
powershell -c "irm bun.sh/install.ps1 | iex"
```

Restart PowerShell and verify:

```powershell
bun --version
```

---

# Clone the Repository

```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>
```

---

# Backend Setup (FastAPI + PyTorch)

## Move into backend

```bash
cd fintech-backend
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

Activate it:

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```powershell
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Start Backend Server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Backend will run at:

```
http://127.0.0.1:8000
```

---

## Backend API Endpoints

| Endpoint       | Method | Description              |
| -------------- | ------ | ------------------------ |
| `/run`         | POST   | Single image inference   |
| `/run-dataset` | POST   | Dataset-based evaluation |
| `/verify`      | POST   | OCR digit verification   |

---

# Frontend Setup (Next.js + Bun)

## Move into frontend

```bash
cd ../fintech-frontend
```

---

## Install Dependencies

```bash
bun install
```

---

## Environment Variables (Frontend)

Create a `.env` file:

```bash
touch .env
```

### `.env` (DO NOT COMMIT THIS)

```env
# Database
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>/<db>?retryWrites=true&w=majority

# Better Auth
BETTER_AUTH_SECRET=your_secret_here
BETTER_AUTH_URL=http://localhost:3000

# Inference backend
INFERENCE_API_URL=http://127.0.0.1:8000

# GitHub OAuth
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret
```

**Never commit `.env` to GitHub**
Add this to `.gitignore`:

```gitignore
.env
.env.local
```

---

## Run Frontend (Development)

```bash
bun run dev
```

Frontend will be available at:

```
http://localhost:3000
```

---

## Build Frontend (Production)

```bash
bun run build
```

---

## Start Production Server

```bash
bun run start
```

---

# Full System Flow

```text
User (Browser)
   ↓
Next.js Frontend (Bun)
   ↓ API calls
FastAPI Backend (PyTorch)
   ↓
MNIST Models / OCR Verification
```

---

# Model Details (Backend)

Loaded MNIST models:

* `baseline_mnist.pth`
* `kd_mnist.pth`
* `lrf_mnist.pth`
* `pruned_mnist.pth`
* `quantized_mnist.pth`
* `ws_mnist.pth`

Metrics computed:

* Latency
* Confidence
* Entropy
* Stability

---

# Common Issues & Fixes

### Bun not found

```bash
source ~/.zshrc
```

---

### Port already in use

```bash
lsof -i :3000
kill -9 <PID>
```

---

### Backend not reachable

Ensure:

```env
INFERENCE_API_URL=http://127.0.0.1:8000
```

---

# 9️⃣ Security Notes (IMPORTANT)

❌ Do NOT commit:

* `.env`
* OAuth secrets
* MongoDB credentials

Rotate secrets if accidentally pushed.

