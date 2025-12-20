FROM python:3.12-slim

# -------------------------
# Environment
# -------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

WORKDIR /app

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

# -------------------------
# Python dependencies
# -------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 🔥 ENSURE pytesseract IS PRESENT (CRITICAL)
RUN pip install --no-cache-dir pytesseract

# -------------------------
# Copy backend code
# -------------------------
COPY . .

# -------------------------
# ⬇️ DOWNLOAD MODELS AT BUILD TIME
# -------------------------
RUN python - <<EOF
from download_models import download
download()
EOF

# -------------------------
# Sanity checks
# -------------------------
RUN python - <<EOF
import pytesseract, os
print("✅ pytesseract version:", pytesseract.get_tesseract_version())
print("✅ Model files:", os.listdir("model"))
EOF

# -------------------------
# Expose port
# -------------------------
EXPOSE 8000

# -------------------------
# Run server
# -------------------------
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
