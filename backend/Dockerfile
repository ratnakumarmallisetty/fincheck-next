FROM python:3.12-slim

# --------------------------------------------------
# Environment
# --------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

WORKDIR /app

# --------------------------------------------------
# System dependencies (OCR + Torch runtime)
# --------------------------------------------------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# Python dependencies (cached layer)
# --------------------------------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir pytesseract

# --------------------------------------------------
# Copy application code
# --------------------------------------------------
COPY . .

# --------------------------------------------------
# Download models at build time
# --------------------------------------------------
RUN python - <<EOF
from download_models import download
download()
EOF

# --------------------------------------------------
# Sanity check (fail fast)
# --------------------------------------------------
RUN python - <<EOF
import pytesseract, os
print("✅ pytesseract:", pytesseract.get_tesseract_version())
models = os.listdir("model")
assert models, "❌ No models found"
print("✅ Models:", models)
EOF

# --------------------------------------------------
# Create non-root user (Render-safe)
# --------------------------------------------------
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser
# --------------------------------------------------
# Run server (Render REQUIRED format)
# --------------------------------------------------
CMD ["sh", "-c", \
     "uvicorn server:app \
      --host 0.0.0.0 \
      --port ${PORT} \
      --workers 2 \
      --timeout-keep-alive 30"]
