FROM python:3.14-slim

WORKDIR /app

# Install system dependencies for the PuLP CBC solver backend
RUN apt-get update && apt-get install -y --no-install-recommends \
    coinor-cbc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render dynamically assigns a PORT (defaults to 10000); this fallback ensures local tests still work on 8000
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"