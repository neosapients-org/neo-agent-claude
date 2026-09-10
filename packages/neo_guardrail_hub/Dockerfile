# Dockerfile for Neo Guardrail Hub API Microservice
#
# Build:
#   docker build -t neo-guardrail-api .
#
# Run:
#   docker run -p 8000:8000 -e OPENAI_API_KEY=your-key neo-guardrail-api
#
# Run with custom config:
#   docker run -p 8000:8000 -v $(pwd)/configs:/app/configs -e OPENAI_API_KEY=your-key neo-guardrail-api

# ---------- build stage ----------
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY neo_guardrail_hub ./neo_guardrail_hub

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[all]" && \
    pip install --no-cache-dir uvicorn[standard] fastapi python-dotenv

# ---------- runtime stage ----------
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Copy configs and entry point
COPY configs ./configs
COPY api_service.py .

# Create non-root user
RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app
USER apiuser

# Environment
ENV NEO_CONFIG_PATH=/app/configs
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run via the package-level app
CMD ["uvicorn", "neo_guardrail_hub.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
