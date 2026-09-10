# ============================================================
#  NeoMemory MCP Server — Production Docker Image
#
#  Multi-stage build:
#    Stage 1 (builder) — compiles C-extension wheels
#    Stage 2 (runtime) — slim image with only runtime deps
#
#  Runtime config (all via env vars):
#    PORT                  MCP listen port       (default 18432)
#    NEOMEM_QDRANT_URL     Qdrant gRPC/HTTP URL  (default http://localhost:6333)
#    NEOMEM_QDRANT_COLLECTION  Collection name    (default neo_memory)
#    OPENAI_API_KEY        Required for LLM ops
#    NEOMEM_LLM_MODEL      GPT model             (default gpt-4o-mini)
#    NEOMEM_LOG_LEVEL      Python log level       (default INFO)
#
#  Build:
#    docker build -t neomem-mcp:latest .
#
#  Run (standalone, Qdrant on host or another container):
#    docker run --rm -p 18432:18432 \
#      -e OPENAI_API_KEY="$OPENAI_API_KEY" \
#      -e NEOMEM_QDRANT_URL=http://qdrant:6333 \
#      neomem-mcp:latest
# ============================================================

# ── Stage 1: Builder ────────────────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Cache-friendly: install deps first, then copy source
COPY pyproject.toml README.md LICENSE ./
RUN python -m pip install --upgrade pip setuptools wheel

COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/

# Build a wheel and install it + mcp extra into /install prefix
RUN python -m pip install --no-cache-dir --prefix=/install ".[mcp]"


# ── Stage 2: Runtime ────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Minimal runtime deps only (no compiler toolchain)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash neomem

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy only what's needed at runtime
COPY scripts/start_mcp_server.py scripts/start_mcp_server.py
COPY config/ config/

# ── Environment ─────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PORT=18432 \
    NEOMEM_QDRANT_URL=http://localhost:6333 \
    NEOMEM_QDRANT_COLLECTION=neo_memory \
    NEOMEM_MCP_HOST=0.0.0.0 \
    NEOMEM_LOG_LEVEL=INFO

EXPOSE ${PORT}

# No HEALTHCHECK — MCP uses streamable-http protocol, not plain
# HTTP GET. Configure health probes in your orchestrator (ECS, K8s)
# if needed, e.g. via a TCP check on the port.

USER neomem

# Start the MCP server via the production launcher script
CMD ["sh", "-c", "exec python -u scripts/start_mcp_server.py --host 0.0.0.0 --port $PORT"]
