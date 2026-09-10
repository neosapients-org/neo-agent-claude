"""
Neo Guardrail Hub — API Service entry point.

This is a thin wrapper that imports the FastAPI app from the package.
All route logic lives in neo_guardrail_hub/api/.

Usage:
    # Development (auto-reload)
    uvicorn api_service:app --host 0.0.0.0 --port 8000 --reload

    # Or use the package directly
    uvicorn neo_guardrail_hub.api.app:app --host 0.0.0.0 --port 8000

    # Programmatic
    from neo_guardrail_hub.api.app import create_app
    app = create_app(config_path="./configs")

Docker:
    docker build -t neo-guardrail-api .
    docker run -p 8000:8000 -e OPENAI_API_KEY=your-key neo-guardrail-api

Endpoints:
    POST /api/v1/guard/input     — Check input for threats
    POST /api/v1/guard/output    — Check LLM output for safety
    POST /api/v1/guard/context   — Check conversation context
    POST /api/v1/guard/all       — Run all 3 layers in one call
    GET  /api/v1/guardrails      — List available guardrail types
    GET  /api/v1/guardrails/{t}  — Details for a guardrail type
    POST /api/v1/config/reload   — Hot-reload YAML configs
    GET  /health                 — Liveness probe
    GET  /ready                  — Readiness probe
"""

import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Import the app from the package
from neo_guardrail_hub.api.app import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    import uvicorn

    if not os.getenv("OPENAI_API_KEY"):
        logging.warning("⚠️  OPENAI_API_KEY not set (only needed for NeMo guardrails)")

    uvicorn.run(
        "api_service:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        log_level="info",
    )
