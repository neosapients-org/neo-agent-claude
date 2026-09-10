"""
Neo Guardrail Hub — FastAPI application factory.

Usage:
    # Programmatic
    from neo_guardrail_hub.api.app import create_app
    app = create_app()

    # CLI
    uvicorn neo_guardrail_hub.api.app:app --host 0.0.0.0 --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .dependencies import OrchestratorManager
from .routes.config import router as config_router
from .routes.guardrails import router as guardrails_router
from .routes.health import router as health_router

# Load .env from project root
_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=_env_path)

logger = logging.getLogger(__name__)


def create_app(
    config_path: str | None = None,
    models_dir: str | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config_path: Path to the YAML configs directory.
                     Defaults to ``./configs`` relative to CWD or the
                     ``NEO_CONFIG_PATH`` env var.
        models_dir:  Path to pre-downloaded LLM Guard models.
                     Defaults to ``NEO_LLM_GUARD_MODELS_DIR`` env var.
    """

    resolved_config_path = config_path or os.getenv(
        "NEO_CONFIG_PATH",
        str(Path.cwd() / "configs"),
    )
    resolved_models_dir = models_dir or os.getenv("NEO_LLM_GUARD_MODELS_DIR")

    # ---------------------------------------------------------------- lifespan

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """Startup / shutdown lifecycle."""
        logger.info("🚀 Starting Neo Guardrail Hub API …")
        manager = OrchestratorManager(
            config_path=resolved_config_path,
            models_dir=resolved_models_dir,
        )
        application.state.orchestrator_manager = manager

        try:
            await manager.initialize()
            logger.info("✓ Orchestrator manager initialized")
        except Exception as exc:
            logger.warning("⚠️  Orchestrator pre-init failed: %s", exc)

        yield

        logger.info("Shutting down Neo Guardrail Hub API …")
        await manager.shutdown()

    # ---------------------------------------------------------------- app

    application = FastAPI(
        title="Neo Guardrail Hub API",
        description=(
            "Microservice for AI safety guardrails.  "
            "Send your agent's YAML config inline or rely on on-disk defaults."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS — allow all for dev; tighten in production
    application.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------------------------------------------------- routers

    application.include_router(health_router)          # /health, /ready
    application.include_router(guardrails_router)      # /api/v1/guard/*
    application.include_router(config_router)          # /api/v1/guardrails, /api/v1/config/reload

    # ---------------------------------------------------------------- root

    @application.get("/", tags=["Root"])
    async def root():
        return {
            "service": "Neo Guardrail Hub API",
            "version": "1.0.0",
            "docs": "/docs",
            "endpoints": {
                "guard_input": "POST /api/v1/guard/input",
                "guard_output": "POST /api/v1/guard/output",
                "guard_context": "POST /api/v1/guard/context",
                "guard_all": "POST /api/v1/guard/all",
                "list_guardrails": "GET /api/v1/guardrails",
                "guardrail_detail": "GET /api/v1/guardrails/{type}",
                "config_reload": "POST /api/v1/config/reload",
                "health": "GET /health",
                "ready": "GET /ready",
            },
        }

    # ---------------------------------------------------------------- error handler

    @application.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc),
                "path": str(request.url),
            },
        )

    return application


# Default app instance for ``uvicorn neo_guardrail_hub.api.app:app``
app = create_app()
