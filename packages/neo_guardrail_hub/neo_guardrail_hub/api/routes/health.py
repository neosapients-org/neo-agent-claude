"""
Health and readiness probe routes.

GET  /health  — lightweight liveness check
GET  /ready   — deeper readiness check (are models loaded?)
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Request

from ..models.responses import HealthResponse, ReadyResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Lightweight liveness probe — always responds if the process is up."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/ready", response_model=ReadyResponse)
async def readiness_check(request: Request):
    """Deeper readiness probe.

    Reports whether the default orchestrator is initialised and
    which guardrails / providers are available.
    """
    manager = request.app.state.orchestrator_manager

    available_guardrails = 0
    available_providers: list[str] = []

    if manager.default_orchestrator:
        available_guardrails = len(manager.default_orchestrator.registry.list_available())
        available_providers = manager.default_orchestrator.registry.list_providers()

    return ReadyResponse(
        ready=manager.is_ready,
        orchestrator_initialized=manager.is_ready,
        available_guardrails=available_guardrails,
        available_providers=available_providers,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
