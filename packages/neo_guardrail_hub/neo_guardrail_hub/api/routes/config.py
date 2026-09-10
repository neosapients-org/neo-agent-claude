"""
Configuration & guardrail discovery routes.

GET   /api/v1/guardrails           — list all available guardrail types
GET   /api/v1/guardrails/{type}    — details for a specific guardrail type
POST  /api/v1/config/reload        — hot-reload on-disk YAML configs
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request

from ...core.models import GuardrailLayer
from ..dependencies import OrchestratorManager
from ..models.requests import ConfigReloadRequest
from ..models.responses import (
    ConfigReloadResponse,
    GuardrailDetailResponse,
    GuardrailInfoResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Configuration"])


@router.get("/guardrails", response_model=GuardrailInfoResponse)
async def list_guardrails(request: Request):
    """List all available guardrail types from the registry.

    Returns guardrail names, layers, descriptions, and provider info.
    """
    manager: OrchestratorManager = request.app.state.orchestrator_manager
    orch = manager.default_orchestrator

    if orch is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    registry = orch.registry
    guardrails_list: List[Dict[str, Any]] = []

    # Collect directly registered guardrails
    for name in sorted(registry._guardrails.keys()):
        cls = registry._guardrails[name]
        guardrails_list.append(
            {
                "name": name,
                "layer": getattr(cls, "layer", GuardrailLayer.INPUT).value,
                "description": getattr(cls, "description", ""),
                "provider": _detect_provider(name, cls),
            }
        )

    # Collect provider-supplied guardrails not already listed
    seen = {g["name"] for g in guardrails_list}
    for provider_name in registry.list_providers():
        provider = registry.get_provider(provider_name)
        if provider:
            for gtype in provider.list_supported_guardrails():
                if gtype not in seen:
                    guardrails_list.append(
                        {
                            "name": gtype,
                            "layer": "input",  # default; provider may not expose layer
                            "description": "",
                            "provider": provider_name,
                        }
                    )
                    seen.add(gtype)

    return GuardrailInfoResponse(
        guardrails=guardrails_list,
        total=len(guardrails_list),
        providers=registry.list_providers(),
    )


@router.get("/guardrails/{guardrail_type}", response_model=GuardrailDetailResponse)
async def get_guardrail_detail(guardrail_type: str, request: Request):
    """Get details for a specific guardrail type.

    Returns the guardrail's layer, description, provider, and
    default configuration options.
    """
    manager: OrchestratorManager = request.app.state.orchestrator_manager
    orch = manager.default_orchestrator

    if orch is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    registry = orch.registry

    # Check direct registrations
    if guardrail_type in registry._guardrails:
        cls = registry._guardrails[guardrail_type]
        # Try to instantiate with empty config to get defaults
        default_config: Dict[str, Any] = {}
        try:
            instance = cls({})
            default_config = instance.get_config()
        except Exception:
            pass

        return GuardrailDetailResponse(
            name=guardrail_type,
            layer=getattr(cls, "layer", GuardrailLayer.INPUT).value,
            description=getattr(cls, "description", "No description available"),
            provider=_detect_provider(guardrail_type, cls),
            default_config=default_config,
        )

    # Check providers
    for provider_name in registry.list_providers():
        provider = registry.get_provider(provider_name)
        if provider and provider.supports(guardrail_type):
            return GuardrailDetailResponse(
                name=guardrail_type,
                layer="input",
                description=f"Provided by {provider_name}",
                provider=provider_name,
                default_config={},
            )

    raise HTTPException(
        status_code=404,
        detail=f"Guardrail type '{guardrail_type}' not found. "
        f"Available: {registry.list_available()}",
    )


@router.post("/config/reload", response_model=ConfigReloadResponse)
async def reload_config(body: ConfigReloadRequest, request: Request):
    """Hot-reload on-disk YAML configs without restarting the service.

    Optionally pass an ``agent_id`` to reload only that agent's config.
    If omitted, all cached configs are cleared and reloaded on next request.
    """
    manager: OrchestratorManager = request.app.state.orchestrator_manager

    try:
        manager.reload_config(agent_id=body.agent_id)
        return ConfigReloadResponse(
            success=True,
            message=f"Config reloaded for agent_id={body.agent_id or 'all'}",
            agent_id=body.agent_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.error("Config reload failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")


# --------------------------------------------------------------------- helpers


def _detect_provider(name: str, cls: type) -> str:
    """Heuristic to detect provider from guardrail class module path."""
    module = getattr(cls, "__module__", "")
    if "nemo" in module or "nemo" in name:
        return "nemo"
    return "llm_guard"
