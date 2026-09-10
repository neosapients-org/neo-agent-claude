"""
Guardrail check routes.

POST  /api/v1/guard/input    — validate user input
POST  /api/v1/guard/output   — validate LLM output
POST  /api/v1/guard/context  — validate conversation context
POST  /api/v1/guard/all      — run all three layers in one call
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException, Request

from ...core.models import AggregatedResult
from ..dependencies import OrchestratorManager
from ..models.requests import (
    GuardAllRequest,
    GuardContextRequest,
    GuardInputRequest,
    GuardOutputRequest,
)
from ..models.responses import (
    GuardAllResponse,
    GuardrailCheckDetail,
    SafetyCheckResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/guard", tags=["Guardrails"])


# ------------------------------------------------------------------ helpers


def _build_reason(result: AggregatedResult) -> str:
    """Build a human-readable reason from the highest-risk failed check."""
    failed = [r for r in result.results if not r.passed]
    if not failed:
        return "Safety check failed"

    highest = max(failed, key=lambda r: r.risk_score)
    name = highest.guardrail_name.replace("_", " ").title()
    message = highest.message or "detected"
    return f"{name}: {message}"


def _result_to_response(
    result: AggregatedResult,
    request_id: str,
    agent_id: str,
) -> SafetyCheckResponse:
    """Convert an AggregatedResult into a SafetyCheckResponse."""
    return SafetyCheckResponse(
        request_id=request_id,
        safe=result.passed,
        risk_score=result.max_risk_score,
        layer=result.layer.value,
        failed_checks=[r.guardrail_name for r in result.failed_checks],
        reason=_build_reason(result) if not result.passed else None,
        sanitized_text=result.final_text if result.passed else None,
        checks=[
            GuardrailCheckDetail(
                name=r.guardrail_name,
                passed=r.passed,
                risk_score=r.risk_score,
                message=r.message,
                latency_ms=r.latency_ms,
            )
            for r in result.results
        ],
        latency_ms=result.total_latency_ms,
        agent_id=agent_id,
    )


def _get_manager(request: Request) -> OrchestratorManager:
    """Retrieve the shared OrchestratorManager from app state."""
    manager: OrchestratorManager = request.app.state.orchestrator_manager
    if not manager.is_ready:
        raise HTTPException(status_code=503, detail="Service not ready — orchestrator not initialized")
    return manager


# ------------------------------------------------------------------ routes


@router.post("/input", response_model=SafetyCheckResponse)
async def guard_input(body: GuardInputRequest, request: Request):
    """Check user input for prompt injection, jailbreak attempts, and other threats.

    Accepts an optional inline YAML config (as JSON dict) so each calling
    agent can define its own guardrails without pre-registering.
    
    Also accepts conversation_history for context-aware guardrails (especially NeMo).
    """
    manager = _get_manager(request)
    request_id = OrchestratorManager.generate_request_id()
    agent_id = body.agent_id or "default"

    try:
        orch = await manager.get_orchestrator(
            inline_config=body.config,
            agent_id=agent_id,
        )
        context = OrchestratorManager.build_context(
            agent_id=agent_id,
            session_id=body.session_id,
            user_id=body.user_id,
            metadata=body.metadata,
            conversation_history=body.conversation_history,
        )

        result = await orch.guard_input(
            text=body.text,
            agent_id=agent_id,
            context=context,
        )

        resp = _result_to_response(result, request_id, agent_id)
        _log_result("input", agent_id, result)
        return resp

    except Exception as e:
        logger.error("Error in guard_input: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/output", response_model=SafetyCheckResponse)
async def guard_output(body: GuardOutputRequest, request: Request):
    """Check LLM output for safety issues (PII leakage, toxicity, etc.).
    
    Accepts conversation_history for context-aware output guardrails (especially NeMo).
    """
    manager = _get_manager(request)
    request_id = OrchestratorManager.generate_request_id()
    agent_id = body.agent_id or "default"

    try:
        orch = await manager.get_orchestrator(
            inline_config=body.config,
            agent_id=agent_id,
        )
        extra_meta = {"user_input": body.prompt} if body.prompt else None
        context = OrchestratorManager.build_context(
            agent_id=agent_id,
            session_id=body.session_id,
            user_id=body.user_id,
            metadata=body.metadata,
            conversation_history=body.conversation_history,
            extra_metadata=extra_meta,
        )

        result = await orch.guard_output(
            text=body.text,
            agent_id=agent_id,
            context=context,
        )

        resp = _result_to_response(result, request_id, agent_id)
        _log_result("output", agent_id, result)
        return resp

    except Exception as e:
        logger.error("Error in guard_output: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/context", response_model=SafetyCheckResponse)
async def guard_context(body: GuardContextRequest, request: Request):
    """Check conversation context for topical / policy violations.
    
    Conversation history is essential for context guardrails to work properly.
    """
    manager = _get_manager(request)
    request_id = OrchestratorManager.generate_request_id()
    agent_id = body.agent_id or "default"

    try:
        orch = await manager.get_orchestrator(
            inline_config=body.config,
            agent_id=agent_id,
        )
        context = OrchestratorManager.build_context(
            agent_id=agent_id,
            session_id=body.session_id,
            user_id=body.user_id,
            metadata=body.metadata,
            conversation_history=body.conversation_history,
        )

        result = await orch.guard_context(
            text=body.text,
            agent_id=agent_id,
            context=context,
        )

        resp = _result_to_response(result, request_id, agent_id)
        _log_result("context", agent_id, result)
        return resp

    except Exception as e:
        logger.error("Error in guard_context: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/all", response_model=GuardAllResponse)
async def guard_all(body: GuardAllRequest, request: Request):
    """Run all three guardrail layers (input → context → output) in one call.

    If the input layer fails, context and output are skipped.
    If context fails, output is skipped.
    """
    manager = _get_manager(request)
    request_id = OrchestratorManager.generate_request_id()
    agent_id = body.agent_id or "default"

    try:
        orch = await manager.get_orchestrator(
            inline_config=body.config,
            agent_id=agent_id,
        )
        context = OrchestratorManager.build_context(
            agent_id=agent_id,
            session_id=body.session_id,
            user_id=body.user_id,
            metadata=body.metadata,
            conversation_history=body.conversation_history,
        )

        results = await orch.guard_all(
            input_text=body.input_text,
            output_text=body.output_text,
            agent_id=agent_id,
            context=context,
        )

        input_resp = _result_to_response(results["input"], request_id, agent_id)
        context_resp = (
            _result_to_response(results["context"], request_id, agent_id)
            if "context" in results
            else None
        )
        output_resp = (
            _result_to_response(results["output"], request_id, agent_id)
            if "output" in results
            else None
        )

        overall_safe = all(
            r.passed
            for r in results.values()
        )

        total_latency = sum(r.total_latency_ms for r in results.values())

        return GuardAllResponse(
            request_id=request_id,
            safe=overall_safe,
            input_result=input_resp,
            context_result=context_resp,
            output_result=output_resp,
            latency_ms=total_latency,
            agent_id=agent_id,
        )

    except Exception as e:
        logger.error("Error in guard_all: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------ logging


def _log_result(layer: str, agent_id: str, result: AggregatedResult) -> None:
    status = "✓ SAFE" if result.passed else "✗ BLOCKED"
    logger.info(
        "%s [%s] | Agent: %s | Risk: %.2f | Failed: %s",
        status,
        layer,
        agent_id,
        result.max_risk_score,
        [r.guardrail_name for r in result.failed_checks],
    )
