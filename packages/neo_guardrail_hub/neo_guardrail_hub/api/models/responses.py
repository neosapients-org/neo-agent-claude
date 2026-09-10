"""API response models for neo_guardrail_hub."""
from pydantic import BaseModel
from typing import Optional, Any


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.0"


class ReadyResponse(BaseModel):
    ready: bool = True
    checks: dict[str, Any] = {}


class ConfigReloadResponse(BaseModel):
    success: bool = True
    message: str = ""


class GuardrailDetailResponse(BaseModel):
    name: str
    enabled: bool
    provider: str = ""
    config: dict[str, Any] = {}


class GuardrailInfoResponse(BaseModel):
    guardrails: list[GuardrailDetailResponse] = []


class GuardrailCheckDetail(BaseModel):
    name: str
    passed: bool
    risk_score: float = 0.0
    message: str = ""
    latency_ms: float = 0.0


class SafetyCheckResponse(BaseModel):
    passed: bool
    layer: str = ""
    results: list[GuardrailCheckDetail] = []
    final_text: Optional[str] = None
    total_latency_ms: float = 0.0


class GuardAllResponse(BaseModel):
    input_result: Optional[SafetyCheckResponse] = None
    context_result: Optional[SafetyCheckResponse] = None
    output_result: Optional[SafetyCheckResponse] = None
    overall_passed: bool = True
