"""API request models for neo_guardrail_hub."""
from pydantic import BaseModel
from typing import Optional, Any


class GuardRequest(BaseModel):
    text: str
    agent_id: Optional[str] = None
    context: Optional[dict[str, Any]] = None


class GuardResponse(BaseModel):
    passed: bool
    results: list[dict[str, Any]] = []
    final_text: Optional[str] = None


class GuardInputRequest(BaseModel):
    text: str
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class GuardOutputRequest(BaseModel):
    text: str
    prompt: Optional[str] = None
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class GuardContextRequest(BaseModel):
    text: str
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class GuardAllRequest(BaseModel):
    input_text: str
    output_text: Optional[str] = None
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class ConfigReloadRequest(BaseModel):
    config_path: Optional[str] = None
    agent_id: Optional[str] = None
