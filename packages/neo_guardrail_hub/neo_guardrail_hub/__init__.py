"""
Neo Guardrail Hub - A Universal, Pluggable Guardrail Framework for Any AI Agent

This framework provides a unified interface for adding security guardrails
to AI agents, LLM applications, and agentic workflows.

Features:
    - Multi-layer protection (input, context, output)
    - Pluggable guardrail providers (LLM Guard, NeMo, custom)
    - Async-first with sync wrappers
    - Easy integration with decorators and context managers
    - Flexible YAML-based configuration

Quick Start:
    from neo_guardrail_hub import NeoGuardrailOrchestrator

    orchestrator = NeoGuardrailOrchestrator(config_path="./configs")
    result = await orchestrator.guard_input("user message")

    if result.passed:
        # Safe to process
        response = await llm.generate(...)
    else:
        # Handle blocked input
        print(result.message)

Using the Decorator:
    from neo_guardrail_hub import guarded

    @guarded(agent_id="my_agent")
    async def my_agent(user_input: str) -> str:
        return await llm.generate(user_input)

Example:
    >>> from neo_guardrail_hub import NeoGuardrailOrchestrator
    >>> orchestrator = NeoGuardrailOrchestrator()
    >>> result = await orchestrator.guard_input("Hello, world!")
    >>> print(result.passed)
    True
"""

from .__version__ import __version__

# Core components
from .core.models import (
    ActionOnFail,
    AggregatedResult,
    ExecutionMode,
    GuardrailConfig,
    GuardrailContext,
    GuardrailLayer,
    GuardrailResult,
    LayerConfig,
)
from .core.interfaces import BaseGuardrail, BaseProvider
from .core.exceptions import (
    ConfigurationError,
    ExecutionTimeoutError,
    GuardrailError,
    GuardrailNotFoundError,
    ProviderError,
    ValidationError,
)
from .core.orchestrator import NeoGuardrailOrchestrator
from .core.config import ConfigLoader
from .core.registry import GuardrailRegistry

# Guardrails
from .guardrails.base import GuardrailBase
from .guardrails.input import (
    PIIDetectionGuardrail,
    PromptInjectionGuardrail,
)
from .guardrails.output import (
    PIIRedactionGuardrail,
    NeMoSelfCheckOutputGuardrail,
    NeMoSelfCheckFactsGuardrail,
    NeMoSelfCheckHallucinationGuardrail,
)

# Providers
from .providers import LLMGuardProvider, ProviderBase

# Integrations
from .integrations.decorators import guarded, guarded_sync
from .integrations.context_manager import GuardSession, GuardSessionSync

# Utilities
from .utils import configure_logging, get_logger

# API (Phase 4) - Pre-generation and Caching
from .api import (
    initialize,
    initialize_sync,
    generate_configs,
    generate_configs_sync,
    PreGenerationResult,
    RailsCache,
    get_rails_cache,
    clear_rails_cache,
    reset_rails_cache,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "NeoGuardrailOrchestrator",
    "ConfigLoader",
    "GuardrailRegistry",
    # Models
    "ActionOnFail",
    "AggregatedResult",
    "ExecutionMode",
    "GuardrailConfig",
    "GuardrailContext",
    "GuardrailLayer",
    "GuardrailResult",
    "LayerConfig",
    # Interfaces
    "BaseGuardrail",
    "BaseProvider",
    # Exceptions
    "ConfigurationError",
    "ExecutionTimeoutError",
    "GuardrailError",
    "GuardrailNotFoundError",
    "ProviderError",
    "ValidationError",
    # Guardrails
    "GuardrailBase",
    "PIIDetectionGuardrail",
    "PIIRedactionGuardrail",
    "PromptInjectionGuardrail",
    "NeMoSelfCheckOutputGuardrail",
    "NeMoSelfCheckFactsGuardrail",
    "NeMoSelfCheckHallucinationGuardrail",
    # Providers
    "LLMGuardProvider",
    "ProviderBase",
    # Integrations
    "guarded",
    "guarded_sync",
    "GuardSession",
    "GuardSessionSync",
    # Utilities
    "configure_logging",
    "get_logger",
    # API (Phase 4)
    "initialize",
    "initialize_sync",
    "generate_configs",
    "generate_configs_sync",
    "PreGenerationResult",
    "RailsCache",
    "get_rails_cache",
    "clear_rails_cache",
    "reset_rails_cache",
]
