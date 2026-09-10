"""External framework providers for Neo Guardrail Hub."""

from .base import ProviderBase
from .llm_guard import LLMGuardProvider

# Conditionally import NeMoProvider if nemoguardrails is available
try:
    from .nemo import NeMoProvider
    _nemo_available = True
except ImportError:
    NeMoProvider = None  # type: ignore
    _nemo_available = False

__all__ = ["ProviderBase", "LLMGuardProvider"]

if _nemo_available:
    __all__.append("NeMoProvider")
