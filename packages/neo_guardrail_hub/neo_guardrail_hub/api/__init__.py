"""
Neo Guardrail Hub API Module.

This module provides:
1. FastAPI microservice (``create_app``, ``app``)
2. Pre-generation of configuration files
3. Initialization and setup utilities
4. Caching management

Microservice usage:
    # Start the API server
    uvicorn neo_guardrail_hub.api.app:app --host 0.0.0.0 --port 8000

    # Or programmatically
    from neo_guardrail_hub.api.app import create_app
    app = create_app(config_path="./configs")

Pre-generation usage:
    from neo_guardrail_hub.api import (
        initialize,
        generate_configs,
        get_rails_cache,
    )

    await initialize(config_path="./configs")
    await initialize(config_path="./configs", agent_id="wealth_advisor")
"""

# FastAPI microservice
from .app import create_app, app  # noqa: F401
from .dependencies import OrchestratorManager  # noqa: F401

# Pre-generation
from .pregeneration import (
    initialize,
    initialize_sync,
    generate_configs,
    generate_configs_sync,
    PreGenerationResult,
)
# Caching
from .caching import (
    RailsCache,
    get_rails_cache,
    clear_rails_cache,
    reset_rails_cache,
)

__all__ = [
    # Microservice
    "create_app",
    "app",
    "OrchestratorManager",
    # Pre-generation
    "initialize",
    "initialize_sync",
    "generate_configs",
    "generate_configs_sync",
    "PreGenerationResult",
    # Caching
    "RailsCache",
    "get_rails_cache",
    "clear_rails_cache",
    "reset_rails_cache",
]
