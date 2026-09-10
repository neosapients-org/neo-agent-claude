"""Tenant context sourced live from the Cortex platform (capabilities map + client roster).

The registry here is the single source of truth for "what can Cortex answer" and "who are
the clients" — both were previously hardcoded in the agent (and in the frontend). See
``registry.py`` for the front-load / TTL-refresh lifecycle.
"""

from .registry import cortex_context  # noqa: F401
