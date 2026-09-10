"""
Framework integrations for Neo Memory Hub.

The public API is `NeoMemoryConnector`.

Usage:
    >>> from neo_memory_hub.integrations import NeoMemoryConnector, MemoryConfig
    >>>
    >>> async with NeoMemoryConnector(memory_config=MemoryConfig()) as connector:
    ...     await connector.add("User prefers dark mode", user_id="alex")
    ...     results = await connector.search("preferences", user_id="alex")
"""

from neo_memory_hub.integrations.connector import (
    ConnectorConfig,
    MemoryItem,
    NeoMemoryConnector,
    OperationLog,
)
from neo_memory_hub.integrations.memory import MemoryConfig


__all__ = [
    "ConnectorConfig",
    "MemoryConfig",
    "MemoryItem",
    # Universal connector
    "NeoMemoryConnector",
    "OperationLog",
]
