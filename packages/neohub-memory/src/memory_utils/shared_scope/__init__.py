"""Shared-scope extensions for multi-agent and enterprise memory isolation.

Provides:
    SharedScopePoolRouter — PoolRoutingHook implementation for TEAM/ORG/SHARED pools
    SharedMemoryStrategy — Write-permission control for shared tiers
    PoolSearchConfig — Configuration for pool-tier searches
    ScopedMemoryConnector — Scope-aware wrapper over NeoMemoryConnector
"""

from memory_utils.shared_scope.pool_routing import (
    PoolSearchConfig,
    SharedScopePoolRouter,
)
from memory_utils.shared_scope.scoped_connector import ScopedMemoryConnector
from memory_utils.shared_scope.strategy import SharedMemoryStrategy

__all__ = [
    "PoolSearchConfig",
    "ScopedMemoryConnector",
    "SharedMemoryStrategy",
    "SharedScopePoolRouter",
]
