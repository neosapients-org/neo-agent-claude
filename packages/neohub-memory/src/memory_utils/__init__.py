"""memory_utils — Shared-scope extensions for neo_memory_hub.

This package provides multi-agent, multi-user, and team/org pool
routing capabilities. It extends memory_core with:

- SharedScopePoolRouter: PoolRoutingHook implementation for TEAM/ORG/SHARED pools
- ScopedMemoryConnector: Scope-aware CRUD operations (add_scoped, search_scoped, etc.)
- SharedMemoryStrategy: Write-permission control for shared tiers

Install this package only when the client needs multi-agent or
enterprise isolation features. memory_core works standalone for
single-user, single-agent use cases.
"""
