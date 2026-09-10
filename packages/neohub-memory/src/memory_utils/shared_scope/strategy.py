"""Shared memory strategy — controls write permissions for shared tiers."""
from __future__ import annotations

from enum import Enum


class SharedMemoryStrategy(str, Enum):
    """Controls write access to shared memory tiers (TEAM/ORG/SYSTEM).

    ENABLED: Full read/write to all shared tiers.
    READ_ONLY: Can read shared tiers, cannot write to them.
    DISABLED: All shared writes blocked; only PRIVATE pool is writable.
    """

    ENABLED = "enabled"
    READ_ONLY = "read_only"
    DISABLED = "disabled"
