"""Isolation scope and access tier definitions for enterprise memory isolation."""

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class AccessTier(str, Enum):
    """
    Memory access tiers for enterprise isolation.

    Hierarchy (most restrictive → least restrictive):
        PRIVATE < SHARED < TEAM < ORG < SYSTEM

    RESTRICTED is orthogonal — removes memory from normal retrieval.
    Only admin/compliance APIs can access RESTRICTED memories.
    """

    PRIVATE = "private"       # Only the owner agent
    SHARED = "shared"         # All agents of the same user
    TEAM = "team"             # All users in the same department (requires dept_id)
    ORG = "org"               # All users in the same tenant
    SYSTEM = "system"         # Cross-tenant system memories (platform guardrails)
    RESTRICTED = "restricted"  # Compliance/audit-only — blocked from normal retrieval


# Pattern for valid identifiers (alphanumeric + underscore + hyphen)
ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


class IsolationScope(BaseModel):
    """
    Defines the isolation boundaries for memory operations.

    Implements 6-tier enterprise isolation model:
    1. Tenant (Organization) - Required, always enforced
    2. Department - Optional, required for TEAM tier
    3. User - Required for non-system/org memories
    4. Agent - Optional, for multi-agent scenarios
    5. Session - Optional, for session-scoped working memory
    6. Subject - Optional, identifies the subject of the memory (e.g. customer ID)

    Hierarchy Rules:
    - session_id requires agent_id
    - agent_id requires user_id (except for SYSTEM/ORG pools)
    - TEAM tier requires dept_id
    - All require tenant_id

    Example:
        >>> scope = IsolationScope(
        ...     tenant_id="acme_corp",
        ...     dept_id="wealth",
        ...     user_id="rm_001",
        ...     agent_id="vic_l3",
        ...     subject_id="cust_sharma_99",
        ...     pool=AccessTier.TEAM,
        ... )
    """

    tenant_id: str = Field(
        ..., min_length=1, max_length=64, description="Tenant/organization identifier (required)"
    )
    dept_id: str | None = Field(
        default=None, max_length=64, description="Department identifier (required for TEAM tier)"
    )
    user_id: str | None = Field(
        default=None, max_length=64, description="User identifier within tenant"
    )
    agent_id: str | None = Field(
        default=None, max_length=64, description="Agent identifier within user context"
    )
    session_id: str | None = Field(
        default=None, max_length=64, description="Session identifier for working memory"
    )
    subject_id: str | None = Field(
        default=None,
        max_length=128,
        description="Subject identifier (e.g. customer ID the memory is about)",
    )
    pool: AccessTier = Field(default=AccessTier.SHARED, description="Memory access tier")

    @field_validator(
        "tenant_id", "dept_id", "user_id", "agent_id", "session_id", "subject_id", mode="before"
    )
    @classmethod
    def validate_identifier(cls, v: str | None) -> str | None:
        """Validate identifier format (alphanumeric + underscore + hyphen)."""
        if v is None:
            return None

        v = str(v).strip()
        if not v:
            return None

        if not ID_PATTERN.match(v):
            raise ValueError(
                f"Invalid identifier format: '{v}'. "
                "Only alphanumeric characters, underscores, and hyphens allowed."
            )
        return v

    @model_validator(mode="after")
    def validate_hierarchy(self) -> "IsolationScope":
        """Validate isolation hierarchy rules."""
        # session_id requires agent_id
        if self.session_id and not self.agent_id:
            raise ValueError("session_id requires agent_id to be set")

        # agent_id requires user_id (except for SYSTEM and ORG pools)
        if (
            self.agent_id
            and not self.user_id
            and self.pool not in (AccessTier.SYSTEM, AccessTier.ORG)
        ):
            raise ValueError("agent_id requires user_id for non-SYSTEM/ORG pools")

        # TEAM tier requires dept_id
        if self.pool == AccessTier.TEAM and not self.dept_id:
            raise ValueError("TEAM tier requires dept_id to be set")

        return self

    def to_composite_key(self) -> str:
        """
        Generate composite key for storage partitioning.

        Format: tenant_id:dept_id:user_id (simplified for v0.2)
        Missing components are omitted.

        Returns:
            Composite key string
        """
        parts = [self.tenant_id]

        if self.dept_id:
            parts.append(self.dept_id)
        if self.user_id:
            parts.append(self.user_id)

        return ":".join(parts)

    def to_filter_dict(self) -> dict[str, Any]:
        """
        Generate filter dictionary for database queries.

        Returns:
            Dictionary of non-None scope fields
        """
        filters: dict[str, Any] = {"tenant_id": self.tenant_id}

        if self.dept_id:
            filters["dept_id"] = self.dept_id
        if self.user_id:
            filters["user_id"] = self.user_id
        if self.agent_id:
            filters["agent_id"] = self.agent_id
        if self.session_id:
            filters["session_id"] = self.session_id
        if self.subject_id:
            filters["subject_id"] = self.subject_id

        filters["pool"] = self.pool.value
        return filters

    def to_mem0_params(self) -> dict[str, Any]:
        """
        Generate Mem0 parameters using native scoping.

        Uses Mem0's native user_id, agent_id, and run_id parameters,
        while carrying tenant_id, dept_id, subject_id, and pool in metadata.

        For ORG and SYSTEM tiers, user_id/agent_id are omitted so that
        memories are visible to all users in the tenant.

        Returns:
            Dictionary with user_id, agent_id, run_id, and metadata
        """
        params: dict[str, Any] = {}

        # Use native Mem0 scoping parameters
        if self.user_id:
            params["user_id"] = self.user_id
        if self.agent_id:
            params["agent_id"] = self.agent_id
        if self.session_id:
            params["run_id"] = self.session_id

        # Carry tenant_id, pool, and new fields in metadata for filtering
        metadata: dict[str, Any] = {
            "tenant_id": self.tenant_id,
            "pool": self.pool.value,
        }
        if self.dept_id:
            metadata["dept_id"] = self.dept_id
        if self.subject_id:
            metadata["subject_id"] = self.subject_id

        params["metadata"] = metadata

        # For ORG/SYSTEM pool memories (cross-user), replace user-level
        # identity with a synthetic agent_id derived from tenant_id.
        # Mem0 requires at least one of user_id/agent_id/run_id.
        if self.pool in (AccessTier.SYSTEM, AccessTier.ORG):
            params.pop("user_id", None)
            params.pop("agent_id", None)
            params["agent_id"] = f"__org_{self.tenant_id}__"

        return params

    def to_mem0_filters(self) -> dict[str, Any]:
        """
        Generate Mem0 search filters including tenant isolation.

        Filter strategy per tier:
        - PRIVATE: tenant + user + agent + pool
        - SHARED: tenant + user + pool
        - TEAM: tenant + dept_id + pool
        - ORG: tenant + pool
        - SYSTEM: tenant + pool
        - RESTRICTED: tenant + pool (but should be blocked by can_access)

        Returns:
            Dictionary of filters for Mem0 search
        """
        filters: dict[str, Any] = {
            "tenant_id": self.tenant_id,
        }

        if self.pool == AccessTier.PRIVATE:
            filters["pool"] = self.pool.value
            if self.user_id:
                filters["user_id"] = self.user_id
            if self.agent_id:
                filters["agent_id"] = self.agent_id

        elif self.pool == AccessTier.SHARED:
            filters["pool"] = self.pool.value
            if self.user_id:
                filters["user_id"] = self.user_id

        elif self.pool == AccessTier.TEAM:
            filters["pool"] = self.pool.value
            if self.dept_id:
                filters["dept_id"] = self.dept_id

        elif self.pool in (AccessTier.ORG, AccessTier.SYSTEM):
            filters["pool"] = self.pool.value

        elif self.pool == AccessTier.RESTRICTED:
            filters["pool"] = self.pool.value

        # Add subject_id filter if set
        if self.subject_id:
            filters["subject_id"] = self.subject_id

        return filters

    def can_access(self, other: "IsolationScope") -> bool:
        """
        Check if this scope can access another scope's memories.

        Access Rules (Memory Firewall):
        - Same tenant required (always)
        - RESTRICTED: never accessible via normal API
        - SYSTEM: anyone in tenant
        - ORG: anyone in tenant
        - TEAM: same department required
        - SHARED: same user required
        - PRIVATE: same user AND same agent required

        Args:
            other: The scope to check access against

        Returns:
            True if this scope can access other's memories
        """
        # Must be same tenant
        if self.tenant_id != other.tenant_id:
            return False

        # RESTRICTED — never accessible via normal API
        if other.pool == AccessTier.RESTRICTED:
            return False

        # SYSTEM pool: anyone in tenant can access
        if other.pool == AccessTier.SYSTEM:
            return True

        # ORG pool: anyone in tenant can access
        if other.pool == AccessTier.ORG:
            return True

        # TEAM pool: same department required
        if other.pool == AccessTier.TEAM:
            if not self.dept_id or not other.dept_id:
                return False
            return self.dept_id == other.dept_id

        # SHARED pool: same user required (explicit None checks)
        if other.pool == AccessTier.SHARED:
            return self.user_id is not None and self.user_id == other.user_id

        # PRIVATE pool: same agent required (explicit None checks)
        if other.pool == AccessTier.PRIVATE:
            return (
                self.user_id is not None
                and self.user_id == other.user_id
                and self.agent_id is not None
                and self.agent_id == other.agent_id
            )

        return False

    def check_access(self, other: "IsolationScope") -> None:
        """
        Check access and raise AccessDeniedError if denied.

        Args:
            other: The scope to check access against

        Raises:
            AccessDeniedError: If access is denied
        """
        if not self.can_access(other):
            from neo_memory_hub.core.exceptions import AccessDeniedError

            raise AccessDeniedError(
                message=(
                    f"Access denied: {self.pool.value} scope cannot access "
                    f"{other.pool.value} memory"
                ),
                requester_scope=self.to_filter_dict(),
                target_scope=other.to_filter_dict(),
            )

    def with_pool(self, pool: AccessTier) -> "IsolationScope":
        """Create a new scope with different pool."""
        return IsolationScope(
            tenant_id=self.tenant_id,
            dept_id=self.dept_id,
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=self.session_id,
            subject_id=self.subject_id,
            pool=pool,
        )

    def with_session(self, session_id: str) -> "IsolationScope":
        """Create a new scope with session_id."""
        return IsolationScope(
            tenant_id=self.tenant_id,
            dept_id=self.dept_id,
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=session_id,
            subject_id=self.subject_id,
            pool=self.pool,
        )

    def with_dept(self, dept_id: str) -> "IsolationScope":
        """Create a new scope with dept_id."""
        return IsolationScope(
            tenant_id=self.tenant_id,
            dept_id=dept_id,
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=self.session_id,
            subject_id=self.subject_id,
            pool=self.pool,
        )

    def with_subject(self, subject_id: str) -> "IsolationScope":
        """Create a new scope with subject_id."""
        return IsolationScope(
            tenant_id=self.tenant_id,
            dept_id=self.dept_id,
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=self.session_id,
            subject_id=subject_id,
            pool=self.pool,
        )

    model_config = {
        "frozen": False,
        "validate_assignment": True,
        "extra": "forbid",
    }
