"""Memory entry and result domain models."""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field, field_validator

from neo_memory_hub.domain.scope import IsolationScope
from neo_memory_hub.domain.types import DEFAULT_RETENTION_DAYS, MemoryType


def utcnow() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(UTC)


def generate_id() -> str:
    """Generate a unique memory ID."""
    return str(uuid4())


class MemoryEntry(BaseModel):
    """
    Core domain model for a memory entry.

    Represents a single piece of information stored in the memory system.
    Can be facts, events, preferences, procedures, etc.

    Attributes:
        id: Unique identifier for the memory
        content: The actual memory content (text)
        memory_type: Type classification (episodic, persona, etc.)
        scope: Isolation scope (tenant, user, agent, session)
        created_at: When the memory was created
        updated_at: When the memory was last updated
        accessed_at: When the memory was last accessed (for ranking)
        importance: Importance score (1-10) for ranking
        embedding: Vector embedding (optional, set by embedding service)
        metadata: Additional metadata (tags, source, etc.)
        is_deleted: Soft delete flag

    Example:
        >>> entry = MemoryEntry(
        ...     content="User prefers dark mode for the interface",
        ...     memory_type=MemoryType.PERSONA,
        ...     scope=IsolationScope(tenant_id="acme", user_id="user123")
        ... )
    """

    id: str = Field(default_factory=generate_id, description="Unique memory identifier")
    content: str = Field(..., min_length=1, max_length=10000, description="Memory content text")
    memory_type: MemoryType = Field(default=MemoryType.EPISODIC, description="Type of memory")
    scope: IsolationScope = Field(..., description="Isolation scope for this memory")
    created_at: datetime = Field(default_factory=utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=utcnow, description="Last update timestamp")
    accessed_at: datetime = Field(
        default_factory=utcnow, description="Last access timestamp (for recency ranking)"
    )
    importance: float = Field(default=5.0, ge=1.0, le=10.0, description="Importance score (1-10)")
    embedding: list[float] | None = Field(default=None, description="Vector embedding")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    is_deleted: bool = Field(default=False, description="Soft delete flag")
    expires_at: datetime | None = Field(
        default=None, description="Expiration datetime (UTC). None = permanent."
    )

    @field_validator("content", mode="before")
    @classmethod
    def strip_content(cls, v: str) -> str:
        """Strip whitespace from content."""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("Content cannot be empty after stripping whitespace")
        return v

    @field_validator("importance", mode="before")
    @classmethod
    def round_importance(cls, v: float) -> float:
        """Round importance to 1 decimal place."""
        if isinstance(v, (int, float)):
            return round(float(v), 1)
        return v

    def touch(self) -> None:
        """Update accessed_at timestamp."""
        self.accessed_at = utcnow()

    def update_content(self, new_content: str) -> None:
        """
        Update memory content and timestamps.

        Args:
            new_content: New content text

        Raises:
            ValueError: If memory type is read-only
        """
        if self.memory_type in MemoryType.read_only_types():
            raise ValueError(
                f"Cannot update content of {self.memory_type.value} memory (read-only)"
            )
        self.content = new_content.strip()
        self.updated_at = utcnow()

    def soft_delete(self) -> None:
        """Mark memory as deleted (soft delete)."""
        self.is_deleted = True
        self.updated_at = utcnow()

    @computed_field
    @property
    def age_hours(self) -> float:
        """Calculate age in hours since creation."""
        now = utcnow()
        delta = now - self.created_at
        return delta.total_seconds() / 3600

    @computed_field
    @property
    def hours_since_access(self) -> float:
        """Calculate hours since last access."""
        now = utcnow()
        delta = now - self.accessed_at
        return delta.total_seconds() / 3600

    @computed_field
    @property
    def is_expired(self) -> bool:
        """Check if this memory has passed its expiration time."""
        if self.expires_at is None:
            return False
        return utcnow() > self.expires_at

    @classmethod
    def calculate_expiry(
        cls,
        memory_type: MemoryType,
        created_at: datetime | None = None,
        custom_retention_days: int | None = None,
    ) -> datetime | None:
        """
        Calculate expiration datetime based on memory type retention policy.

        Args:
            memory_type: The type of memory (determines default retention)
            created_at: Base datetime (defaults to now UTC)
            custom_retention_days: Override the default retention days

        Returns:
            Expiration datetime or None if permanent
        """
        if custom_retention_days is not None:
            retention = custom_retention_days
        else:
            retention = DEFAULT_RETENTION_DAYS.get(memory_type)

        if retention is None:
            return None

        base = created_at or utcnow()
        return base + timedelta(days=retention)

    model_config = {
        "frozen": False,
        "validate_assignment": True,
        "extra": "forbid",
        "json_schema_extra": {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "content": "User prefers dark mode",
                "memory_type": "persona",
                "scope": {"tenant_id": "acme_corp", "user_id": "user_123"},
                "importance": 7.5,
            }
        },
    }


class MemoryResult(BaseModel):
    """
    Memory retrieval result with scoring information.

    Wraps a MemoryEntry with relevance and ranking scores
    computed during retrieval.

    Attributes:
        entry: The memory entry
        relevance_score: Vector similarity score (0-1)
        recency_score: Recency score based on access time (0-1)
        importance_score: Normalized importance (0-1)
        final_score: Combined ranking score
        token_count: Estimated token count for the content
        retrieval_method: How the memory was retrieved
    """

    entry: MemoryEntry = Field(..., description="The memory entry")
    relevance_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Vector similarity score"
    )
    recency_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Recency score")
    importance_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Normalized importance score"
    )
    final_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Combined ranking score")
    token_count: int = Field(default=0, ge=0, description="Estimated token count")
    retrieval_method: str = Field(default="vector", description="Retrieval method used")

    @classmethod
    def from_entry(
        cls,
        entry: MemoryEntry,
        relevance_score: float = 0.0,
        token_count: int = 0,
        retrieval_method: str = "vector",
    ) -> "MemoryResult":
        """
        Create MemoryResult from a MemoryEntry.

        Args:
            entry: The memory entry
            relevance_score: Vector similarity score
            token_count: Estimated tokens
            retrieval_method: How it was retrieved

        Returns:
            MemoryResult instance
        """
        return cls(
            entry=entry,
            relevance_score=relevance_score,
            token_count=token_count,
            retrieval_method=retrieval_method,
        )

    model_config = {
        "frozen": False,
        "validate_assignment": True,
    }
