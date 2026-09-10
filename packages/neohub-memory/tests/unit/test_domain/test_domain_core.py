"""
Consolidated unit tests for domain models.

Tests critical functionality for:
- MemoryEntry: creation, validation, mutations
- MemoryResult: scoring, conversion
- IsolationScope: hierarchy, access control
- MemoryType: classification
"""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError as PydanticValidationError

from neo_memory_hub.domain.memory import MemoryEntry, MemoryResult
from neo_memory_hub.domain.scope import IsolationScope, AccessTier
from neo_memory_hub.domain.types import (
    DEFAULT_RETENTION_DAYS,
    DEFAULT_STORAGE_BACKENDS,
    MEMORY_TYPE_CONFIG,
    MemoryType,
)


# ============================================================================
# MemoryEntry Tests
# ============================================================================


class TestMemoryEntry:
    """Core tests for MemoryEntry model."""

    def test_create_with_defaults(self) -> None:
        """Should create entry with sensible defaults."""
        scope = IsolationScope(tenant_id="tenant", user_id="user")
        entry = MemoryEntry(content="Test memory", scope=scope)

        assert entry.content == "Test memory"
        assert entry.memory_type == MemoryType.EPISODIC
        assert entry.importance == 5.0
        assert entry.is_deleted is False
        assert entry.id is not None
        assert entry.created_at is not None

    def test_validation_empty_content_rejected(self) -> None:
        """Should reject empty or whitespace-only content."""
        scope = IsolationScope(tenant_id="tenant", user_id="user")

        with pytest.raises(PydanticValidationError):
            MemoryEntry(content="", scope=scope)

        with pytest.raises(PydanticValidationError):
            MemoryEntry(content="   ", scope=scope)

    def test_validation_importance_range(self) -> None:
        """Should enforce importance in 1-10 range."""
        scope = IsolationScope(tenant_id="tenant", user_id="user")

        with pytest.raises(PydanticValidationError):
            MemoryEntry(content="Test", scope=scope, importance=0.5)

        with pytest.raises(PydanticValidationError):
            MemoryEntry(content="Test", scope=scope, importance=11.0)

    def test_read_only_types_cannot_update(self) -> None:
        """Read-only types should reject content updates."""
        scope = IsolationScope(tenant_id="tenant", user_id="user")

        # SECURITY is read-only
        entry = MemoryEntry(
            content="Security policy",
            scope=scope,
            memory_type=MemoryType.SECURITY,
        )
        with pytest.raises(ValueError, match="read-only"):
            entry.update_content("Modified policy")

        # EPISODIC is read-only
        entry2 = MemoryEntry(
            content="Past event",
            scope=scope,
            memory_type=MemoryType.EPISODIC,
        )
        with pytest.raises(ValueError, match="read-only"):
            entry2.update_content("Modified event")

    def test_soft_delete(self) -> None:
        """Soft delete should set flag without removing."""
        scope = IsolationScope(tenant_id="tenant", user_id="user")
        entry = MemoryEntry(content="Test", scope=scope)

        assert entry.is_deleted is False
        entry.soft_delete()
        assert entry.is_deleted is True


# ============================================================================
# MemoryResult Tests
# ============================================================================


class TestMemoryResult:
    """Core tests for MemoryResult model."""

    def test_from_entry_factory(self) -> None:
        """Should create result from entry with scores."""
        scope = IsolationScope(tenant_id="tenant", user_id="user")
        entry = MemoryEntry(content="Test", scope=scope)

        result = MemoryResult.from_entry(
            entry=entry,
            relevance_score=0.85,
            token_count=20,
        )

        assert result.entry == entry
        assert result.relevance_score == 0.85
        assert result.token_count == 20

    def test_score_validation(self) -> None:
        """Scores must be in 0-1 range."""
        scope = IsolationScope(tenant_id="tenant", user_id="user")
        entry = MemoryEntry(content="Test", scope=scope)

        with pytest.raises(PydanticValidationError):
            MemoryResult(entry=entry, relevance_score=1.5)

        with pytest.raises(PydanticValidationError):
            MemoryResult(entry=entry, relevance_score=-0.1)


# ============================================================================
# IsolationScope Tests
# ============================================================================


class TestIsolationScope:
    """Core tests for IsolationScope model."""

    def test_hierarchy_validation(self) -> None:
        """Should enforce scope hierarchy rules."""
        # session_id requires agent_id
        with pytest.raises(PydanticValidationError, match="session_id requires agent_id"):
            IsolationScope(
                tenant_id="tenant",
                user_id="user",
                session_id="session",
            )

        # agent_id requires user_id for non-SYSTEM pools
        with pytest.raises(PydanticValidationError, match="agent_id requires user_id"):
            IsolationScope(
                tenant_id="tenant",
                agent_id="agent",
                pool=AccessTier.PRIVATE,
            )

    def test_system_pool_allows_agent_without_user(self) -> None:
        """SYSTEM pool should allow agent without user."""
        scope = IsolationScope(
            tenant_id="tenant",
            agent_id="agent",
            pool=AccessTier.SYSTEM,
        )
        assert scope.agent_id == "agent"

    def test_composite_key_generation(self) -> None:
        """Should generate correct composite keys (v0.2: tenant:dept:user)."""
        scope = IsolationScope(
            tenant_id="tenant",
            dept_id="wealth",
            user_id="user",
            agent_id="agent",
            session_id="session",
        )
        assert scope.to_composite_key() == "tenant:wealth:user"

        no_dept = IsolationScope(
            tenant_id="tenant",
            user_id="user",
        )
        assert no_dept.to_composite_key() == "tenant:user"

        minimal = IsolationScope(tenant_id="tenant", pool=AccessTier.SYSTEM)
        assert minimal.to_composite_key() == "tenant"

    def test_access_control_cross_tenant_denied(self) -> None:
        """Cross-tenant access should always be denied."""
        requester = IsolationScope(tenant_id="tenant_a", user_id="user")
        target = IsolationScope(tenant_id="tenant_b", pool=AccessTier.SYSTEM)
        assert requester.can_access(target) is False

    def test_access_control_same_tenant_system_allowed(self) -> None:
        """SYSTEM pool should be accessible to same tenant."""
        requester = IsolationScope(tenant_id="tenant", user_id="user")
        target = IsolationScope(tenant_id="tenant", pool=AccessTier.SYSTEM)
        assert requester.can_access(target) is True

    def test_access_control_private_requires_same_agent(self) -> None:
        """PRIVATE pool requires same user and agent."""
        owner = IsolationScope(
            tenant_id="tenant",
            user_id="user",
            agent_id="agent_a",
        )
        target = IsolationScope(
            tenant_id="tenant",
            user_id="user",
            agent_id="agent_a",
            pool=AccessTier.PRIVATE,
        )
        other_agent = IsolationScope(
            tenant_id="tenant",
            user_id="user",
            agent_id="agent_b",
        )

        assert owner.can_access(target) is True
        assert other_agent.can_access(target) is False


# ============================================================================
# MemoryType Tests
# ============================================================================


class TestMemoryType:
    """Core tests for MemoryType enum."""

    def test_type_classification(self) -> None:
        """Types should be properly classified."""
        # Read-only types
        assert MemoryType.SECURITY in MemoryType.read_only_types()
        assert MemoryType.EPISODIC in MemoryType.read_only_types()
        assert MemoryType.PERSONA not in MemoryType.read_only_types()

        # Volatile types use Redis
        assert MemoryType.WORKING in MemoryType.volatile_types()
        assert MemoryType.PERSONA not in MemoryType.volatile_types()

    def test_from_string_case_insensitive(self) -> None:
        """Should parse type strings case-insensitively."""
        assert MemoryType.from_string("episodic") == MemoryType.EPISODIC
        assert MemoryType.from_string("PERSONA") == MemoryType.PERSONA
        assert MemoryType.from_string("Working") == MemoryType.WORKING

    def test_from_string_invalid_raises(self) -> None:
        """Invalid type string should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid memory type"):
            MemoryType.from_string("unknown_type")

    def test_storage_backend_mapping(self) -> None:
        """Each type should have a storage backend."""
        for mem_type in MemoryType:
            assert mem_type in DEFAULT_STORAGE_BACKENDS

        assert DEFAULT_STORAGE_BACKENDS[MemoryType.WORKING] == "redis"
        assert DEFAULT_STORAGE_BACKENDS[MemoryType.PERSONA] == "milvus"

    def test_retention_mapping(self) -> None:
        """Each type should have retention config."""
        for mem_type in MemoryType:
            assert mem_type in DEFAULT_RETENTION_DAYS

        assert DEFAULT_RETENTION_DAYS[MemoryType.EPISODIC] == 365
        assert DEFAULT_RETENTION_DAYS[MemoryType.PERSONA] is None  # Permanent

    def test_conversation_type_classification(self) -> None:
        """CONVERSATION type should be properly configured."""
        assert MemoryType.CONVERSATION in MemoryType
        assert MemoryType.CONVERSATION in MemoryType.summarizable_types()
        # CONVERSATION is read-only (chat history should not be modified)
        assert MemoryType.CONVERSATION in MemoryType.read_only_types()
        assert DEFAULT_STORAGE_BACKENDS[MemoryType.CONVERSATION] == "milvus"
        assert DEFAULT_RETENTION_DAYS[MemoryType.CONVERSATION] == 90

    def test_entity_type_classification(self) -> None:
        """ENTITY type should be properly configured."""
        assert MemoryType.ENTITY in MemoryType
        assert MemoryType.ENTITY in MemoryType.graph_types()
        assert MemoryType.ENTITY not in MemoryType.read_only_types()
        # ENTITY stored in Milvus (relationships in Neo4j handled separately)
        assert DEFAULT_STORAGE_BACKENDS[MemoryType.ENTITY] == "milvus"
        assert DEFAULT_RETENTION_DAYS[MemoryType.ENTITY] is None  # Permanent

    def test_graph_types_method(self) -> None:
        """Should return types that use graph storage."""
        graph_types = MemoryType.graph_types()
        assert MemoryType.ENTITY in graph_types
        assert MemoryType.EPISODIC not in graph_types

    def test_summarizable_types_method(self) -> None:
        """Should return types that can be summarized."""
        summarizable = MemoryType.summarizable_types()
        assert MemoryType.CONVERSATION in summarizable
        assert MemoryType.EPISODIC in summarizable
        assert MemoryType.SECURITY not in summarizable

    def test_memory_type_config_completeness(self) -> None:
        """All memory types should have complete config."""
        for mem_type in MemoryType:
            assert mem_type in MEMORY_TYPE_CONFIG, f"Missing config for {mem_type}"
            config = MEMORY_TYPE_CONFIG[mem_type]
            assert "editable" in config
            assert "auto_summarize" in config
            assert "graph_enabled" in config
            assert "always_load" in config
            assert "description" in config
