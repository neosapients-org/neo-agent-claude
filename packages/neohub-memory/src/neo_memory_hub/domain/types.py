"""Memory types enumeration and configuration."""

from enum import Enum
from typing import Literal


class MemoryType(str, Enum):
    """
    Enumeration of supported memory types.

    Each type has specific behaviors, retention policies, and storage backends.

    Core Types:
    - EPISODIC: Past events, interactions (365 days retention, not editable)
    - PERSONA: User preferences, characteristics (permanent, editable)
    - SECURITY: Policies, guardrails, rules (permanent, read-only)
    - SEMANTIC: Facts, knowledge, learned information (permanent, editable)
    - PROCEDURAL: Workflows, processes, how-to (permanent, editable)
    - WORKING: Current session context (session-scoped, Redis-backed)


    - CONVERSATION: Chat history with summarization (configurable retention)
    - ENTITY: People, companies, products with relationships (permanent, graph-backed)

    Extended Types:
    - TOOL: Tool definitions and usage patterns
    - REFLECTION: Agent self-insights
    - FEEDBACK: User corrections
    """

    EPISODIC = "episodic"
    PERSONA = "persona"
    SECURITY = "security"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"

    CONVERSATION = "conversation"
    ENTITY = "entity"

    # Extended types
    TOOL = "tool"
    REFLECTION = "reflection"
    FEEDBACK = "feedback"

    @classmethod
    def core_types(cls) -> list["MemoryType"]:
        """Get list of core memory types."""
        return [
            cls.EPISODIC,
            cls.PERSONA,
            cls.SECURITY,
            cls.SEMANTIC,
            cls.PROCEDURAL,
            cls.WORKING,
            cls.CONVERSATION,
            cls.ENTITY,
        ]

    @classmethod
    def persistent_types(cls) -> list["MemoryType"]:
        """Get types that use persistent storage (vector DB)."""
        return [
            cls.EPISODIC,
            cls.PERSONA,
            cls.SECURITY,
            cls.SEMANTIC,
            cls.PROCEDURAL,
            cls.CONVERSATION,
            cls.ENTITY,
            cls.TOOL,
            cls.REFLECTION,
            cls.FEEDBACK,
        ]

    @classmethod
    def volatile_types(cls) -> list["MemoryType"]:
        """Get types that use volatile storage (Redis)."""
        return [cls.WORKING]

    @classmethod
    def read_only_types(cls) -> list["MemoryType"]:
        """Get types that are read-only after creation."""
        return [cls.SECURITY, cls.EPISODIC, cls.CONVERSATION]

    @classmethod
    def summarizable_types(cls) -> list["MemoryType"]:
        """Get types that support summarization."""
        return [cls.CONVERSATION, cls.EPISODIC]

    @classmethod
    def graph_types(cls) -> list["MemoryType"]:
        """Get types that use graph storage for relationships."""
        return [cls.ENTITY, cls.PROCEDURAL]

    @classmethod
    def from_string(cls, value: str) -> "MemoryType":
        """
        Create MemoryType from string value.

        Args:
            value: String representation of memory type

        Returns:
            MemoryType enum member

        Raises:
            ValueError: If value doesn't match any memory type
        """
        try:
            return cls(value.lower())
        except ValueError as e:
            valid_types = [t.value for t in cls]
            raise ValueError(f"Invalid memory type: '{value}'. Valid types: {valid_types}") from e


# Type alias for storage backend selection
StorageBackend = Literal["milvus", "qdrant", "redis", "neo4j"]


# Default storage backend mapping
DEFAULT_STORAGE_BACKENDS: dict[MemoryType, StorageBackend] = {
    MemoryType.EPISODIC: "milvus",
    MemoryType.PERSONA: "milvus",
    MemoryType.SECURITY: "milvus",
    MemoryType.SEMANTIC: "milvus",
    MemoryType.PROCEDURAL: "milvus",
    MemoryType.WORKING: "redis",
    MemoryType.CONVERSATION: "milvus",
    MemoryType.ENTITY: "milvus",  # Primary in Milvus, relationships in Neo4j
    MemoryType.TOOL: "milvus",
    MemoryType.REFLECTION: "milvus",
    MemoryType.FEEDBACK: "milvus",
}


# Default retention in days (None = permanent)
DEFAULT_RETENTION_DAYS: dict[MemoryType, int | None] = {
    MemoryType.EPISODIC: 365,
    MemoryType.PERSONA: None,
    MemoryType.SECURITY: None,
    MemoryType.SEMANTIC: None,
    MemoryType.PROCEDURAL: None,
    MemoryType.WORKING: 1,
    MemoryType.CONVERSATION: 90,  # 90 days default, configurable
    MemoryType.ENTITY: None,  # Permanent
    MemoryType.TOOL: None,
    MemoryType.REFLECTION: None,
    MemoryType.FEEDBACK: 180,
}


# Memory type configuration for routing
MEMORY_TYPE_CONFIG: dict[MemoryType, dict] = {
    MemoryType.EPISODIC: {
        "editable": False,
        "auto_summarize": True,
        "graph_enabled": False,
        "always_load": False,
        "description": "Past events and specific experiences",
    },
    MemoryType.PERSONA: {
        "editable": True,
        "auto_summarize": False,
        "graph_enabled": False,
        "always_load": True,  # Core memory - always included in context
        "description": "User preferences, traits, and identity",
    },
    MemoryType.SECURITY: {
        "editable": False,
        "auto_summarize": False,
        "graph_enabled": False,
        "always_load": True,  # Core memory - guardrails always active
        "description": "Policies, guardrails, and constraints",
    },
    MemoryType.SEMANTIC: {
        "editable": True,
        "auto_summarize": False,
        "graph_enabled": False,
        "always_load": False,
        "description": "Facts, domain knowledge, learned information",
    },
    MemoryType.PROCEDURAL: {
        "editable": True,
        "auto_summarize": False,
        "graph_enabled": True,  # Workflows can link to each other
        "always_load": False,
        "description": "Workflows, SOPs, step-by-step processes",
    },
    MemoryType.WORKING: {
        "editable": True,
        "auto_summarize": False,
        "graph_enabled": False,
        "always_load": False,
        "description": "Current session context, short-term memory",
    },
    MemoryType.CONVERSATION: {
        "editable": False,
        "auto_summarize": True,
        "graph_enabled": False,
        "always_load": False,
        "description": "Chat history with automatic summarization",
    },
    MemoryType.ENTITY: {
        "editable": True,
        "auto_summarize": False,
        "graph_enabled": True,  # Relationships stored in Neo4j
        "always_load": False,
        "description": "People, companies, products with relationships",
    },
    MemoryType.TOOL: {
        "editable": True,
        "auto_summarize": False,
        "graph_enabled": False,
        "always_load": False,
        "description": "Tool definitions, schemas, usage patterns",
    },
    MemoryType.REFLECTION: {
        "editable": True,
        "auto_summarize": False,
        "graph_enabled": False,
        "always_load": False,
        "description": "Agent self-insights and learnings",
    },
    MemoryType.FEEDBACK: {
        "editable": True,
        "auto_summarize": False,
        "graph_enabled": False,
        "always_load": False,
        "description": "User corrections and feedback",
    },
}
