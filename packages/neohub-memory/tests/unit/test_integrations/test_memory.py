"""Tests for MemoryConfig (connector configuration)."""

from neo_memory_hub import MemoryConfig


class TestMemoryConfig:
    """Test MemoryConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = MemoryConfig()
        assert config.vector_store["provider"] == "qdrant"
        assert config.llm["provider"] == "openai"
        assert config.embedder["provider"] == "openai"

    def test_custom_config(self):
        """Test custom configuration."""
        config = MemoryConfig(
            vector_store={"provider": "qdrant", "config": {"url": "http://custom:6333"}},
            llm={"provider": "anthropic", "config": {"model": "claude-3-sonnet"}},
        )
        assert config.vector_store["provider"] == "qdrant"
        assert config.llm["provider"] == "anthropic"

    def test_to_mem0_config(self):
        """Test conversion to Mem0 config dictionary."""
        config = MemoryConfig()
        d = config.to_mem0_config()
        assert "vector_store" in d
        assert "llm" in d
        assert "embedder" in d

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "vector_store": {"provider": "milvus", "config": {}},
            "llm": {"provider": "openai", "config": {"model": "gpt-4"}},
            "embedder": {"provider": "openai", "config": {}},
        }
        config = MemoryConfig.from_dict(d)
        assert config.vector_store["provider"] == "milvus"
        assert config.llm["config"]["model"] == "gpt-4"

    def test_for_pgvector(self):
        """Test pgvector configuration helper."""
        config = MemoryConfig.for_pgvector(
            connection_string="postgresql://user:pass@localhost:5432/neo_memory"
        )
        assert config.vector_store["provider"] == "pgvector"
        assert config.vector_store["config"]["connection_string"] == (
            "postgresql://user:pass@localhost:5432/neo_memory"
        )
