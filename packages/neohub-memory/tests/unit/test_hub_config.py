"""Tests for MemoryHubConfig and load_config."""
import os
import tempfile

import pytest
import yaml

from neo_memory_hub.config.hub_config import (
    BufferingConfig,
    DedupConfig,
    EmbedderConfig,
    ExtractionConfig,
    FeatureConfig,
    HistoryConfig,
    IsolationConfig,
    LLMConfig,
    MemoryHubConfig,
    MemoryTypeConfig,
    RetrievalConfig,
    SalienceConfig,
    StorageConfig,
    StorageRoutingConfig,
    VectorStoreConfig,
    _resolve_env_vars,
    load_config,
)


class TestMemoryHubConfigDefaults:
    """Test default configuration values."""

    def test_default_config_creates_successfully(self):
        cfg = MemoryHubConfig()
        assert cfg.vector_store.provider == "qdrant"
        assert cfg.llm.model == "gpt-4o-mini"
        assert cfg.embedder.model == "text-embedding-3-small"

    def test_vector_store_defaults(self):
        vs = VectorStoreConfig()
        assert vs.provider == "qdrant"
        assert vs.collection_name == "memories"
        assert vs.qdrant_url == "http://localhost:6333"
        assert vs.embedding_dims == 1536

    def test_llm_defaults(self):
        llm = LLMConfig()
        assert llm.model == "gpt-4o-mini"
        assert llm.temperature == 0.1

    def test_embedder_defaults(self):
        emb = EmbedderConfig()
        assert emb.model == "text-embedding-3-small"
        assert emb.dimensions == 1536
        assert emb.max_input_tokens == 7000
        assert emb.max_input_chars == 40000

    def test_retrieval_defaults(self):
        ret = RetrievalConfig()
        assert ret.limit == 10
        assert ret.min_relevance_score == 0.1
        assert ret.thresholds["persona"] == 0.20
        assert ret.thresholds["procedural"] == 0.30

    def test_salience_defaults(self):
        sal = SalienceConfig()
        assert sal.enabled is True
        assert sal.min_threshold == 0.3
        assert sal.default_score == 0.5

    def test_extraction_defaults(self):
        ext = ExtractionConfig()
        assert ext.enabled is True
        assert ext.prompt is None
        assert "persona" in ext.valid_categories

    def test_dedup_defaults(self):
        dd = DedupConfig()
        assert dd.enabled is True
        assert dd.similarity_threshold == 0.55
        assert dd.max_candidates == 5
        assert dd.model is None

    def test_buffering_defaults(self):
        buf = BufferingConfig()
        assert buf.flush_threshold == 10
        assert buf.max_buffer_chars == 50000

    def test_isolation_defaults(self):
        iso = IsolationConfig()
        assert iso.default_agent_id == "default_agent"
        assert iso.default_pool == "private"
        assert iso.require_user_id is True

    def test_history_defaults(self):
        hist = HistoryConfig()
        assert hist.enabled is True
        assert hist.db_path == ":memory:"

    def test_feature_defaults(self):
        feat = FeatureConfig()
        assert feat.retrieval_enabled is True
        assert feat.storage_enabled is True


class TestMemoryHubConfigCustom:
    """Test custom configuration."""

    def test_override_llm_model(self):
        cfg = MemoryHubConfig(llm=LLMConfig(model="gpt-4o"))
        assert cfg.llm.model == "gpt-4o"

    def test_memory_types_config(self):
        cfg = MemoryHubConfig(
            memory_types={
                "persona": MemoryTypeConfig(importance=9.0, description="Identity"),
                "episodic": MemoryTypeConfig(importance=7.0),
            }
        )
        assert cfg.memory_types["persona"].importance == 9.0
        assert cfg.memory_types["persona"].description == "Identity"
        assert cfg.memory_types["episodic"].importance == 7.0

    def test_get_valid_categories(self):
        cfg = MemoryHubConfig()
        cats = cfg.get_valid_categories()
        assert "persona" in cats
        assert "episodic" in cats
        assert "conversational" in cats

    def test_should_use_qdrant(self):
        cfg = MemoryHubConfig()
        assert cfg.should_use_qdrant("persona") is True
        assert cfg.should_use_qdrant("conversational") is False


class TestLoadConfig:
    """Test load_config function."""

    def test_load_defaults(self):
        cfg = load_config()
        assert isinstance(cfg, MemoryHubConfig)
        assert cfg.llm.model == "gpt-4o-mini"

    def test_load_from_dict(self):
        cfg = load_config(config_dict={
            "llm": {"model": "gpt-4o"},
            "retrieval": {"limit": 20},
        })
        assert cfg.llm.model == "gpt-4o"
        assert cfg.retrieval.limit == 20

    def test_load_from_yaml_file(self):
        data = {
            "llm": {"model": "gpt-4o"},
            "salience": {"enabled": False},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            cfg = load_config(config_path=f.name)
        os.unlink(f.name)
        assert cfg.llm.model == "gpt-4o"
        assert cfg.salience.enabled is False

    def test_load_missing_file_returns_defaults(self):
        cfg = load_config(config_path="/nonexistent/path.yaml")
        assert isinstance(cfg, MemoryHubConfig)
        assert cfg.llm.model == "gpt-4o-mini"

    def test_dict_overrides_file(self):
        """When both provided, dict takes priority."""
        cfg = load_config(
            config_path="/nonexistent/path.yaml",
            config_dict={"llm": {"model": "gpt-4o"}},
        )
        assert cfg.llm.model == "gpt-4o"


class TestEnvVarResolution:
    """Test ${VAR:-default} resolution."""

    def test_resolve_with_default(self):
        result = _resolve_env_vars("${NONEXISTENT_VAR:-fallback_value}")
        assert result == "fallback_value"

    def test_resolve_null_default(self):
        result = _resolve_env_vars("${NONEXISTENT_VAR:-null}")
        assert result is None

    def test_resolve_no_default(self):
        result = _resolve_env_vars("${NONEXISTENT_VAR_XYZ}")
        assert result == ""

    def test_resolve_from_env(self, monkeypatch):
        monkeypatch.setenv("TEST_NEO_VAR", "real_value")
        result = _resolve_env_vars("${TEST_NEO_VAR:-default}")
        assert result == "real_value"

    def test_resolve_dict(self):
        data = {"key": "${MISSING:-hello}", "nested": {"inner": "${MISSING:-world}"}}
        result = _resolve_env_vars(data)
        assert result["key"] == "hello"
        assert result["nested"]["inner"] == "world"

    def test_resolve_list(self):
        data = ["${MISSING:-a}", "${MISSING:-b}", "plain"]
        result = _resolve_env_vars(data)
        assert result == ["a", "b", "plain"]

    def test_non_env_string_passthrough(self):
        result = _resolve_env_vars("plain string")
        assert result == "plain string"

    def test_int_passthrough(self):
        result = _resolve_env_vars(42)
        assert result == 42

    def test_full_yaml_with_env_vars(self):
        """Simulate loading YAML with env vars."""
        cfg = load_config(config_dict={
            "vector_store": {"provider": "${VECTOR_STORE_PROVIDER:-qdrant}"},
            "llm": {"model": "${LLM_MODEL:-gpt-4o-mini}"},
        })
        assert cfg.vector_store.provider == "qdrant"
        assert cfg.llm.model == "gpt-4o-mini"
