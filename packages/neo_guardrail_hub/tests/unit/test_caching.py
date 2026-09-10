"""
Unit tests for the Rails Caching module (Phase 4).

Tests cover:
1. RailsCache class functionality
2. Cache entry management (get, set, eviction)
3. TTL-based expiration
4. Staleness detection
5. Global cache singleton
"""

import pytest
import asyncio
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from neo_guardrail_hub.api.caching import (
    RailsCache,
    CacheEntry,
    get_rails_cache,
    clear_rails_cache,
    reset_rails_cache,
    _compute_config_hash,
)


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""
    
    def test_entry_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(
            rails=Mock(),
            config=Mock(),
            created_at=time.time(),
            last_accessed=time.time(),
            config_path=Path("/tmp/test"),
            config_hash="abc123",
        )
        
        assert entry.rails is not None
        assert entry.config is not None
        assert entry.config_hash == "abc123"
    
    def test_is_expired_no_ttl(self):
        """Test expiration check with no TTL."""
        entry = CacheEntry(
            rails=Mock(),
            config=Mock(),
            created_at=time.time() - 1000,
            last_accessed=time.time() - 1000,
            config_path=Path("/tmp"),
            config_hash="abc",
        )
        
        assert entry.is_expired(None) is False
    
    def test_is_expired_within_ttl(self):
        """Test entry within TTL."""
        entry = CacheEntry(
            rails=Mock(),
            config=Mock(),
            created_at=time.time(),
            last_accessed=time.time(),
            config_path=Path("/tmp"),
            config_hash="abc",
        )
        
        assert entry.is_expired(60.0) is False
    
    def test_is_expired_past_ttl(self):
        """Test entry past TTL."""
        entry = CacheEntry(
            rails=Mock(),
            config=Mock(),
            created_at=time.time() - 100,
            last_accessed=time.time() - 100,
            config_path=Path("/tmp"),
            config_hash="abc",
        )
        
        assert entry.is_expired(60.0) is True


class TestRailsCache:
    """Tests for RailsCache class."""
    
    def test_cache_creation(self):
        """Test cache instantiation."""
        cache = RailsCache()
        
        assert cache.size == 0
        assert cache._ttl_seconds is None
        assert cache._max_size == 100
    
    def test_cache_creation_with_options(self):
        """Test cache with custom options."""
        cache = RailsCache(
            ttl_seconds=300.0,
            max_size=50,
            check_staleness=False,
        )
        
        assert cache._ttl_seconds == 300.0
        assert cache._max_size == 50
        assert cache._check_staleness is False
    
    def test_contains_empty_cache(self):
        """Test contains on empty cache."""
        cache = RailsCache()
        
        assert cache.contains("nonexistent") is False
    
    def test_get_empty_cache(self):
        """Test get on empty cache."""
        cache = RailsCache()
        
        assert cache.get("nonexistent") is None
    
    def test_remove_from_empty_cache(self):
        """Test remove from empty cache."""
        cache = RailsCache()
        
        assert cache.remove("nonexistent") is False
    
    def test_clear_empty_cache(self):
        """Test clearing empty cache."""
        cache = RailsCache()
        cache.clear()
        
        assert cache.size == 0
    
    def test_stats_empty_cache(self):
        """Test stats for empty cache."""
        cache = RailsCache()
        stats = cache.stats
        
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
    
    def test_list_keys_empty(self):
        """Test listing keys in empty cache."""
        cache = RailsCache()
        
        assert cache.list_keys() == []
    
    @pytest.mark.asyncio
    async def test_get_or_create_mock(self):
        """Test get_or_create with mocked NeMo."""
        cache = RailsCache()
        
        # Mock the _create_rails method
        mock_rails = Mock()
        mock_config = Mock()
        cache._create_rails = AsyncMock(return_value=(mock_rails, mock_config))
        
        # Create temp config directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir)
            (config_path / "config.yml").write_text("models: []")
            
            rails = await cache.get_or_create(
                key="test",
                config_path=config_path,
            )
            
            assert rails == mock_rails
            assert cache.size == 1
            assert cache.contains("test")
    
    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test cache hit behavior."""
        cache = RailsCache()
        
        mock_rails = Mock()
        mock_config = Mock()
        cache._create_rails = AsyncMock(return_value=(mock_rails, mock_config))
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir)
            (config_path / "config.yml").write_text("models: []")
            
            # First call - miss
            rails1 = await cache.get_or_create("test", config_path)
            
            # Second call - hit
            rails2 = await cache.get_or_create("test", config_path)
            
            assert rails1 == rails2
            assert cache.stats["hits"] == 1
            assert cache.stats["misses"] == 1
    
    @pytest.mark.asyncio
    async def test_force_reload(self):
        """Test force reload bypasses cache."""
        cache = RailsCache()
        
        call_count = 0
        
        async def mock_create_rails(config_path):
            nonlocal call_count
            call_count += 1
            return Mock(name=f"rails_{call_count}"), Mock()
        
        cache._create_rails = mock_create_rails
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir)
            (config_path / "config.yml").write_text("models: []")
            
            await cache.get_or_create("test", config_path)
            await cache.get_or_create("test", config_path, force_reload=True)
            
            assert call_count == 2
    
    def test_eviction_on_max_size(self):
        """Test that oldest entries are evicted when max size reached."""
        cache = RailsCache(max_size=2)
        
        # Manually add entries
        now = time.time()
        cache._cache["old"] = CacheEntry(
            rails=Mock(), config=Mock(),
            created_at=now - 100, last_accessed=now - 100,
            config_path=Path("/tmp"), config_hash="a",
        )
        cache._cache["new"] = CacheEntry(
            rails=Mock(), config=Mock(),
            created_at=now, last_accessed=now,
            config_path=Path("/tmp"), config_hash="b",
        )
        
        # Trigger eviction
        cache._evict_if_needed()
        
        assert cache.size == 1
        assert "old" not in cache._cache
        assert "new" in cache._cache


class TestGlobalCache:
    """Tests for global cache singleton."""
    
    def setup_method(self):
        """Reset global cache before each test."""
        reset_rails_cache()
    
    def teardown_method(self):
        """Clean up after each test."""
        reset_rails_cache()
    
    def test_get_rails_cache_singleton(self):
        """Test that get_rails_cache returns same instance."""
        cache1 = get_rails_cache()
        cache2 = get_rails_cache()
        
        assert cache1 is cache2
    
    def test_get_rails_cache_with_options(self):
        """Test first call sets options."""
        cache = get_rails_cache(ttl_seconds=120.0, max_size=25)
        
        assert cache._ttl_seconds == 120.0
        assert cache._max_size == 25
    
    def test_clear_rails_cache(self):
        """Test clearing the global cache."""
        cache = get_rails_cache()
        
        # Add a dummy entry
        cache._cache["test"] = CacheEntry(
            rails=Mock(), config=Mock(),
            created_at=time.time(), last_accessed=time.time(),
            config_path=Path("/tmp"), config_hash="a",
        )
        
        clear_rails_cache()
        
        assert cache.size == 0
    
    def test_reset_rails_cache(self):
        """Test resetting the global cache."""
        cache1 = get_rails_cache()
        reset_rails_cache()
        cache2 = get_rails_cache()
        
        assert cache1 is not cache2


class TestComputeConfigHash:
    """Tests for _compute_config_hash function."""
    
    def test_hash_empty_dir(self, tmp_path):
        """Test hash of empty directory."""
        hash1 = _compute_config_hash(tmp_path)
        hash2 = _compute_config_hash(tmp_path)
        
        assert hash1 == hash2
    
    def test_hash_with_files(self, tmp_path):
        """Test hash includes file content."""
        (tmp_path / "config.yml").write_text("content1")
        hash1 = _compute_config_hash(tmp_path)
        
        (tmp_path / "config.yml").write_text("content2")
        hash2 = _compute_config_hash(tmp_path)
        
        assert hash1 != hash2
    
    def test_hash_ignores_unrelated_files(self, tmp_path):
        """Test hash only includes relevant files."""
        (tmp_path / "config.yml").write_text("content")
        hash1 = _compute_config_hash(tmp_path)
        
        # Add unrelated file
        (tmp_path / "readme.txt").write_text("readme")
        hash2 = _compute_config_hash(tmp_path)
        
        assert hash1 == hash2
    
    def test_hash_includes_subdirs(self, tmp_path):
        """Test hash includes rails subdirectory."""
        rails_dir = tmp_path / "rails"
        rails_dir.mkdir()
        
        hash1 = _compute_config_hash(tmp_path)
        
        (rails_dir / "input.co").write_text("flow content")
        hash2 = _compute_config_hash(tmp_path)
        
        assert hash1 != hash2
