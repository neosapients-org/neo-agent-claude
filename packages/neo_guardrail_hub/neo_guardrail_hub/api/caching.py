"""
NeMo Rails Caching Module for Neo Guardrail Hub.

This module provides caching for NeMo LLMRails instances to avoid
expensive re-initialization. NeMo Rails are heavy objects that load
LLM configurations, Colang flows, and prompts.

Caching Strategy:
- Singleton cache per process
- Key-based lookup by agent_id or config path
- TTL-based expiration (optional)
- Thread-safe and async-safe access

Usage:
    from neo_guardrail_hub.api.caching import get_rails_cache, clear_rails_cache
    
    # Get the global cache instance
    cache = get_rails_cache()
    
    # Get or create Rails for an agent
    rails = await cache.get_or_create(
        key="wealth_advisor",
        config_path="./neo_configs/wealth_advisor"
    )
    
    # Clear all cached Rails
    clear_rails_cache()
"""

import asyncio
import hashlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING
from weakref import WeakValueDictionary

from ..utils.logging import get_logger

if TYPE_CHECKING:
    from nemoguardrails import LLMRails, RailsConfig

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Cache entry for a Rails instance."""
    rails: Any  # LLMRails
    config: Any  # RailsConfig
    created_at: float
    last_accessed: float
    config_path: Path
    config_hash: str
    
    def is_expired(self, ttl_seconds: Optional[float]) -> bool:
        """Check if this entry has expired."""
        if ttl_seconds is None:
            return False
        return (time.time() - self.last_accessed) > ttl_seconds
    
    def is_stale(self, config_path: Path) -> bool:
        """Check if the config files have changed."""
        current_hash = _compute_config_hash(config_path)
        return current_hash != self.config_hash


class RailsCache:
    """
    Cache for NeMo LLMRails instances.
    
    This cache provides efficient reuse of Rails instances across
    multiple guardrail checks. It supports:
    - Key-based lookup (by agent_id or custom key)
    - TTL-based expiration
    - Automatic staleness detection when config files change
    - Thread-safe access
    
    Example:
        cache = RailsCache()
        
        # Get or create Rails
        rails = await cache.get_or_create(
            key="my_agent",
            config_path="./neo_configs/my_agent"
        )
        
        # Use Rails
        response = await rails.generate_async(prompt="Hello")
        
        # Clear specific entry
        cache.remove("my_agent")
        
        # Clear all
        cache.clear()
    """
    
    def __init__(
        self,
        ttl_seconds: Optional[float] = None,
        max_size: int = 100,
        check_staleness: bool = True,
    ):
        """
        Initialize the Rails cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries in seconds.
                        None means no expiration.
            max_size: Maximum number of entries to cache.
            check_staleness: If True, check if config files changed
                            before returning cached entry.
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        self._ttl_seconds = ttl_seconds
        self._max_size = max_size
        self._check_staleness = check_staleness
        
        # Stats
        self._hits = 0
        self._misses = 0
    
    async def get_or_create(
        self,
        key: str,
        config_path: Path,
        force_reload: bool = False,
    ) -> "LLMRails":
        """
        Get Rails from cache or create new instance.
        
        Args:
            key: Cache key (typically agent_id)
            config_path: Path to NeMo config directory
            force_reload: If True, always create new instance
            
        Returns:
            LLMRails instance
            
        Raises:
            ImportError: If NeMo Guardrails is not installed
        """
        config_path = Path(config_path)
        
        async with self._async_lock:
            # Check if we have a valid cached entry
            if not force_reload and key in self._cache:
                entry = self._cache[key]
                
                # Check expiration
                if entry.is_expired(self._ttl_seconds):
                    logger.debug("cache_entry_expired", key=key)
                    del self._cache[key]
                elif self._check_staleness and entry.is_stale(config_path):
                    logger.debug("cache_entry_stale", key=key)
                    del self._cache[key]
                else:
                    # Valid cache hit
                    entry.last_accessed = time.time()
                    self._hits += 1
                    logger.debug("cache_hit", key=key)
                    return entry.rails
            
            # Cache miss - create new Rails
            self._misses += 1
            logger.debug("cache_miss", key=key)
            
            # Ensure we don't exceed max size
            self._evict_if_needed()
            
            # Create new Rails instance
            rails, config = await self._create_rails(config_path)
            
            # Store in cache
            now = time.time()
            self._cache[key] = CacheEntry(
                rails=rails,
                config=config,
                created_at=now,
                last_accessed=now,
                config_path=config_path,
                config_hash=_compute_config_hash(config_path),
            )
            
            logger.info(
                "rails_cached",
                key=key,
                config_path=str(config_path),
            )
            
            return rails
    
    def get_or_create_sync(
        self,
        key: str,
        config_path: Path,
        force_reload: bool = False,
    ) -> "LLMRails":
        """Synchronous version of get_or_create."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.get_or_create(key, config_path, force_reload)
        )
    
    async def _create_rails(self, config_path: Path) -> tuple["LLMRails", "RailsConfig"]:
        """Create a new LLMRails instance."""
        try:
            from nemoguardrails import LLMRails, RailsConfig
        except ImportError:
            raise ImportError(
                "NeMo Guardrails not installed. "
                "Install with: pip install neo-guardrail-hub[nemo]"
            )
        
        # Load config
        config = RailsConfig.from_path(str(config_path))
        
        # Create Rails instance
        rails = LLMRails(config)
        
        return rails, config
    
    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is full."""
        with self._lock:
            while len(self._cache) >= self._max_size:
                # Find the oldest entry
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].last_accessed
                )
                logger.debug("evicting_cache_entry", key=oldest_key)
                del self._cache[oldest_key]
    
    def get(self, key: str) -> Optional["LLMRails"]:
        """Get Rails from cache without creating."""
        with self._lock:
            entry = self._cache.get(key)
            if entry and not entry.is_expired(self._ttl_seconds):
                entry.last_accessed = time.time()
                return entry.rails
            return None
    
    def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            return key in self._cache
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug("cache_entry_removed", key=key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            logger.info("cache_cleared", entries_removed=count)
    
    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = (
                self._hits / (self._hits + self._misses)
                if (self._hits + self._misses) > 0
                else 0.0
            )
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "max_size": self._max_size,
                "ttl_seconds": self._ttl_seconds,
            }
    
    def list_keys(self) -> list[str]:
        """List all cache keys."""
        with self._lock:
            return list(self._cache.keys())


# Global cache instance (singleton pattern)
_global_cache: Optional[RailsCache] = None
_global_cache_lock = threading.Lock()


def get_rails_cache(
    ttl_seconds: Optional[float] = None,
    max_size: int = 100,
    check_staleness: bool = True,
) -> RailsCache:
    """
    Get the global Rails cache instance.
    
    This function implements a singleton pattern - the first call
    creates the cache with the specified settings, subsequent calls
    return the same instance.
    
    Args:
        ttl_seconds: Time-to-live for cache entries
        max_size: Maximum cache size
        check_staleness: Whether to check for config file changes
        
    Returns:
        Global RailsCache instance
        
    Example:
        cache = get_rails_cache()
        rails = await cache.get_or_create("agent", "./configs")
    """
    global _global_cache
    
    with _global_cache_lock:
        if _global_cache is None:
            _global_cache = RailsCache(
                ttl_seconds=ttl_seconds,
                max_size=max_size,
                check_staleness=check_staleness,
            )
            logger.info(
                "global_cache_created",
                max_size=max_size,
                ttl_seconds=ttl_seconds,
            )
        return _global_cache


def clear_rails_cache() -> None:
    """
    Clear the global Rails cache.
    
    This frees all cached LLMRails instances. Use this when:
    - Configuration files have changed
    - You want to free memory
    - Testing
    """
    global _global_cache
    
    with _global_cache_lock:
        if _global_cache is not None:
            _global_cache.clear()
            logger.info("global_cache_cleared")


def reset_rails_cache() -> None:
    """
    Reset the global Rails cache to None.
    
    This completely removes the cache instance. The next call to
    get_rails_cache() will create a new instance with potentially
    different settings.
    """
    global _global_cache
    
    with _global_cache_lock:
        if _global_cache is not None:
            _global_cache.clear()
        _global_cache = None
        logger.info("global_cache_reset")


def _compute_config_hash(config_path: Path) -> str:
    """Compute a hash of the config files for staleness checking."""
    hasher = hashlib.sha256()
    
    # Hash all relevant files
    for pattern in ["*.yml", "*.yaml", "*.co", "*.py"]:
        for file in sorted(config_path.glob(pattern)):
            try:
                hasher.update(file.read_bytes())
                hasher.update(file.name.encode())
            except Exception:
                pass
    
    # Also hash subdirectories
    rails_dir = config_path / "rails"
    if rails_dir.exists():
        for file in sorted(rails_dir.glob("*")):
            try:
                hasher.update(file.read_bytes())
            except Exception:
                pass
    
    return hasher.hexdigest()[:16]
