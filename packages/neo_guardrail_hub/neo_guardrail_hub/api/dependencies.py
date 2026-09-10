"""
FastAPI dependency injection for Neo Guardrail Hub API.

Provides the OrchestratorManager that creates and caches orchestrator
instances. When a request supplies inline config, a temporary orchestrator
is built from that config. Otherwise the server's default orchestrator
(loaded from configs/ on disk) is used.
"""

import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.config import ConfigLoader
from ..core.models import GuardrailContext
from ..core.orchestrator import NeoGuardrailOrchestrator

logger = logging.getLogger(__name__)


class OrchestratorManager:
    """Manages NeoGuardrailOrchestrator instances.

    - Keeps a default orchestrator initialised from on-disk YAML configs.
    - For requests that include inline config, creates (and caches by
      config hash) ad-hoc orchestrators so the same config doesn't
      trigger repeated model loads.
    """

    def __init__(
        self,
        config_path: str = "./configs",
        models_dir: Optional[str] = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.models_dir = models_dir

        # Default orchestrator — loaded from on-disk YAML
        self._default_orchestrator: Optional[NeoGuardrailOrchestrator] = None

        # Cache: config_hash -> (orchestrator, last_used_timestamp)
        self._inline_cache: Dict[str, tuple[NeoGuardrailOrchestrator, float]] = {}
        self._cache_max_size: int = 20
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create and warm up the default orchestrator."""
        if self._initialized:
            return

        logger.info("Initializing default orchestrator from %s", self.config_path)
        self._default_orchestrator = NeoGuardrailOrchestrator(
            config_path=str(self.config_path),
            models_dir=self.models_dir,
        )
        try:
            await self._default_orchestrator.initialize(agent_id="default")
            logger.info("Default orchestrator initialized successfully")
        except Exception as e:
            logger.warning("Failed to pre-initialize default orchestrator: %s", e)

        self._initialized = True

    async def shutdown(self) -> None:
        """Clean up all orchestrator instances."""
        if self._default_orchestrator:
            await self._default_orchestrator.cleanup()
        for orch, _ in self._inline_cache.values():
            try:
                await orch.cleanup()
            except Exception:
                pass
        self._inline_cache.clear()
        self._initialized = False

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._default_orchestrator is not None

    @property
    def default_orchestrator(self) -> Optional[NeoGuardrailOrchestrator]:
        return self._default_orchestrator

    async def get_orchestrator(
        self,
        inline_config: Optional[Dict[str, Any]] = None,
        agent_id: str = "default",
    ) -> NeoGuardrailOrchestrator:
        """Return an orchestrator for the given request.

        If *inline_config* is provided the manager hashes it and either
        returns a cached orchestrator or creates a fresh one.  Otherwise
        the default (disk-config) orchestrator is returned.
        """
        if inline_config is not None:
            return await self._get_or_create_inline(inline_config, agent_id)

        if self._default_orchestrator is None:
            await self.initialize()

        assert self._default_orchestrator is not None
        return self._default_orchestrator

    def reload_config(self, agent_id: Optional[str] = None) -> None:
        """Hot-reload on-disk config for the default orchestrator."""
        if self._default_orchestrator:
            self._default_orchestrator.reload_config(agent_id)
            logger.info("Config reloaded for agent_id=%s", agent_id or "all")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_or_create_inline(
        self,
        config: Dict[str, Any],
        agent_id: str,
    ) -> NeoGuardrailOrchestrator:
        """Get or create an orchestrator from an inline config dict."""
        config_hash = self._hash_config(config)

        if config_hash in self._inline_cache:
            orch, _ = self._inline_cache[config_hash]
            self._inline_cache[config_hash] = (orch, time.time())
            return orch

        # Evict oldest entry if cache is full
        if len(self._inline_cache) >= self._cache_max_size:
            oldest_key = min(self._inline_cache, key=lambda k: self._inline_cache[k][1])
            old_orch, _ = self._inline_cache.pop(oldest_key)
            try:
                await old_orch.cleanup()
            except Exception:
                pass

        # Build a new orchestrator driven by the supplied dict.
        # We still point config_path at the server's configs/ so that
        # the ConfigLoader can fall back to defaults, but we inject the
        # supplied config as a runtime override.
        orch = NeoGuardrailOrchestrator(
            config_path=str(self.config_path),
            models_dir=self.models_dir,
        )

        # Inject the inline config into the orchestrator's cache so
        # _get_config(agent_id) returns it directly.
        orch._config_cache[agent_id] = config

        try:
            await orch.initialize(agent_id=agent_id)
        except Exception as e:
            logger.warning("Failed to initialize inline orchestrator: %s", e)

        self._inline_cache[config_hash] = (orch, time.time())
        return orch

    @staticmethod
    def _hash_config(config: Dict[str, Any]) -> str:
        """Deterministic hash of a config dict."""
        raw = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Utility: build GuardrailContext from request fields
    # ------------------------------------------------------------------

    @staticmethod
    def build_context(
        agent_id: str = "default",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[list] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[GuardrailContext]:
        """Build a GuardrailContext from common request fields.

        Returns None when there is no meaningful context to attach.
        """
        has_context = any([session_id, user_id, metadata, conversation_history, extra_metadata])
        if not has_context:
            return None

        merged_meta: Dict[str, Any] = {}
        if metadata:
            merged_meta.update(metadata)
        if extra_metadata:
            merged_meta.update(extra_metadata)

        ctx = GuardrailContext(
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            metadata=merged_meta,
        )
        if conversation_history:
            for msg in conversation_history:
                ctx.add_message(role=msg.get("role", "user"), content=msg.get("content", ""))

        return ctx

    @staticmethod
    def generate_request_id() -> str:
        """Generate a short, unique request ID."""
        return f"req_{uuid.uuid4().hex[:12]}"
