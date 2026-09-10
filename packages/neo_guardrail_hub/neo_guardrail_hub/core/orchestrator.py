"""Main orchestrator for Neo Guardrail Hub.

This module provides the primary entry point for using guardrails,
coordinating configuration loading, guardrail execution, and result aggregation.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import ConfigLoader
from .exceptions import ConfigurationError, GuardrailError
from .executor import ParallelExecutor, ResultAggregator
from .interfaces import BaseGuardrail
from .models import (
    ActionOnFail,
    AggregatedResult,
    ExecutionMode,
    GuardrailContext,
    GuardrailLayer,
    GuardrailResult,
)
from .registry import GuardrailRegistry


class NeoGuardrailOrchestrator:
    """Main entry point for Neo Guardrail Hub.

    Orchestrates the entire guardrail pipeline including configuration,
    guardrail instantiation, parallel execution, and result aggregation.

    Example:
        orchestrator = NeoGuardrailOrchestrator(config_path="./configs")

        # Async usage
        result = await orchestrator.guard_input("user message", agent_id="my_agent")

        # Sync usage
        result = orchestrator.guard_input_sync("user message", agent_id="my_agent")
    """

    def __init__(
        self,
        config_path: Union[str, Path] = "./configs",
        auto_discover: bool = True,
        models_dir: Optional[str] = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            config_path: Path to the configuration directory
            auto_discover: Whether to auto-discover guardrails
            models_dir: Optional path to directory containing local models for LLM Guard guardrails
        """
        self.config_path = Path(config_path)
        self.config_loader = ConfigLoader(config_path)
        self.registry = GuardrailRegistry(auto_discover=auto_discover)
        self.executor = ParallelExecutor()
        self.aggregator = ResultAggregator()
        self.models_dir = models_dir

        self._config_cache: Dict[str, Dict[str, Any]] = {}
        self._initialized = False

    async def initialize(self, agent_id: str = "default") -> None:
        """Initialize the orchestrator and all providers.
        
        Args:
            agent_id: Agent identifier to load config for provider setup
        """
        if self._initialized:
            return

        # Load config for this agent to detect which providers are needed
        config = self._get_config(agent_id)
        
        # Create LLM Guard provider with models_dir if specified
        if self.models_dir:
            llm_guard_provider = self.registry.create_provider("llm_guard", {
                "models_dir": self.models_dir
            })
            if llm_guard_provider:
                await llm_guard_provider.initialize()
        
        # Create NeMo provider if NeMo config is present and enabled
        nemo_config = config.get("nemo", {})
        if nemo_config.get("enabled", False):
            nemo_provider = self.registry.create_provider("nemo", {
                "config_path": str(self.config_path),
                "nemo_config": nemo_config,
                "agent_id": agent_id,  # Pass agent_id so NeMo loads correct config
            })
            if nemo_provider:
                await nemo_provider.initialize()

        # Initialize all other registered providers
        for provider_name in self.registry.list_providers():
            provider = self.registry.get_provider(provider_name)
            if provider and not getattr(provider, '_initialized', False):
                await provider.initialize()

        self._initialized = True

    async def cleanup(self) -> None:
        """Clean up resources used by the orchestrator."""
        for provider_name in self.registry.list_providers():
            provider = self.registry.get_provider(provider_name)
            if provider:
                await provider.cleanup()

        self._config_cache.clear()
        self._initialized = False

    def _get_config(self, agent_id: str) -> Dict[str, Any]:
        """Get configuration for an agent, with caching."""
        if agent_id not in self._config_cache:
            self._config_cache[agent_id] = self.config_loader.load(agent_id=agent_id)
        return self._config_cache[agent_id]

    def _get_guardrails_for_layer(
        self, config: Dict[str, Any], layer: GuardrailLayer
    ) -> List[BaseGuardrail]:
        """Get instantiated guardrails for a layer based on config.
        
        Handles ALL providers uniformly from guardrails.{layer}.checks:
        - NeMo: Gets from NeMo provider (requires provider instance in config)
        - Others: Gets from registry (auto-discovers via registry.get())
        """
        import structlog
        logger = structlog.get_logger(__name__)
        
        layer_config = self.config_loader.get_layer_config(config, layer.value)

        if not layer_config.enabled:
            return []

        guardrails: List[BaseGuardrail] = []
        default_on_fail = config.get("defaults", {}).get("on_fail", "block")

        for check_config in layer_config.checks:
            try:
                # Get provider name (defaults to 'llm_guard' for backward compat)
                provider_name = check_config.provider if hasattr(check_config, 'provider') else check_config.get("provider", "llm_guard")
                
                # Prepare config for guardrail
                guardrail_config = {
                    "threshold": check_config.threshold,
                    "on_fail": check_config.on_fail.value,
                    **check_config.config,
                }
                
                if provider_name == "nemo":
                    # NeMo guardrails need provider instance
                    nemo_provider = self.registry.get_provider("nemo")
                    
                    if nemo_provider:
                        guardrail = nemo_provider.get_guardrail(check_config.type, guardrail_config)
                        
                        if guardrail:
                            guardrails.append(guardrail)
                        else:
                            logger.warning(
                                "nemo_guardrail_not_found",
                                guardrail_type=check_config.type,
                            )
                    else:
                        logger.warning(
                            "nemo_provider_not_found",
                            message="NeMo provider not registered. Ensure NeMo is enabled in config."
                        )
                else:
                    # Other guardrails: use registry.get() which auto-discovers
                    # This works for llm_guard, custom, and any other registered providers
                    
                    # Pass models_dir to guardrail config if available
                    if self.models_dir:
                        guardrail_config["models_dir"] = self.models_dir
                    
                    guardrail = self.registry.get(check_config.type, guardrail_config)
                    guardrails.append(guardrail)
                    
            except Exception as e:
                # Log error but continue with other guardrails
                logger.warning(
                    "failed_to_load_guardrail",
                    guardrail_type=check_config.type,
                    provider=provider_name,
                    error=str(e),
                )

        return guardrails

    async def guard_input(
        self,
        text: str,
        agent_id: str = "default",
        context: Optional[GuardrailContext] = None,
    ) -> AggregatedResult:
        """Validate input before LLM processing.

        Args:
            text: User input text to validate
            agent_id: Agent identifier for config lookup
            context: Optional context for stateful checks

        Returns:
            AggregatedResult with pass/fail status and details
        """
        return await self._guard_layer(
            text=text,
            layer=GuardrailLayer.INPUT,
            agent_id=agent_id,
            context=context,
        )

    async def guard_context(
        self,
        text: str,
        agent_id: str = "default",
        context: Optional[GuardrailContext] = None,
    ) -> AggregatedResult:
        """Validate context/conversation flow.

        Args:
            text: Current message text
            agent_id: Agent identifier for config lookup
            context: Context with conversation history

        Returns:
            AggregatedResult with pass/fail status and details
        """
        return await self._guard_layer(
            text=text,
            layer=GuardrailLayer.CONTEXT,
            agent_id=agent_id,
            context=context,
        )

    async def guard_output(
        self,
        text: str,
        agent_id: str = "default",
        context: Optional[GuardrailContext] = None,
    ) -> AggregatedResult:
        """Validate output before returning to user.

        Args:
            text: LLM output text to validate
            agent_id: Agent identifier for config lookup
            context: Optional context for stateful checks

        Returns:
            AggregatedResult with pass/fail status and details
        """
        return await self._guard_layer(
            text=text,
            layer=GuardrailLayer.OUTPUT,
            agent_id=agent_id,
            context=context,
        )

    async def guard_all(
        self,
        input_text: str,
        output_text: str,
        agent_id: str = "default",
        context: Optional[GuardrailContext] = None,
    ) -> Dict[str, AggregatedResult]:
        """Run all three layers of guardrails.

        Args:
            input_text: User input to validate
            output_text: LLM output to validate
            agent_id: Agent identifier for config lookup
            context: Optional context for stateful checks

        Returns:
            Dictionary with 'input', 'context', and 'output' results
        """
        # Run input and output in parallel (context depends on input)
        input_result = await self.guard_input(
            text=input_text, agent_id=agent_id, context=context
        )

        results = {"input": input_result}

        # Only continue if input passed
        if input_result.passed:
            context_result = await self.guard_context(
                text=input_text, agent_id=agent_id, context=context
            )
            results["context"] = context_result

            if context_result.passed:
                output_result = await self.guard_output(
                    text=output_text, agent_id=agent_id, context=context
                )
                results["output"] = output_result
            else:
                results["output"] = self._create_skipped_result(GuardrailLayer.OUTPUT)
        else:
            results["context"] = self._create_skipped_result(GuardrailLayer.CONTEXT)
            results["output"] = self._create_skipped_result(GuardrailLayer.OUTPUT)

        return results

    async def _guard_layer(
        self,
        text: str,
        layer: GuardrailLayer,
        agent_id: str,
        context: Optional[GuardrailContext] = None,
    ) -> AggregatedResult:
        """Execute guardrails for a specific layer.

        Args:
            text: Text to check
            layer: Which layer to execute
            agent_id: Agent identifier
            context: Optional context

        Returns:
            AggregatedResult for the layer
        """
        config = self._get_config(agent_id)

        # Check if guardrails are globally enabled
        if not config.get("enabled", True):
            return self._create_disabled_result(layer, text)

        # Get guardrails for this layer (all providers including NeMo and LLM Guard)
        guardrails = self._get_guardrails_for_layer(config, layer)

        if not guardrails:
            return AggregatedResult(
                passed=True,
                layer=layer,
                results=[],
                action_taken=ActionOnFail.BLOCK,
                final_text=text,
                total_latency_ms=0.0,
            )

        # Get execution settings
        layer_config = self.config_loader.get_layer_config(config, layer.value)
        execution_mode = layer_config.execution
        timeout_ms = layer_config.timeout_ms

        # Execute guardrails
        if execution_mode == ExecutionMode.PARALLEL:
            results = await self.executor.execute_parallel(
                guardrails=guardrails,
                text=text,
                context=context,
                timeout_ms=timeout_ms,
            )
        else:
            results = await self.executor.execute_sequential(
                guardrails=guardrails,
                text=text,
                context=context,
                stop_on_fail=True,
                timeout_ms=timeout_ms,
            )

        # Aggregate results
        default_action = ActionOnFail(config.get("defaults", {}).get("on_fail", "block"))
        return self.aggregator.aggregate(
            results=results,
            layer=layer,
            original_text=text,
            default_action=default_action,
        )

    def _create_disabled_result(
        self, layer: GuardrailLayer, text: str
    ) -> AggregatedResult:
        """Create a result for when guardrails are disabled."""
        return AggregatedResult(
            passed=True,
            layer=layer,
            results=[],
            action_taken=ActionOnFail.BLOCK,
            final_text=text,
            total_latency_ms=0.0,
        )

    def _create_skipped_result(self, layer: GuardrailLayer) -> AggregatedResult:
        """Create a result for skipped layers."""
        return AggregatedResult(
            passed=False,
            layer=layer,
            results=[
                GuardrailResult(
                    passed=False,
                    guardrail_name="skipped",
                    layer=layer,
                    message="Layer skipped due to previous failure",
                )
            ],
            action_taken=ActionOnFail.BLOCK,
            final_text=None,
            total_latency_ms=0.0,
        )

    # Sync wrappers for non-async usage

    def guard_input_sync(
        self,
        text: str,
        agent_id: str = "default",
        context: Optional[GuardrailContext] = None,
    ) -> AggregatedResult:
        """Synchronous wrapper for guard_input."""
        return asyncio.run(self.guard_input(text, agent_id, context))

    def guard_context_sync(
        self,
        text: str,
        agent_id: str = "default",
        context: Optional[GuardrailContext] = None,
    ) -> AggregatedResult:
        """Synchronous wrapper for guard_context."""
        return asyncio.run(self.guard_context(text, agent_id, context))

    def guard_output_sync(
        self,
        text: str,
        agent_id: str = "default",
        context: Optional[GuardrailContext] = None,
    ) -> AggregatedResult:
        """Synchronous wrapper for guard_output."""
        return asyncio.run(self.guard_output(text, agent_id, context))

    def guard_all_sync(
        self,
        input_text: str,
        output_text: str,
        agent_id: str = "default",
        context: Optional[GuardrailContext] = None,
    ) -> Dict[str, AggregatedResult]:
        """Synchronous wrapper for guard_all."""
        return asyncio.run(self.guard_all(input_text, output_text, agent_id, context))

    def reload_config(self, agent_id: Optional[str] = None) -> None:
        """Reload configuration from disk.

        Args:
            agent_id: Specific agent to reload, or None for all
        """
        if agent_id:
            self._config_cache.pop(agent_id, None)
        else:
            self._config_cache.clear()
        self.config_loader.clear_cache()

    def register_guardrail(
        self, name: str, guardrail_class: type, override: bool = False
    ) -> None:
        """Register a custom guardrail class.

        Args:
            name: Name for the guardrail
            guardrail_class: The guardrail class to register
            override: Whether to override existing registration
        """
        self.registry.register(name, guardrail_class, override=override)

    def register_provider(self, provider: Any) -> None:
        """Register an external provider.

        Args:
            provider: Provider instance to register
        """
        self.registry.register_provider(provider)
