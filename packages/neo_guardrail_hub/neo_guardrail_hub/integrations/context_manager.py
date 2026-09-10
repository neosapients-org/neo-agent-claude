"""Context manager for guardrail sessions.

This module provides the GuardSession context manager for
managing guardrail state across multiple checks.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..core.models import AggregatedResult, GuardrailContext, GuardrailLayer
from ..core.orchestrator import NeoGuardrailOrchestrator


class GuardSession:
    """Context manager for managing guardrail sessions.

    Maintains state across multiple guardrail checks, including
    conversation history and session context.

    Example:
        async with GuardSession(agent_id="my_agent") as session:
            input_result = await session.guard_input(user_message)
            if not input_result.passed:
                return input_result.message

            response = await llm.generate(user_message)

            output_result = await session.guard_output(response)
            return output_result.final_text
    """

    def __init__(
        self,
        agent_id: str = "default",
        config_path: Union[str, Path] = "./configs",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        models_dir: Optional[str] = None,
    ) -> None:
        """Initialize the guard session.

        Args:
            agent_id: Agent identifier for config lookup
            config_path: Path to configuration directory
            session_id: Optional session identifier
            user_id: Optional user identifier
            metadata: Optional metadata for context
            models_dir: Optional path to directory containing local models for LLM Guard guardrails
        """
        self.agent_id = agent_id
        self.config_path = Path(config_path)
        self.models_dir = models_dir
        self._orchestrator: Optional[NeoGuardrailOrchestrator] = None

        # Build context
        self.context = GuardrailContext(
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {},
        )

        # Track results
        self._results: List[AggregatedResult] = []

    async def __aenter__(self) -> "GuardSession":
        """Enter the session context."""
        self._orchestrator = NeoGuardrailOrchestrator(
            config_path=self.config_path,
            models_dir=self.models_dir
        )
        await self._orchestrator.initialize(agent_id=self.agent_id)
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """Exit the session context."""
        if self._orchestrator:
            await self._orchestrator.cleanup()
            self._orchestrator = None

    async def guard_input(
        self,
        text: str,
        add_to_history: bool = True,
    ) -> AggregatedResult:
        """Validate input text.

        Args:
            text: User input to validate
            add_to_history: Whether to add to conversation history

        Returns:
            AggregatedResult with validation result
        """
        if self._orchestrator is None:
            raise RuntimeError("Session not initialized. Use 'async with' context.")

        if add_to_history:
            self.context.add_message("user", text)

        result = await self._orchestrator.guard_input(
            text=text,
            agent_id=self.agent_id,
            context=self.context,
        )

        self._results.append(result)
        return result

    async def guard_context(
        self,
        text: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> AggregatedResult:
        """Validate context/conversation flow.

        Args:
            text: Current message text
            history: Optional conversation history override

        Returns:
            AggregatedResult with validation result
        """
        if self._orchestrator is None:
            raise RuntimeError("Session not initialized. Use 'async with' context.")

        # Use provided history or session history
        if history is not None:
            self.context.conversation_history = history

        result = await self._orchestrator.guard_context(
            text=text,
            agent_id=self.agent_id,
            context=self.context,
        )

        self._results.append(result)
        return result

    async def guard_output(
        self,
        text: str,
        add_to_history: bool = True,
    ) -> AggregatedResult:
        """Validate output text.

        Args:
            text: LLM output to validate
            add_to_history: Whether to add to conversation history

        Returns:
            AggregatedResult with validation result
        """
        if self._orchestrator is None:
            raise RuntimeError("Session not initialized. Use 'async with' context.")

        result = await self._orchestrator.guard_output(
            text=text,
            agent_id=self.agent_id,
            context=self.context,
        )

        if add_to_history:
            # Add the final text (sanitized or original)
            final_text = result.final_text or text
            self.context.add_message("assistant", final_text)

        self._results.append(result)
        return result

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.context.conversation_history.copy()

    def get_results(self) -> List[AggregatedResult]:
        """Get all guardrail results from this session."""
        return self._results.copy()

    def get_failed_checks(self) -> List[AggregatedResult]:
        """Get only failed guardrail results."""
        return [r for r in self._results if not r.passed]

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.context.conversation_history.clear()

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the session context."""
        self.context.metadata[key] = value


# Sync version of GuardSession
class GuardSessionSync:
    """Synchronous version of GuardSession.

    For use in non-async contexts.
    """

    def __init__(
        self,
        agent_id: str = "default",
        config_path: Union[str, Path] = "./configs",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the sync guard session."""
        self.agent_id = agent_id
        self.config_path = Path(config_path)
        self._orchestrator: Optional[NeoGuardrailOrchestrator] = None

        self.context = GuardrailContext(
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {},
        )

        self._results: List[AggregatedResult] = []

    def __enter__(self) -> "GuardSessionSync":
        """Enter the session context."""
        self._orchestrator = NeoGuardrailOrchestrator(config_path=self.config_path)
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """Exit the session context."""
        self._orchestrator = None

    def guard_input(self, text: str) -> AggregatedResult:
        """Validate input text synchronously."""
        if self._orchestrator is None:
            raise RuntimeError("Session not initialized. Use 'with' context.")

        self.context.add_message("user", text)

        result = self._orchestrator.guard_input_sync(
            text=text,
            agent_id=self.agent_id,
            context=self.context,
        )

        self._results.append(result)
        return result

    def guard_output(self, text: str) -> AggregatedResult:
        """Validate output text synchronously."""
        if self._orchestrator is None:
            raise RuntimeError("Session not initialized. Use 'with' context.")

        result = self._orchestrator.guard_output_sync(
            text=text,
            agent_id=self.agent_id,
            context=self.context,
        )

        final_text = result.final_text or text
        self.context.add_message("assistant", final_text)

        self._results.append(result)
        return result
