"""VIC-style Memory Agent — Full MCP Integration Example.

Demonstrates ALL memory capabilities from the VIC Memory Operations Reference:
- Investor name resolution (session-tracked)
- Multi-category retrieval with investor scoping
- Exchange storage with LLM fact extraction + salience gating + dedup
- Exchange buffering with threshold and idle flush
- Direct fact and preference storage
- Vector search with category and investor filtering
- Memory CRUD (get, get_all, update, delete, delete_all)
- Scoped storage (private / team / org pools)
- History and audit trail
- Salience scoring
- TTL-based cleanup

Usage:
    python examples/vic_mcp_agent/agent.py

Requires:
    - MCP server running: python scripts/start_mcp_server.py
    - Qdrant running on port 6335
    - OPENAI_API_KEY set
    - pip install deepagents
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger("vic_mcp_agent")

# =====================================================================
# MCP Bridge — async-to-sync bridge for Deep Agent tool calls
# =====================================================================

class _MCPBridge:
    """Single-task async lifecycle bridge for MCP client.

    Runs an asyncio event loop in a background thread. All MCP operations
    (connect, call_tool, list_tools, close) are dispatched to this loop
    via a command queue, ensuring connect and close happen in the same
    AnyIO task (required by streamable-http transport).
    """

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._cmd_q: asyncio.Queue | None = None
        self._started = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._started.wait(timeout=30)

    def stop(self) -> None:
        if self._loop and self._cmd_q:
            fut = asyncio.run_coroutine_threadsafe(
                self._cmd_q.put(("close", None, None)), self._loop
            )
            fut.result(timeout=10)
        if self._thread:
            self._thread.join(timeout=10)

    def list_tools(self):
        return self._dispatch("list_tools", None)

    def call_tool(self, name: str, arguments: dict) -> dict:
        return self._dispatch("call_tool", {"name": name, "arguments": arguments})

    def read_resource(self, uri: str) -> str:
        return self._dispatch("read_resource", {"uri": uri})

    def _dispatch(self, cmd: str, payload: Any) -> Any:
        result_holder = {}

        def _go():
            inner_f = self._loop.create_future()
            asyncio.run_coroutine_threadsafe(
                self._cmd_q.put((cmd, payload, inner_f)), self._loop
            )
            result_holder["future"] = inner_f

        self._loop.call_soon_threadsafe(_go)
        # Wait for the future to complete
        import time
        for _ in range(600):  # 60s timeout
            time.sleep(0.1)
            f = result_holder.get("future")
            if f and f.done():
                return f.result()
        raise TimeoutError(f"MCP command {cmd} timed out")

    def _run_loop(self):
        asyncio.run(self._lifecycle())

    async def _lifecycle(self):
        from neomem_mcp.client import NeoMemMCPClient

        self._loop = asyncio.get_running_loop()
        self._cmd_q = asyncio.Queue()
        self._started.set()

        client = NeoMemMCPClient(host=self._host, port=self._port)
        await client.connect()
        logger.info("Connected to MCP server at http://%s:%s/mcp", self._host, self._port)

        try:
            while True:
                cmd, payload, future = await self._cmd_q.get()
                try:
                    if cmd == "close":
                        break
                    elif cmd == "list_tools":
                        result = await client.list_tools()
                        future.set_result(result)
                    elif cmd == "call_tool":
                        result = await client.call_tool(payload["name"], payload["arguments"])
                        future.set_result(result)
                    elif cmd == "read_resource":
                        result = await client.read_resource(payload["uri"])
                        future.set_result(result)
                except Exception as e:
                    if future and not future.done():
                        future.set_exception(e)
        finally:
            await client.close()
            logger.info("MCP client disconnected")


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class MemoryOperation:
    timestamp: str
    operation: str      # CREATE, READ, UPDATE, DELETE
    tool: str           # Short tool name (e.g. "store_exchange")
    args_summary: str
    result_summary: str
    raw_result: dict = field(default_factory=dict)


# =====================================================================
# VIC Memory Agent
# =====================================================================

class VICMemoryAgent:
    """Production-style memory agent mirroring VIC's memory operations.

    Implements all patterns from VIC_MEMORY_OPERATIONS_REFERENCE.md:
    - Session-tracked investor name resolution
    - Guaranteed store after every exchange
    - Buffering for low-priority exchanges
    - Multi-category retrieval with investor scoping
    - Direct fact/preference storage
    - Scoped memory (private/team/org pools)
    - History and audit trail
    """

    def __init__(
        self,
        user_id: str = "demo_rm",
        agent_id: str = "vic_l3",
        dept_id: str = "wealth",
        mcp_host: str | None = None,
        mcp_port: int | None = None,
        system_prompt: str | None = None,
        custom_extraction_prompt: str | None = None,
    ):
        self.user_id = user_id
        self.agent_id = agent_id
        self.dept_id = dept_id
        self.mcp_host = mcp_host or os.environ.get("NEOMEM_MCP_HOST", "localhost")
        self.mcp_port = mcp_port or int(os.environ.get("NEOMEM_MCP_PORT", "18432"))
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.custom_extraction_prompt = custom_extraction_prompt

        # Session state — mirrors VIC's VICContextManager
        self._session_investors: dict[str, str] = {}   # session_key → investor_name
        self._session_history: dict[str, list] = {}    # session_key → last N messages
        self._session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._max_history_turns = 4

        self._bridge: _MCPBridge | None = None
        self._agent = None
        self._ops: list[MemoryOperation] = []
        self._messages: list[dict[str, str]] = []

    # ── Connection ────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._bridge is not None

    def connect(self) -> int:
        self._bridge = _MCPBridge(host=self.mcp_host, port=self.mcp_port)
        self._bridge.start()
        tools = self._bridge.list_tools()
        tool_count = len(tools)
        logger.info("Connected to MCP %s:%s — %d tools",
                     self.mcp_host, self.mcp_port, tool_count)
        self._agent = self._create_agent()
        return tool_count

    def close(self) -> None:
        # Flush buffer on session end (VIC pattern §19)
        if self._bridge:
            try:
                self._call_mcp("neomem_flush_buffer", {
                    "user_id": self.user_id,
                    "session_id": self._session_id,
                    "agent_id": self.agent_id,
                })
            except Exception as e:
                logger.warning("Flush on close failed: %s", e)
            self._bridge.stop()
            self._bridge = None

    # ── Investor Resolution (VIC §12) ────────────────────────────────

    @property
    def current_investor(self) -> str | None:
        key = f"{self.user_id}:{self._session_id}"
        return self._session_investors.get(key)

    def _resolve_investor(self, query: str) -> str | None:
        """Simple investor name resolution from query text.

        In production VIC, this is an LLM call with session history.
        For the demo, we use regex-based extraction with session fallback.
        """
        key = f"{self.user_id}:{self._session_id}"

        # Check for explicit name mentions (Title Case words)
        # Patterns: "for Mohan", "about Priya", "Mohan's portfolio", client name mentions
        patterns = [
            r"(?:for|about|regarding|of|client)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'s\s+(?:portfolio|investment|holding|account|risk|preference|allocation|ESG|fund|SIP|profile|concern)",
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:wants|needs|prefers|is|has)",
        ]
        for pat in patterns:
            m = re.search(pat, query)
            if m:
                name = m.group(1).strip()
                # Skip common non-name words
                if name.lower() not in {"the", "this", "what", "how", "who", "list", "all", "my", "can", "do", "tell"}:
                    self._session_investors[key] = name
                    return name

        # Pronoun resolution — fall back to session-tracked investor
        pronouns = {"his", "her", "their", "he", "she", "they", "him", "them"}
        query_words = set(query.lower().split())
        if query_words & pronouns and key in self._session_investors:
            return self._session_investors[key]

        # General query — no specific investor
        return self._session_investors.get(key)

    def _add_to_session_history(self, role: str, content: str) -> None:
        """Maintain session conversation history (VIC §14, last 4 turns)."""
        key = f"{self.user_id}:{self._session_id}"
        if key not in self._session_history:
            self._session_history[key] = []
        # Truncate content to 500 chars
        self._session_history[key].append({
            "role": role,
            "content": content[:500],
        })
        # Keep last N*2 messages (4 turns = 8 messages)
        max_msgs = self._max_history_turns * 2
        if len(self._session_history[key]) > max_msgs:
            self._session_history[key] = self._session_history[key][-max_msgs:]

    # ── Chat ──────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> tuple[str, list[MemoryOperation]]:
        """Process a user message with full VIC-style memory pipeline.

        Pipeline (mirrors VIC §20):
        1. Resolve investor name from query
        2. Retrieve memory context (with investor scoping)
        3. Deep Agent generates response (with memory context as tools)
        4. Store exchange (guaranteed, programmatic)
        5. Update session history
        """
        if not self._agent:
            raise RuntimeError("Not connected. Call agent.connect() first.")

        ops_before = len(self._ops)

        # Step 1: Resolve investor
        investor = self._resolve_investor(user_message)

        # Step 2+3: Agent invocation (agent calls recall_context internally)
        messages = list(self._messages)
        messages.append({"role": "user", "content": user_message})

        result = self._agent.invoke({"messages": messages})
        response = self._extract_response(result)

        # Step 4: Guaranteed storage (VIC always stores)
        already_stored = any(
            op.tool == "store_exchange" for op in self._ops[ops_before:]
        )
        if not already_stored:
            try:
                store_args = {
                    "query": user_message,
                    "response": response,
                    "user_id": self.user_id,
                    "agent_id": self.agent_id,
                    "session_id": self._session_id,
                    "dept_id": self.dept_id,
                }
                if investor:
                    store_args["investor_name"] = investor
                if self.custom_extraction_prompt:
                    store_args["custom_extraction_prompt"] = self.custom_extraction_prompt
                self._call_mcp("neomem_store_exchange", store_args)
            except Exception as e:
                logger.warning("Auto-store failed: %s", e)

        # Step 5: Update session history
        self._add_to_session_history("user", user_message)
        self._add_to_session_history("assistant", response)

        # Save to message history
        self._messages.append({"role": "user", "content": user_message})
        self._messages.append({"role": "assistant", "content": response})

        new_ops = self._ops[ops_before:]
        return response, new_ops

    # ── Direct MCP calls ──────────────────────────────────────────────

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        if not self._bridge:
            raise RuntimeError("Not connected.")
        result = self._bridge.call_tool(tool_name, arguments)
        self._log_op(_classify_operation(tool_name),
                     tool_name.replace("neomem_", ""), arguments, result)
        return result

    def list_tools(self) -> list[str]:
        if not self._bridge:
            raise RuntimeError("Not connected.")
        tools = self._bridge.list_tools()
        return sorted(t.name for t in tools)

    def read_config(self) -> dict:
        if not self._bridge:
            raise RuntimeError("Not connected.")
        text = self._bridge.read_resource("neomem://config")
        return json.loads(text) if text else {}

    # ── Operations log ────────────────────────────────────────────────

    @property
    def operations(self) -> list[MemoryOperation]:
        return list(self._ops)

    def clear_operations(self) -> None:
        self._ops.clear()

    @property
    def messages(self) -> list[dict[str, str]]:
        return list(self._messages)

    def clear_history(self) -> None:
        self._messages.clear()
        self._session_history.clear()
        self._session_investors.clear()
        self._session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._agent = self._create_agent()

    # ── Internal ──────────────────────────────────────────────────────

    def _call_mcp(self, tool_name: str, arguments: dict) -> dict:
        result = self._bridge.call_tool(tool_name, arguments)
        self._log_op(_classify_operation(tool_name),
                     tool_name.replace("neomem_", ""), arguments, result)
        return result

    def _log_op(self, operation: str, tool: str, args: dict, result: dict) -> None:
        self._ops.append(MemoryOperation(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            operation=operation,
            tool=tool,
            args_summary=_summarize_args(args),
            result_summary=_summarize_result(result),
            raw_result=result,
        ))

    def _create_agent(self):
        from deepagents import create_deep_agent
        return create_deep_agent(
            model="openai:gpt-4o",
            tools=self._build_tools(),
            system_prompt=self.system_prompt,
        )

    def _build_tools(self) -> list:
        """Build all tool functions bound to this agent instance.

        Covers all VIC memory operations from the reference doc:
        - Core: store_conversation, recall_context, search_memories
        - Direct storage: store_fact, store_preference
        - Retrieval: get_memory, get_all_memories, count_memories
        - Update/Delete: update_memory, delete_memory, delete_all_memories
        - Buffering: buffer_conversation, flush_buffer
        - Analysis: score_salience
        - History: get_memory_history, get_all_history, get_history_stats
        - Lifecycle: cleanup_expired
        - Scoped: store_scoped, search_scoped, add_scoped, get_all_scoped
        """
        agent = self

        # ── Core Exchange Storage ─────────────────────────────────────

        def store_conversation(query: str, response: str) -> str:
            """Store a conversation exchange into long-term memory.

            Runs the FULL pipeline: LLM fact extraction → salience scoring →
            deduplication → storage. Extracts persona, preference, episodic,
            and procedural facts automatically.

            Args:
                query: The user/RM's message.
                response: The advisor's reply.

            Returns:
                JSON with facts_stored, facts_extracted, facts_skipped, investor_name.
            """
            args = {
                "query": query, "response": response,
                "user_id": agent.user_id, "agent_id": agent.agent_id,
                "session_id": agent._session_id, "dept_id": agent.dept_id,
            }
            investor = agent.current_investor
            if investor:
                args["investor_name"] = investor
            if agent.custom_extraction_prompt:
                args["custom_extraction_prompt"] = agent.custom_extraction_prompt
            return json.dumps(agent._call_mcp("neomem_store_exchange", args), indent=2)

        def recall_context(query: str) -> str:
            """Retrieve relevant memories for a query with investor scoping.

            Performs multi-category retrieval across persona, preference,
            episodic, and procedural memories. When an investor is tracked
            in session, uses investor-aware thresholds for better recall.

            Args:
                query: The user's latest question or topic.

            Returns:
                JSON with context (formatted text), results, by_category, investor_name.
            """
            args = {
                "query": query,
                "user_id": agent.user_id,
                "agent_id": agent.agent_id,
                "session_id": agent._session_id,
                "dept_id": agent.dept_id,
            }
            investor = agent.current_investor
            if investor:
                args["investor_name"] = investor
            return json.dumps(agent._call_mcp("neomem_retrieve_context", args), indent=2)

        def search_memories(query: str, limit: int = 10) -> str:
            """Vector search across stored memories with optional filtering.

            Args:
                query: Search text (natural language).
                limit: Max results to return (default 10).

            Returns:
                JSON results list with id, memory, score, metadata.
            """
            args = {
                "query": query,
                "user_id": agent.user_id,
                "agent_id": agent.agent_id,
                "limit": limit,
            }
            investor = agent.current_investor
            if investor:
                args["investor_name"] = investor
            return json.dumps(agent._call_mcp("neomem_search", args), indent=2)

        # ── Direct Storage ────────────────────────────────────────────

        def store_fact(fact: str, category: str = "persona") -> str:
            """Store a specific fact directly (bypasses LLM extraction).

            Use for explicit client statements: preferences, life events,
            risk tolerance, etc.

            Args:
                fact: The fact string to store.
                category: Memory category — persona, preference, episodic, procedural.

            Returns:
                JSON with stored memory ID.
            """
            args = {
                "fact": fact,
                "user_id": agent.user_id,
                "agent_id": agent.agent_id,
                "categories": [category],
            }
            investor = agent.current_investor
            if investor:
                args["investor_name"] = investor
            return json.dumps(agent._call_mcp("neomem_store_fact", args), indent=2)

        def store_preference(preference: str) -> str:
            """Store a client preference directly.

            Tags with 'preference' category automatically.
            Examples: communication style, report format, sector exclusions.

            Args:
                preference: The preference text.

            Returns:
                JSON with stored memory ID.
            """
            return json.dumps(agent._call_mcp("neomem_store_preference", {
                "preference": preference,
                "user_id": agent.user_id,
                "agent_id": agent.agent_id,
            }), indent=2)

        # ── Retrieval ─────────────────────────────────────────────────

        def get_memory(memory_id: str) -> str:
            """Get a specific memory by UUID.

            Args:
                memory_id: The memory's UUID.

            Returns:
                JSON memory record with id, memory, metadata.
            """
            return json.dumps(agent._call_mcp("neomem_get_memory", {
                "memory_id": memory_id,
            }), indent=2)

        def get_all_memories() -> str:
            """Get all stored memories for the current user/agent.

            Returns:
                JSON with results list of all memories.
            """
            return json.dumps(agent._call_mcp("neomem_get_all", {
                "user_id": agent.user_id,
                "agent_id": agent.agent_id,
            }), indent=2)

        def count_memories() -> str:
            """Count total stored memories.

            Returns:
                JSON with count.
            """
            return json.dumps(agent._call_mcp("neomem_count_memories", {
                "user_id": agent.user_id,
            }), indent=2)

        # ── Update & Delete ───────────────────────────────────────────

        def update_memory(memory_id: str, new_content: str) -> str:
            """Update an existing memory's content.

            Args:
                memory_id: UUID to update.
                new_content: New text content.

            Returns:
                JSON confirmation.
            """
            return json.dumps(agent._call_mcp("neomem_update_memory", {
                "memory_id": memory_id, "data": new_content,
            }), indent=2)

        def delete_memory(memory_id: str) -> str:
            """Delete a specific memory.

            Args:
                memory_id: UUID to delete.

            Returns:
                JSON confirmation.
            """
            return json.dumps(agent._call_mcp("neomem_delete_memory", {
                "memory_id": memory_id,
            }), indent=2)

        def delete_all_memories() -> str:
            """Delete ALL memories for the current user. Use with caution.

            Returns:
                JSON confirmation with deleted count.
            """
            return json.dumps(agent._call_mcp("neomem_delete_all", {
                "user_id": agent.user_id,
                "agent_id": agent.agent_id,
            }), indent=2)

        # ── Buffering (VIC §5) ────────────────────────────────────────

        def buffer_conversation(query: str, response: str) -> str:
            """Buffer a low-priority exchange for deferred processing.

            Instead of immediately running LLM extraction, the exchange
            is buffered. The buffer auto-flushes at threshold (10 exchanges)
            or after idle timeout (60s).

            Args:
                query: The user's message.
                response: The advisor's reply.

            Returns:
                JSON with buffer status (size, threshold).
            """
            args = {
                "query": query, "response": response,
                "user_id": agent.user_id, "agent_id": agent.agent_id,
                "session_id": agent._session_id,
            }
            investor = agent.current_investor
            if investor:
                args["investor_name"] = investor
            return json.dumps(agent._call_mcp("neomem_buffer_exchange", args), indent=2)

        def flush_buffer() -> str:
            """Manually flush all buffered exchanges through the extraction pipeline.

            Called automatically on session disconnect or when buffer threshold is reached.

            Returns:
                JSON with flushed count and extraction results.
            """
            return json.dumps(agent._call_mcp("neomem_flush_buffer", {
                "user_id": agent.user_id,
                "session_id": agent._session_id,
                "agent_id": agent.agent_id,
            }), indent=2)

        # ── Analysis (VIC §7) ─────────────────────────────────────────

        def score_salience(facts: list[str]) -> str:
            """Score facts for importance (0.0 = trivial, 1.0 = critical).

            Use before storing to evaluate whether facts are worth persisting.
            Salience ranges:
            - 0.85-1.0: Critical (core identity, life-changing events)
            - 0.65-0.84: High (actionable preferences, significant events)
            - 0.40-0.64: Medium (useful context)
            - 0.20-0.39: Low (marginal observations)
            - 0.00-0.19: Trivial (pleasantries — filtered by gate)

            Args:
                facts: List of fact strings to score.

            Returns:
                JSON with scored facts and salience values.
            """
            return json.dumps(agent._call_mcp("neomem_score_salience", {
                "facts": facts,
            }), indent=2)

        # ── History & Audit (VIC §22) ─────────────────────────────────

        def get_memory_history(memory_id: str) -> str:
            """Get the audit trail for a specific memory (ADD, UPDATE, DELETE events).

            Args:
                memory_id: UUID to get history for.

            Returns:
                JSON with history events list.
            """
            return json.dumps(agent._call_mcp("neomem_get_history", {
                "memory_id": memory_id,
            }), indent=2)

        def get_all_history(limit: int = 50) -> str:
            """Get the full audit trail of all memory operations.

            Args:
                limit: Max events to return (default 50).

            Returns:
                JSON with all history events.
            """
            return json.dumps(agent._call_mcp("neomem_get_all_history", {
                "limit": limit,
            }), indent=2)

        def get_history_stats() -> str:
            """Get aggregate statistics about memory operations.

            Returns:
                JSON with stats (total events, adds, updates, deletes).
            """
            return json.dumps(agent._call_mcp("neomem_get_history_stats", {}), indent=2)

        # ── Lifecycle ─────────────────────────────────────────────────

        def cleanup_expired() -> str:
            """Remove expired memories past their TTL.

            Returns:
                JSON with cleanup results.
            """
            return json.dumps(agent._call_mcp("neomem_cleanup_expired", {}), indent=2)

        # ── Scoped Memory (VIC §11 — pool hierarchy) ──────────────────

        def store_scoped(user_message: str, assistant_response: str,
                         pool: str = "private", tenant_id: str = "default",
                         dept_id: str = "") -> str:
            """Store a conversation with enterprise scope isolation.

            Pool hierarchy:
            - private: RM + specific investor only (default)
            - team: Entire department (compliance flags, team alerts)
            - org: Organization-wide (market insights, policy changes)

            Args:
                user_message: The user's message.
                assistant_response: The advisor's reply.
                pool: Access tier — private, team, or org.
                tenant_id: Organization ID.
                dept_id: Department ID (required for team pool).

            Returns:
                JSON with stored results.
            """
            args = {
                "user_message": user_message,
                "assistant_response": assistant_response,
                "user_id": agent.user_id,
                "tenant_id": tenant_id,
                "pool": pool,
            }
            if dept_id:
                args["dept_id"] = dept_id
            return json.dumps(agent._call_mcp("neomem_store_scoped", args), indent=2)

        def search_scoped(query: str, pool: str = "private",
                          tenant_id: str = "default", dept_id: str = "") -> str:
            """Search within an enterprise scope.

            Args:
                query: Search text.
                pool: Access tier — private, team, or org.
                tenant_id: Organization ID.
                dept_id: Department ID.

            Returns:
                JSON with scoped search results.
            """
            args = {
                "query": query,
                "user_id": agent.user_id,
                "tenant_id": tenant_id,
                "pool": pool,
            }
            if dept_id:
                args["dept_id"] = dept_id
            return json.dumps(agent._call_mcp("neomem_search_scoped", args), indent=2)

        def add_scoped(message: str, pool: str = "private",
                       tenant_id: str = "default", dept_id: str = "") -> str:
            """Store a single fact with enterprise scope isolation.

            Args:
                message: The fact or message to store.
                pool: Access tier — private, team, or org.
                tenant_id: Organization ID.
                dept_id: Department ID (required for team pool).

            Returns:
                JSON with stored results.
            """
            args = {
                "messages": message,
                "user_id": agent.user_id,
                "tenant_id": tenant_id,
                "pool": pool,
            }
            if dept_id:
                args["dept_id"] = dept_id
            return json.dumps(agent._call_mcp("neomem_add_scoped", args), indent=2)

        def get_all_scoped(pool: str = "private",
                           tenant_id: str = "default", dept_id: str = "") -> str:
            """Get all memories within an enterprise scope.

            Args:
                pool: Access tier — private, team, or org.
                tenant_id: Organization ID.
                dept_id: Department ID.

            Returns:
                JSON with all scoped memories.
            """
            args = {
                "user_id": agent.user_id,
                "tenant_id": tenant_id,
                "pool": pool,
            }
            if dept_id:
                args["dept_id"] = dept_id
            return json.dumps(agent._call_mcp("neomem_get_all_scoped", args), indent=2)

        return [
            # Core exchange (3)
            store_conversation, recall_context, search_memories,
            # Direct storage (2)
            store_fact, store_preference,
            # Retrieval (3)
            get_memory, get_all_memories, count_memories,
            # Update & Delete (3)
            update_memory, delete_memory, delete_all_memories,
            # Buffering (2)
            buffer_conversation, flush_buffer,
            # Analysis (1)
            score_salience,
            # History & Audit (3)
            get_memory_history, get_all_history, get_history_stats,
            # Lifecycle (1)
            cleanup_expired,
            # Scoped / enterprise (4)
            store_scoped, search_scoped, add_scoped, get_all_scoped,
        ]

    @staticmethod
    def _extract_response(result) -> str:
        if hasattr(result, "messages"):
            for msg in reversed(result.messages):
                content = getattr(msg, "content", None)
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(block["text"])
                        elif isinstance(block, str):
                            parts.append(block)
                    if parts:
                        return "\n".join(parts)
                elif isinstance(content, str) and content.strip():
                    return content
        return "I processed your request."


# =====================================================================
# Default system prompt — VIC-style wealth advisor (§21)
# =====================================================================

DEFAULT_SYSTEM_PROMPT = """\
You are a senior wealth management advisor (VIC — Virtual Investment Concierge)
with access to a persistent memory system. You help relationship managers (RMs)
serve high-net-worth client investors with portfolio strategy, risk management,
tax planning, retirement, and estate planning.

## Memory Tools Available

You have 22 tools for managing client memory:

### Before Responding — ALWAYS Recall Context
- **recall_context(query)**: ALWAYS call this FIRST for every user message.
  Retrieves relevant past memories scoped to the current investor. Returns
  persona, preference, episodic, and procedural memories as formatted context.

### Storage (Automatic + Manual)
- **store_conversation(query, response)**: Store a conversation exchange.
  Automatically runs: fact extraction → salience scoring → dedup → persist.
  Note: This is called automatically after every response.
- **store_fact(fact, category)**: Directly store an important fact.
  Categories: persona, preference, episodic, procedural.
- **store_preference(preference)**: Store a client preference.

### Search & Retrieval
- **search_memories(query, limit)**: Vector search across all stored memories.
- **get_memory(memory_id)**: Get a specific memory by ID.
- **get_all_memories()**: List all memories for the current client.
- **count_memories()**: Count total stored memories.

### Update & Delete
- **update_memory(memory_id, new_content)**: Update a memory's content.
- **delete_memory(memory_id)**: Delete a specific memory.
- **delete_all_memories()**: Delete ALL memories (use with extreme caution).

### Buffering (Deferred Processing)
- **buffer_conversation(query, response)**: Buffer low-priority exchanges
  for batch processing instead of immediate extraction.
- **flush_buffer()**: Process all buffered conversations.

### Analysis
- **score_salience(facts)**: Score facts for importance (0.0–1.0).

### History & Audit
- **get_memory_history(memory_id)**: Audit trail for a specific memory.
- **get_all_history(limit)**: Full audit trail of all memory operations.
- **get_history_stats()**: Aggregate stats (total events, adds, updates, deletes).

### Lifecycle
- **cleanup_expired()**: Remove expired memories past their TTL.

### Enterprise Scoped Memory (Pool Hierarchy)
- **store_scoped(user_message, assistant_response, pool)**: Store with scope.
  - private: RM + specific investor only (default)
  - team: Entire department
  - org: Organization-wide
- **search_scoped(query, pool)**: Search within a scope.
- **add_scoped(message, pool)**: Store a fact within a scope.
- **get_all_scoped(pool)**: List memories within a scope.

## Workflow

For EVERY user message:
1. **ALWAYS** call **recall_context** FIRST to retrieve relevant past context.
2. Use the recalled context to personalize your response — weave memory
   naturally, don't quote it verbatim.
3. Storage is handled automatically after every exchange.

When the RM shares important client details:
- Call **store_fact** for explicit statements (risk tolerance, life events, goals).
- Call **store_preference** for communication/service preferences.

## Memory Categories
- **persona**: WHO the client is — risk behavior, family context, career, goals.
- **preference**: HOW they want to be served — report format, meeting cadence.
- **episodic**: Significant EVENTS — market reactions, decisions, life changes.
- **procedural**: SYSTEM-LEARNED behaviors — reminders, recurring reviews.

## Important Rules
- If recalled memory conflicts with newer information, prefer the latest.
- Never reference investor details when query is about a different topic.
- If no memory context is available, that's fine — just respond normally.
- Be particularly sensitive to emotional context in episodic memories.
"""


# =====================================================================
# Helpers
# =====================================================================

def _classify_operation(tool_name: str) -> str:
    t = tool_name.replace("neomem_", "")
    if t in ("store_exchange", "store_fact", "buffer_exchange", "flush_buffer",
             "store_preference", "add_scoped", "store_scoped"):
        return "CREATE"
    if t in ("update_memory",):
        return "UPDATE"
    if t in ("delete_memory", "delete_all", "cleanup_expired"):
        return "DELETE"
    return "READ"


def _summarize_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 60:
            s = s[:57] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts[:4])


def _summarize_result(result: dict) -> str:
    if not isinstance(result, dict):
        return str(result)[:80]
    parts = []
    if "facts_stored" in result:
        parts.append(f"stored={result['facts_stored']}")
    if "facts_extracted" in result:
        parts.append(f"extracted={result['facts_extracted']}")
    if "facts_skipped" in result:
        parts.append(f"skipped={result['facts_skipped']}")
    if "investor_name" in result and result["investor_name"]:
        parts.append(f"investor={result['investor_name']}")
    if "results" in result:
        r = result["results"]
        if isinstance(r, list):
            parts.append(f"count={len(r)}")
    if "context" in result:
        ctx = result["context"]
        if isinstance(ctx, str):
            parts.append(f"ctx_len={len(ctx)}")
    if "count" in result:
        parts.append(f"count={result['count']}")
    if "buffered" in result:
        parts.append(f"buffered={result['buffered']}")
    if parts:
        return ", ".join(parts)
    return json.dumps(result, default=str)[:80]


# =====================================================================
# CLI — Interactive chat
# =====================================================================

def _print_ops(ops: list[MemoryOperation]) -> None:
    if not ops:
        return
    print(f"\n  [{len(ops)} memory ops this turn]")
    for op in ops:
        icon = {"CREATE": "🟢", "READ": "🔵", "UPDATE": "🟡", "DELETE": "🔴"}.get(op.operation, "⚪")
        print(f"    {icon} {op.tool}: {op.result_summary}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    agent = VICMemoryAgent()
    tool_count = agent.connect()
    print(f"  ✅ Connected — {tool_count} tools available")
    print(f"  Session: {agent._session_id}")
    print(f"  User: {agent.user_id} | Agent: {agent.agent_id} | Dept: {agent.dept_id}")
    print()
    print("  Commands: quit | ops | tools | count | memories | clear | investor | flush")
    print()

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                break
            if user_input.lower() == "ops":
                for op in agent.operations[-20:]:
                    icon = {"CREATE": "🟢", "READ": "🔵", "UPDATE": "🟡", "DELETE": "🔴"}.get(op.operation, "⚪")
                    print(f"  {icon} [{op.timestamp}] {op.tool}: {op.result_summary}")
                continue
            if user_input.lower() == "tools":
                for t in agent.list_tools():
                    print(f"  • {t}")
                continue
            if user_input.lower() == "count":
                r = agent.call_tool("neomem_count_memories", {"user_id": agent.user_id})
                print(f"  Memories: {r}")
                continue
            if user_input.lower() == "memories":
                r = agent.call_tool("neomem_get_all", {
                    "user_id": agent.user_id, "agent_id": agent.agent_id,
                })
                results = r.get("results", [])
                print(f"  Total: {len(results)} memories")
                for m in results[:20]:
                    mem_text = m.get("memory", "")[:80]
                    cat = m.get("metadata", {}).get("categories", [])
                    inv = m.get("metadata", {}).get("investor_name", "")
                    print(f"  • [{', '.join(cat) if cat else '?'}] {inv or '-'}: {mem_text}")
                continue
            if user_input.lower() == "clear":
                agent.clear_history()
                agent.clear_operations()
                print("  ✅ History and operations cleared")
                continue
            if user_input.lower() == "investor":
                print(f"  Current investor: {agent.current_investor or '(none)'}")
                continue
            if user_input.lower() == "flush":
                r = agent.call_tool("neomem_flush_buffer", {
                    "user_id": agent.user_id,
                    "session_id": agent._session_id,
                    "agent_id": agent.agent_id,
                })
                print(f"  Buffer flushed: {r}")
                continue

            response, ops = agent.chat(user_input)
            print(f"\nAdvisor: {response}")
            _print_ops(ops)
            if agent.current_investor:
                print(f"  📌 Investor: {agent.current_investor}")
            print()

    except KeyboardInterrupt:
        print("\n  Shutting down...")
    finally:
        agent.close()
        print("  ✅ Disconnected")


if __name__ == "__main__":
    main()
