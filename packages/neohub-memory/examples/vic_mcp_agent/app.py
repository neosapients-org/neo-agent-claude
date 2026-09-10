"""Streamlit UI for VIC Memory Agent.

Mirrors VIC's Streamlit debug panel with:
- Chat interface with memory operations sidebar
- Investor tracking display
- Memory browser (all memories with category/investor filtering)
- Operations log (audit trail of every MCP call)
- Settings (system prompt, extraction prompt, user/agent config)

Run:
    cd examples/vic_mcp_agent
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
import os

import streamlit as st

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(__file__))
from agent import VICMemoryAgent, MemoryOperation, DEFAULT_SYSTEM_PROMPT


# =====================================================================
# Session state
# =====================================================================

def init_state():
    defaults = {
        "messages": [],
        "user_id": "demo_rm",
        "agent_id": "vic_l3",
        "dept_id": "wealth",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "custom_extraction_prompt": "",
        "agent_initialized": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_agent() -> VICMemoryAgent:
    if "agent" not in st.session_state or not st.session_state.agent.is_connected:
        agent = VICMemoryAgent(
            user_id=st.session_state.user_id,
            agent_id=st.session_state.agent_id,
            dept_id=st.session_state.dept_id,
            system_prompt=st.session_state.system_prompt,
            custom_extraction_prompt=st.session_state.custom_extraction_prompt or None,
        )
        agent.connect()
        st.session_state.agent = agent
        st.session_state.agent_initialized = True
    return st.session_state.agent


# =====================================================================
# Sidebar
# =====================================================================

def render_sidebar():
    with st.sidebar:
        st.title("🧠 VIC Memory Agent")

        # Connection status
        if st.session_state.get("agent_initialized"):
            agent = st.session_state.agent
            st.success(f"Connected — {len(agent.list_tools())} tools")
            st.caption(f"Session: `{agent._session_id}`")

            # Investor tracking
            investor = agent.current_investor
            if investor:
                st.info(f"📌 Current Investor: **{investor}**")
            else:
                st.caption("No investor tracked yet")
        else:
            st.warning("Not connected")

        st.divider()

        # User config
        st.subheader("👤 Identity")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.user_id = st.text_input("User ID", st.session_state.user_id)
        with col2:
            st.session_state.agent_id = st.text_input("Agent ID", st.session_state.agent_id)
        st.session_state.dept_id = st.text_input("Dept ID", st.session_state.dept_id)

        st.divider()

        # Actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset Agent", use_container_width=True):
                if "agent" in st.session_state:
                    try:
                        st.session_state.agent.close()
                    except Exception:
                        pass
                    del st.session_state.agent
                st.session_state.agent_initialized = False
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🗑 Clear Chat", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.get("agent_initialized"):
                    st.session_state.agent.clear_history()
                    st.session_state.agent.clear_operations()
                st.rerun()


# =====================================================================
# Chat tab
# =====================================================================

def render_chat():
    st.header("💬 Chat")

    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "ops" in msg and msg["ops"]:
                with st.expander(f"📊 {len(msg['ops'])} memory operations"):
                    for op in msg["ops"]:
                        icon = {"CREATE": "🟢", "READ": "🔵", "UPDATE": "🟡", "DELETE": "🔴"}.get(op.operation, "⚪")
                        st.markdown(f"{icon} **{op.tool}**: {op.result_summary}")

    # Input
    if prompt := st.chat_input("Ask your wealth advisor..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response, ops = get_agent().chat(prompt)
                except Exception as e:
                    response = f"Error: {e}"
                    ops = []
                st.markdown(response)
                if ops:
                    with st.expander(f"📊 {len(ops)} memory operations"):
                        for op in ops:
                            icon = {"CREATE": "🟢", "READ": "🔵", "UPDATE": "🟡", "DELETE": "🔴"}.get(op.operation, "⚪")
                            st.markdown(f"{icon} **{op.tool}**: {op.result_summary}")

        st.session_state.messages.append({
            "role": "assistant", "content": response, "ops": ops,
        })


# =====================================================================
# Memory Browser tab
# =====================================================================

def render_memory_browser():
    st.header("🗄 Memory Browser")

    if not st.session_state.get("agent_initialized"):
        st.info("Connect to MCP server first (send a message in Chat).")
        return

    agent = get_agent()

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Refresh Memories", use_container_width=True):
            st.rerun()
    with col2:
        category_filter = st.selectbox(
            "Filter by category",
            ["all", "persona", "preference", "episodic", "procedural"],
        )
    with col3:
        investor_filter = st.text_input("Filter by investor", "")

    try:
        result = agent.call_tool("neomem_get_all", {
            "user_id": agent.user_id,
            "agent_id": agent.agent_id,
        })
        memories = result.get("results", [])
    except Exception as e:
        st.error(f"Failed to fetch memories: {e}")
        return

    # Apply filters
    filtered = []
    for m in memories:
        meta = m.get("metadata", {})
        cats = meta.get("categories", [])
        inv = meta.get("investor_name", "")

        if category_filter != "all" and category_filter not in cats:
            continue
        if investor_filter and investor_filter.lower() not in (inv or "").lower():
            continue
        filtered.append(m)

    st.caption(f"Showing {len(filtered)} of {len(memories)} memories")

    for m in filtered:
        meta = m.get("metadata", {})
        cats = meta.get("categories", [])
        inv = meta.get("investor_name", "")
        mem_id = m.get("id", "?")
        mem_text = m.get("memory", "")
        salience = meta.get("salience", "")
        created = m.get("created_at", "")

        cat_str = ", ".join(cats) if cats else "uncategorized"
        header = f"[{cat_str}] {inv or 'no investor'}"
        if salience:
            header += f" | salience: {salience}"

        with st.expander(header):
            st.markdown(f"**Memory:** {mem_text}")
            st.caption(f"ID: `{mem_id}` | Created: {created}")
            st.json(meta)

            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"📜 History", key=f"hist_{mem_id}"):
                    try:
                        hist = agent.call_tool("neomem_get_history", {"memory_id": mem_id})
                        st.json(hist)
                    except Exception as e:
                        st.error(str(e))
            with col2:
                if st.button(f"🗑 Delete", key=f"del_{mem_id}"):
                    try:
                        agent.call_tool("neomem_delete_memory", {"memory_id": mem_id})
                        st.success("Deleted")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))


# =====================================================================
# Operations Log tab
# =====================================================================

def render_operations():
    st.header("📋 Operations Log")

    if not st.session_state.get("agent_initialized"):
        st.info("Connect to MCP server first.")
        return

    agent = get_agent()
    ops = agent.operations

    if not ops:
        st.info("No operations yet. Start chatting!")
        return

    # Stats
    col1, col2, col3, col4 = st.columns(4)
    creates = sum(1 for o in ops if o.operation == "CREATE")
    reads = sum(1 for o in ops if o.operation == "READ")
    updates = sum(1 for o in ops if o.operation == "UPDATE")
    deletes = sum(1 for o in ops if o.operation == "DELETE")
    col1.metric("🟢 Creates", creates)
    col2.metric("🔵 Reads", reads)
    col3.metric("🟡 Updates", updates)
    col4.metric("🔴 Deletes", deletes)

    st.divider()

    # History stats from server
    if st.button("Fetch Server Audit Stats"):
        try:
            stats = agent.call_tool("neomem_get_history_stats", {})
            st.json(stats)
        except Exception as e:
            st.error(str(e))

    st.divider()

    # Operations list
    for op in reversed(ops):
        icon = {"CREATE": "🟢", "READ": "🔵", "UPDATE": "🟡", "DELETE": "🔴"}.get(op.operation, "⚪")
        with st.expander(f"{icon} [{op.timestamp}] {op.tool} — {op.result_summary}"):
            st.caption(f"Args: {op.args_summary}")
            st.json(op.raw_result)


# =====================================================================
# Settings tab
# =====================================================================

def render_settings():
    st.header("⚙️ Settings")

    st.subheader("🤖 System Prompt")
    st.caption("Changes take effect after 'Reset Agent'.")
    st.session_state.system_prompt = st.text_area(
        "System prompt",
        value=st.session_state.system_prompt,
        height=300,
    )

    st.divider()

    st.subheader("🔬 Custom Extraction Prompt")
    st.caption(
        "Per-call override for the fact extraction LLM prompt. "
        "Leave blank to use the library default. Changes take effect after 'Reset Agent'."
    )
    st.session_state.custom_extraction_prompt = st.text_area(
        "Extraction prompt",
        value=st.session_state.custom_extraction_prompt,
        height=200,
        placeholder="Leave blank to use default extraction prompt...",
    )

    st.divider()

    st.subheader("🧪 MCP Server Info")
    if st.button("Fetch Server Config"):
        try:
            st.json(get_agent().read_config())
        except Exception as e:
            st.error(str(e))
    if st.button("List Registered Tools"):
        try:
            tools = get_agent().list_tools()
            st.write(f"**{len(tools)} tools registered:**")
            for t in tools:
                st.code(t)
        except Exception as e:
            st.error(str(e))


# =====================================================================
# Main
# =====================================================================

def main():
    st.set_page_config(
        page_title="VIC Memory Agent",
        page_icon="🧠",
        layout="wide",
    )

    init_state()
    render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Chat", "🗄 Memory Browser", "📋 Operations", "⚙️ Settings",
    ])

    with tab1:
        render_chat()
    with tab2:
        render_memory_browser()
    with tab3:
        render_operations()
    with tab4:
        render_settings()


if __name__ == "__main__":
    main()
