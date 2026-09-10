#!/usr/bin/env python3
"""Quick verification of demo files."""
import sys, ast

sys.path.insert(0, "src")
sys.path.insert(0, "examples/demo")

# Parse both files
for f in ["examples/demo/agent.py", "examples/demo/app.py"]:
    with open(f) as fh:
        tree = ast.parse(fh.read())
    print(f"  OK  {f} parses ({len(tree.body)} nodes)")

# Verify MemoryAgent
from agent import MemoryAgent, MemoryOperation, DEFAULT_SYSTEM_PROMPT

print(f"  OK  MemoryAgent importable")
print(f"  OK  DEFAULT_SYSTEM_PROMPT: {len(DEFAULT_SYSTEM_PROMPT)} chars")

agent = MemoryAgent(user_id="test")
tools = agent._build_tools()
print(f"  OK  _build_tools() returns {len(tools)} tools")

assert hasattr(agent, "custom_fact_extraction_prompt")
assert hasattr(agent, "custom_update_memory_prompt")
print("  OK  Both custom prompt fields present")

from agent import _MCPBridge
print("  OK  _MCPBridge importable")

print("\nAll checks passed")
