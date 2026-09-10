#!/usr/bin/env python3
"""Quick verification of the prompt pipeline wiring."""
import sys, ast

sys.path.insert(0, "src")
sys.path.insert(0, "examples/demo")

# 1. Parse all modified files
for f in [
    "examples/demo/agent.py", "examples/demo/app.py",
    "src/neomem_mcp/config.py", "src/neomem_mcp/server.py",
    "src/neo_memory_hub/config/hub_config.py",
    "src/neo_memory_hub/connector.py",
]:
    ast.parse(open(f).read())
    print(f"  OK  {f} parses")

# 2. Agent no longer has custom prompt fields
from agent import MemoryAgent
a = MemoryAgent(user_id="test")
print(f"  OK  MemoryAgent has {len(a._build_tools())} tools")
assert not hasattr(a, "custom_fact_extraction_prompt"), "should be removed"
assert not hasattr(a, "custom_update_memory_prompt"), "should be removed"
print("  OK  custom_*_prompt removed from agent")

# 3. MCPServerConfig -> MemoryHubConfig prompt wiring
from neomem_mcp.config import MCPServerConfig, build_memory_config

cfg = MCPServerConfig(
    extraction_prompt="test_extraction",
    dedup_prompt="test_dedup",
    custom_fact_extraction_prompt="test_mem0_extract",
    custom_update_memory_prompt="test_mem0_update",
)
mem_cfg = build_memory_config(cfg)
assert mem_cfg.extraction.prompt == "test_extraction", f"got: {mem_cfg.extraction.prompt}"
assert mem_cfg.dedup.prompt == "test_dedup", f"got: {mem_cfg.dedup.prompt}"
assert mem_cfg.custom_fact_extraction_prompt == "test_mem0_extract"
assert mem_cfg.custom_update_memory_prompt == "test_mem0_update"
print("  OK  MCPServerConfig -> MemoryHubConfig prompt wiring")

# 4. MemoryHubConfig -> MemoryConfig (mem0) wiring via _build_mem0_config
from neo_memory_hub.connector import HighLevelMemoryConnector
hlc = HighLevelMemoryConnector(config=mem_cfg)
mem0_cfg = hlc._build_mem0_config()
assert mem0_cfg.custom_fact_extraction_prompt == "test_mem0_extract"
assert mem0_cfg.custom_update_memory_prompt == "test_mem0_update"
print("  OK  MemoryHubConfig -> MemoryConfig (mem0) prompt bridge")

# 5. Env var path (default = None when not set)
cfg2 = MCPServerConfig.from_env()
assert cfg2.extraction_prompt is None
assert cfg2.custom_fact_extraction_prompt is None
print("  OK  Env var defaults (None when unset)")

print("\nAll checks passed!")
