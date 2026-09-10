"""MCP Server for neo_memory_hub — dynamic registration engine.

Contains ZERO hardcoded tool definitions. Reads capability YAMLs at startup,
checks which packages are installed, and registers only the intersection.
"""

__version__ = "0.1.0"

from neomem_mcp.models import CapabilityDefinition, ToolMetadata

__all__ = [
    "__version__",
    "CapabilityDefinition",
    "ToolMetadata",
]
