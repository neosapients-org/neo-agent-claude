"""Data models for MCP capability definitions and tool metadata."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CapabilityDefinition:
    """Parsed from a *.capability.yaml file.

    This is the single source of truth for each MCP tool/resource.
    The MCP server reads these at startup — it contains zero hardcoded definitions.
    """

    name: str                                  # e.g. "memory.store_exchange"
    mcp_tool_name: str                         # e.g. "neomem_store_exchange"
    required_package: str                      # e.g. "neo_memory_hub" or "memory_utils"
    connector: str                             # "high_level" | "low_level" | "scoped" | "scorer"
    method: str                                # e.g. "store_exchange"
    description: str                           # tool description for LLM
    input_schema: dict = field(default_factory=dict)
    execution_mode: str = "optional"           # "mandatory" | "optional"
    hook_position: Optional[str] = None        # "pre" | "post" | None
    annotations: dict = field(default_factory=dict)
    version: str = "0.3.0"
    mcp_type: str = "tool"                     # "tool" | "resource"
    mcp_resource_uri: Optional[str] = None     # only for mcp_type="resource"
    source_file: str = ""

    @classmethod
    def from_dict(cls, data: dict, source_file: str = "") -> "CapabilityDefinition":
        return cls(
            name=data["name"],
            mcp_tool_name=data.get("mcp_tool_name", ""),
            required_package=data["required_package"],
            connector=data.get("connector", "high_level"),
            method=data.get("method", ""),
            description=data.get("description", ""),
            input_schema=data.get("input_schema", {}),
            execution_mode=data.get("execution_mode", "optional"),
            hook_position=data.get("hook_position"),
            annotations=data.get("annotations", {}),
            version=data.get("version", "0.3.0"),
            mcp_type=data.get("mcp_type", "tool"),
            mcp_resource_uri=data.get("mcp_resource_uri"),
            source_file=source_file,
        )

    def to_dict(self) -> dict:
        result = {
            "name": self.name,
            "mcp_tool_name": self.mcp_tool_name,
            "required_package": self.required_package,
            "connector": self.connector,
            "method": self.method,
            "description": self.description,
            "input_schema": self.input_schema,
            "execution_mode": self.execution_mode,
            "hook_position": self.hook_position,
            "annotations": self.annotations,
            "version": self.version,
            "mcp_type": self.mcp_type,
        }
        if self.mcp_resource_uri:
            result["mcp_resource_uri"] = self.mcp_resource_uri
        return result


@dataclass
class ToolMetadata:
    """Extended tool info returned by list_tools_with_metadata().

    Includes execution_mode and hook_position so frameworks can auto-wire
    mandatory vs optional tools.
    """

    name: str
    description: str
    execution_mode: str
    hook_position: Optional[str]
    required_package: str
    connector: str
    version: str
    input_schema: dict = field(default_factory=dict)

    def is_mandatory(self) -> bool:
        return self.execution_mode == "mandatory"

    def is_pre_hook(self) -> bool:
        return self.hook_position == "pre"

    def is_post_hook(self) -> bool:
        return self.hook_position == "post"
