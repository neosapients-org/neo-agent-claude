"""Handler bridge — translates MCP inputs to library method calls.

Each handler is a thin async function that:
1. Receives MCP arguments as keyword args
2. Extracts the right connector from lifespan context
3. Calls the appropriate library method
4. Returns a JSON string (MCP text content)
5. Catches ALL exceptions — never leaks raw tracebacks

The resolve_handler() factory creates one handler per capability YAML.
It dynamically builds a function signature from the YAML input_schema
so FastMCP can introspect the parameters and generate correct Pydantic models.
"""

import asyncio
import inspect
import json
import logging
from typing import Any, Callable, Optional

from mcp.server.fastmcp import Context

from neomem_mcp.models import CapabilityDefinition

logger = logging.getLogger(__name__)

# JSON Schema type → Python type mapping
_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def resolve_handler(cap: CapabilityDefinition) -> Callable:
    """Create an async handler function for a capability definition.

    Builds a dynamic function signature from cap.input_schema so FastMCP
    can introspect parameters. ctx is typed as Context so FastMCP injects
    the lifespan context automatically.

    Routes to the correct connector based on cap.connector:
      - "high_level" → AppContext.hlc (HighLevelMemoryConnector)
      - "low_level"  → AppContext.low_level (NeoMemoryConnector)
      - "scoped"     → AppContext.scoped (ScopedMemoryConnector)
      - "scorer"     → AppContext.scorer (SalienceScorer)
    """
    connector_type = cap.connector
    method_name = cap.method
    tool_name = cap.mcp_tool_name

    async def handler(ctx: Context, **kwargs) -> str:
        try:
            app = ctx.request_context.lifespan_context

            # Strip None values — optional MCP params arrive as None
            # but downstream methods use their own defaults.
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            target = _resolve_target(app, connector_type, tool_name)
            if target is None:
                return _json_error(
                    "PackageNotInstalled",
                    f"Required package for {tool_name} is not installed.",
                )

            if connector_type == "scoped":
                result = await _call_scoped_method(target, method_name, kwargs)
            elif connector_type == "scorer":
                result = await _call_scorer_method(target, method_name, kwargs)
            else:
                method = getattr(target, method_name, None)
                if method is None:
                    return _json_error(
                        "MethodNotFound",
                        f"Method {method_name} not found on connector.",
                    )
                if asyncio.iscoroutinefunction(method):
                    result = await method(**kwargs)
                else:
                    result = method(**kwargs)

            return json.dumps(_serialize_result(result), indent=2, default=str)

        except Exception as e:
            logger.exception("Error in handler %s", tool_name)
            return json.dumps(_make_error_response(e, tool_name))

    handler.__name__ = tool_name
    handler.__doc__ = cap.description

    # Build explicit signature from input_schema so FastMCP generates
    # a correct Pydantic model instead of exposing ctx/**kwargs as fields.
    handler.__signature__ = _build_signature(cap.input_schema)

    return handler


def _build_signature(input_schema: dict) -> inspect.Signature:
    """Build an inspect.Signature from a JSON Schema input_schema.

    Creates: ctx: Context as first positional param (FastMCP injects it),
    followed by keyword-only params from the schema properties.
    Required properties have no default; optional ones default to None.
    """
    params = [
        inspect.Parameter(
            "ctx",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=Context,
        )
    ]

    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    for name, prop in properties.items():
        json_type = prop.get("type", "string")
        py_type = _TYPE_MAP.get(json_type, str)

        if name in required:
            p = inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=py_type,
            )
        else:
            p = inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=Optional[py_type],
                default=None,
            )
        params.append(p)

    return inspect.Signature(params, return_annotation=str)


def _resolve_target(app: Any, connector_type: str, tool_name: str) -> Any:
    """Resolve the target connector from AppContext."""
    if connector_type == "high_level":
        return app.hlc
    elif connector_type == "low_level":
        return app.low_level
    elif connector_type == "scoped":
        if app.scoped is None:
            return None
        return app.scoped
    elif connector_type == "scorer":
        return app.scorer
    else:
        logger.error("Unknown connector type %s for %s", connector_type, tool_name)
        return None


async def _call_scoped_method(scoped_connector: Any, method_name: str, kwargs: dict) -> Any:
    """Build IsolationScope from flat MCP params and call scoped method."""
    from neo_memory_hub.domain.scope import AccessTier, IsolationScope

    scope = IsolationScope(
        tenant_id=kwargs.pop("tenant_id", None) or "default",
        user_id=kwargs.pop("user_id", None),
        agent_id=kwargs.pop("agent_id", None),
        dept_id=kwargs.pop("dept_id", None),
        pool=AccessTier(kwargs.pop("pool", None) or "private"),
    )

    strategy_str = kwargs.pop("strategy", None)
    # Strip remaining None values — optional MCP params arrive as None
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if strategy_str:
        from memory_utils.shared_scope import SharedMemoryStrategy
        scoped_connector._strategy = SharedMemoryStrategy(strategy_str)

    method = getattr(scoped_connector, method_name, None)
    if method is None:
        raise AttributeError(f"Method {method_name} not found on ScopedMemoryConnector")

    return await method(scope=scope, **kwargs)


async def _call_scorer_method(scorer: Any, method_name: str, kwargs: dict) -> Any:
    """Call SalienceScorer methods with appropriate parameter mapping."""
    if method_name == "score_facts":
        facts = kwargs.get("facts", [])
        if isinstance(facts, str):
            facts = [facts]
        context = kwargs.get("context")
        return await scorer.score_facts(facts, context=context)
    elif method_name == "score_single":
        fact = kwargs.get("fact", "")
        context = kwargs.get("context")
        return await scorer.score_single(fact, context=context)
    else:
        method = getattr(scorer, method_name, None)
        if method is None:
            raise AttributeError(f"Method {method_name} not found on SalienceScorer")
        return await method(**kwargs)


def _serialize_result(result: Any) -> dict:
    """Convert library return types to JSON-serializable dicts."""
    if result is None:
        return {"result": None}
    if isinstance(result, dict):
        return result
    if isinstance(result, list):
        serialized = []
        for item in result:
            if hasattr(item, "model_dump"):
                serialized.append(item.model_dump())
            elif hasattr(item, "__dict__"):
                serialized.append(
                    {k: v for k, v in item.__dict__.items() if not k.startswith("_")}
                )
            else:
                serialized.append(item)
        return {"results": serialized}
    if isinstance(result, str):
        return {"result": result}
    if isinstance(result, (int, float, bool)):
        return {"result": result}
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "__dict__"):
        return {k: v for k, v in result.__dict__.items() if not k.startswith("_")}

    return {"result": str(result)}


def _make_error_response(e: Exception, tool_name: str) -> dict:
    """Convert exceptions to structured error dicts.

    Never exposes raw tracebacks — they leak internal paths and logic.
    """
    try:
        from neo_memory_hub.core.exceptions import (
            AccessDeniedError,
            ConfigurationError,
            NeoMemoryError,
            NotFoundError,
            StorageError,
            ValidationError,
        )

        if isinstance(e, ValidationError):
            return {"error": "ValidationError", "message": str(e)}
        elif isinstance(e, AccessDeniedError):
            return {"error": "AccessDeniedError", "message": str(e)}
        elif isinstance(e, NotFoundError):
            return {"error": "NotFoundError", "message": str(e)}
        elif isinstance(e, StorageError):
            return {"error": "StorageError", "message": str(e)}
        elif isinstance(e, ConfigurationError):
            return {"error": "ConfigurationError", "message": str(e)}
        elif isinstance(e, NeoMemoryError):
            return {"error": "NeoMemoryError", "message": str(e)}
    except ImportError:
        pass

    return {
        "error": "InternalError",
        "message": f"An internal error occurred in {tool_name}. "
        f"Contact your platform administrator.",
    }


def _json_error(error_type: str, message: str) -> str:
    """Create a JSON error string."""
    return json.dumps({"error": error_type, "message": message})
