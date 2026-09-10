"""
Compatibility layer for optional dependencies.

This module provides graceful fallbacks when optional dependencies are missing.
All internal ns_probe modules should import from here rather than directly
importing optional packages.
"""

from typing import Any, Dict, Optional, Callable
import warnings

# =============================================================================
# JSON Serialization
# =============================================================================

try:
    import orjson

    def json_dumps(obj: Any) -> str:
        """Serialize to JSON string (fast path with orjson)."""
        return orjson.dumps(obj).decode("utf-8")

    def json_dumps_bytes(obj: Any) -> bytes:
        """Serialize to JSON bytes (fast path with orjson)."""
        return orjson.dumps(obj)

    def json_loads(data: str) -> Any:
        """Deserialize from JSON string (fast path with orjson)."""
        return orjson.loads(data)

    JSON_ENCODER = "orjson"

except ImportError:
    import json

    def json_dumps(obj: Any) -> str:
        """Serialize to JSON string (stdlib fallback)."""
        return json.dumps(obj, separators=(",", ":"))

    def json_dumps_bytes(obj: Any) -> bytes:
        """Serialize to JSON bytes (stdlib fallback)."""
        return json.dumps(obj, separators=(",", ":")).encode("utf-8")

    def json_loads(data: str) -> Any:
        """Deserialize from JSON string (stdlib fallback)."""
        return json.loads(data)

    JSON_ENCODER = "json"


# =============================================================================
# HTTP Client
# =============================================================================

try:
    import requests

    def http_post(url: str, data: bytes, headers: Dict[str, str], timeout: float = 5.0) -> Any:
        """POST request (requests library)."""
        return requests.post(url, data=data, headers=headers, timeout=timeout)

    HTTP_CLIENT = "requests"

except ImportError:
    from urllib import request
    from urllib.error import URLError, HTTPError

    def http_post(url: str, data: bytes, headers: Dict[str, str], timeout: float = 5.0) -> Any:
        """POST request (urllib fallback)."""
        req = request.Request(url, data=data, headers=headers, method="POST")
        return request.urlopen(req, timeout=timeout)

    HTTP_CLIENT = "urllib"


# =============================================================================
# Protobuf
# =============================================================================

try:
    from google.protobuf import message as proto_message

    PROTOBUF_AVAILABLE = True

    def warn_if_protobuf_needed():
        """No-op if protobuf is available."""
        pass

except ImportError:
    PROTOBUF_AVAILABLE = False

    def warn_if_protobuf_needed():
        """Warn once if trying to use protobuf without it installed."""
        _warn_protobuf_missing()


class PerformanceWarning(UserWarning):
    """A slower fallback is in use because an optional dependency is missing.

    Its own category so a caller can silence exactly this and nothing else.
    It did not exist before, while `warn_if_protobuf_needed` already named it —
    so the one path that warned about degraded performance raised NameError
    instead of warning, and only when protobuf was already missing.
    """


_warned_protobuf = False


def _warn_protobuf_missing() -> None:
    """Warn once per process. Called from the export path, so once is the point."""
    global _warned_protobuf
    if _warned_protobuf:
        return
    warnings.warn(
        "Protobuf not installed. Using JSON format (slower). "
        "For better performance: pip install protobuf",
        PerformanceWarning,
        stacklevel=3,
    )
    _warned_protobuf = True


# =============================================================================
# MessagePack (Alternative Serialization)
# =============================================================================

try:
    import msgpack

    MSGPACK_AVAILABLE = True

    def msgpack_dumps(obj: Any) -> bytes:
        """Serialize to MessagePack bytes."""
        return msgpack.packb(obj)

    def msgpack_loads(data: bytes) -> Any:
        """Deserialize from MessagePack bytes."""
        return msgpack.unpackb(data, raw=False)

except ImportError:
    MSGPACK_AVAILABLE = False

    def msgpack_dumps(obj: Any) -> bytes:
        """Fallback to JSON if msgpack not available."""
        return json_dumps_bytes(obj)

    def msgpack_loads(data: bytes) -> Any:
        """Fallback to JSON if msgpack not available."""
        return json_loads(data.decode("utf-8"))


# =============================================================================
# Performance Warning Helper
# =============================================================================


def get_performance_hints() -> list[str]:
    """
    Get list of performance optimization hints based on missing dependencies.

    Returns:
        List of pip install commands for missing optional dependencies
    """
    hints = []

    if JSON_ENCODER != "orjson":
        hints.append("pip install orjson  # 5-10x faster JSON encoding")

    if not PROTOBUF_AVAILABLE:
        hints.append("pip install protobuf  # Efficient wire format for collectors")

    if HTTP_CLIENT != "requests":
        hints.append("pip install requests  # Better HTTP/2 support")

    if not MSGPACK_AVAILABLE:
        hints.append("pip install msgpack  # Alternative efficient serialization")

    return hints


def print_performance_hints():
    """Print performance hints if any dependencies are missing."""
    hints = get_performance_hints()
    if hints:
        print("[ns_probe] Performance optimizations available:")
        for hint in hints:
            print(f"  {hint}")
        print("  Or install all: pip install ns_probe[all]")


# Export all
__all__ = [
    # JSON
    "json_dumps",
    "json_dumps_bytes",
    "json_loads",
    "JSON_ENCODER",
    # HTTP
    "http_post",
    "HTTP_CLIENT",
    # Protobuf
    "PROTOBUF_AVAILABLE",
    "warn_if_protobuf_needed",
    # MessagePack
    "MSGPACK_AVAILABLE",
    "msgpack_dumps",
    "msgpack_loads",
    # Helpers
    "get_performance_hints",
    "print_performance_hints",
]
