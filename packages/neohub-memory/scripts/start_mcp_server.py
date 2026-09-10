#!/usr/bin/env python3
"""Start the NeoMemory MCP Server with pre-flight checks.

A production-ready launcher that validates dependencies, checks Qdrant
connectivity, and boots the server. Replaces the old bash script with
a portable Python equivalent.

Usage:
    python scripts/start_mcp_server.py                    # defaults
    python scripts/start_mcp_server.py --port 9000        # custom port
    python scripts/start_mcp_server.py --inspect           # + MCP Inspector UI

Environment:
    OPENAI_API_KEY          Required
    NEOMEM_MCP_PORT         Server port  (default 18432)
    NEOMEM_MCP_HOST         Bind address (default 0.0.0.0)
    NEOMEM_QDRANT_URL       Qdrant URL   (default http://localhost:6335)
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

# ── Resolve project root ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# Load .env early
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

# ── Colours (ANSI, noop on non-TTY) ──────────────────────────────────
_IS_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _IS_TTY else text


def ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠️  {_c('33', msg)}")


def fail(msg: str) -> None:
    print(f"  ❌ {_c('31', msg)}")


def heading(msg: str) -> None:
    print(f"\n── {_c('1', msg)} {'─' * max(0, 55 - len(msg))}")


# ── Pre-flight checks ────────────────────────────────────────────────

def check_openai_key() -> bool:
    if os.environ.get("OPENAI_API_KEY"):
        ok("OPENAI_API_KEY is set")
        return True
    fail("OPENAI_API_KEY not found — set it or add to .env")
    return False


def check_python() -> bool:
    v = sys.version.split()[0]
    ok(f"Python {v}")
    return True


def check_import(package: str, label: str | None = None) -> bool:
    label = label or package
    try:
        importlib.import_module(package)
        ok(f"{label}: installed")
        return True
    except ImportError:
        fail(f"{label}: NOT installed")
        return False


def check_qdrant(url: str) -> bool:
    """Check if Qdrant is reachable via a simple HTTP GET."""
    try:
        import urllib.request
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5):
            pass
        ok(f"Qdrant: reachable at {url}")
        return True
    except Exception as e:
        warn(f"Qdrant not reachable at {url} ({e})")
        return False


def check_port(port: int) -> bool:
    """Check if the port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            ok(f"Port {port}: available")
            return True
        except OSError:
            fail(f"Port {port}: in use — stop the other process or use --port")
            return False


def count_capabilities() -> int:
    from neomem_mcp.loader import get_bundled_capabilities_dir
    caps_dir = Path(get_bundled_capabilities_dir())
    if not caps_dir.exists():
        return 0
    return len(list(caps_dir.glob("*.capability.yaml")))


# ── MCP Inspector launcher ───────────────────────────────────────────

def launch_inspector(server_url: str) -> None:
    """Launch MCP Inspector in the background and open the browser."""
    heading("MCP Inspector")
    try:
        env = os.environ.copy()
        env["DANGEROUSLY_OMIT_AUTH"] = "true"
        subprocess.Popen(
            ["npx", "-y", "@modelcontextprotocol/inspector", "--url", server_url],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        ok("Inspector starting on http://localhost:6274")

        # Open browser
        if platform.system() == "Darwin":
            subprocess.Popen(["open", "http://localhost:6274"])
        elif platform.system() == "Linux":
            subprocess.Popen(["xdg-open", "http://localhost:6274"])
        ok("Browser opening → http://localhost:6274")
    except FileNotFoundError:
        warn("npx not found — install Node.js to use MCP Inspector")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start the NeoMemory MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python scripts/start_mcp_server.py
              python scripts/start_mcp_server.py --port 9000
              python scripts/start_mcp_server.py --inspect
        """),
    )
    parser.add_argument("--port", type=int, default=None, help="Server port (default: 18432)")
    parser.add_argument("--host", type=str, default=None, help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--inspect", action="store_true", help="Also launch MCP Inspector UI")
    args = parser.parse_args()

    # Resolve config
    port = args.port or int(os.environ.get("NEOMEM_MCP_PORT", "18432"))
    host = args.host or os.environ.get("NEOMEM_MCP_HOST", "0.0.0.0")
    qdrant_url = os.environ.get("NEOMEM_QDRANT_URL", "http://localhost:6335")

    # Banner
    print("=" * 60)
    print("  NeoMemory MCP Server — Startup")
    print("=" * 60)

    heading("Pre-flight checks")
    critical_ok = True
    critical_ok &= check_openai_key()
    critical_ok &= check_python()
    critical_ok &= check_import("neo_memory_hub")
    critical_ok &= check_import("neomem_mcp")
    check_import("memory_utils")  # optional — scoped tools
    check_qdrant(qdrant_url)
    critical_ok &= check_port(port)

    caps = count_capabilities()
    ok(f"Capabilities: {caps} YAML files (bundled in neomem_mcp)")

    if not critical_ok:
        print(f"\n{_c('31', '  ⛔ Pre-flight checks failed. Fix errors above.')}")
        sys.exit(1)

    # Set env vars so MCPServerConfig.from_env() picks them up
    os.environ["NEOMEM_MCP_HOST"] = host
    os.environ["NEOMEM_MCP_PORT"] = str(port)

    heading("Starting server")
    server_url = f"http://{host}:{port}/mcp"
    print(f"  Host:      {host}")
    print(f"  Port:      {port}")
    print(f"  MCP URL:   {server_url}")
    print(f"  Qdrant:    {qdrant_url}")
    print(f"  Transport: streamable-http")
    print()

    # Launch Inspector if requested
    if args.inspect:
        launch_inspector(server_url)
        print()

    print("  Press Ctrl+C to stop the server\n")

    # Boot the server
    from neomem_mcp.config import MCPServerConfig
    from neomem_mcp.server import create_mcp_server

    config = MCPServerConfig(
        host=host,
        port=port,
        transport="streamable-http",
    )
    mcp = create_mcp_server(server_config=config)
    mcp.run(transport=config.transport)


if __name__ == "__main__":
    main()
