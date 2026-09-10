"""CLI for neomem MCP server.

Commands:
    neomem-mcp serve     — Start the MCP server
    neomem-mcp validate  — Validate installation and capabilities
"""

import argparse
import importlib
import logging
import sys

from neomem_mcp.config import MCPServerConfig
from neomem_mcp.loader import (
    detect_installed_packages,
    filter_capabilities,
    load_capabilities,
)


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the MCP server."""
    config = MCPServerConfig.from_env()

    if args.port:
        config.port = args.port
    if args.host:
        config.host = args.host
    if args.transport:
        config.transport = args.transport
    if args.capabilities_dir:
        config.capabilities_dir = args.capabilities_dir

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    from neomem_mcp.server import create_mcp_server

    mcp = create_mcp_server(server_config=config)
    mcp.run(transport=config.transport)


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate installation — check packages, capabilities, versions."""
    config = MCPServerConfig.from_env()
    if args.capabilities_dir:
        config.capabilities_dir = args.capabilities_dir  # explicit override — use as-is

    print("=" * 60)
    print("  neomem-mcp installation validator")
    print("=" * 60)
    print()

    # Check core package
    errors = 0

    try:
        neo_hub = importlib.import_module("neo_memory_hub")
        version = getattr(neo_hub, "__version__", "unknown")
        print(f"  ✅ neo-memory-hub: {version}")
    except ImportError:
        print("  ❌ neo-memory-hub: NOT INSTALLED")
        errors += 1

    # Check MCP package
    try:
        neomem_mcp_mod = importlib.import_module("neomem_mcp")
        version = getattr(neomem_mcp_mod, "__version__", "unknown")
        print(f"  ✅ neomem-mcp: {version}")
    except ImportError:
        print("  ❌ neomem-mcp: NOT INSTALLED")
        errors += 1

    # Check MCP SDK
    try:
        importlib.import_module("mcp")
        print("  ✅ mcp SDK: installed")
    except ImportError:
        print("  ❌ mcp SDK: NOT INSTALLED (pip install 'mcp[cli]')")
        errors += 1

    # Check installed packages
    installed = detect_installed_packages()
    print(f"\n  Installed packages: {sorted(installed)}")

    if "memory_utils" in installed:
        print("  ✅ memory_utils: available (scoped tools enabled)")
    else:
        print("  ⏭️  memory_utils: not installed (scoped tools will be skipped)")

    # Check capabilities
    print(f"\n  Capabilities directory: {config.capabilities_dir}")
    capabilities = load_capabilities(config.capabilities_dir)

    if not capabilities:
        print("  ⚠️  No capability YAMLs found!")
        errors += 1
    else:
        print(f"  ✅ Capability YAMLs found: {len(capabilities)}")

        registerable, skipped = filter_capabilities(capabilities, installed)
        print(f"  ✅ Registerable tools: {len(registerable)}")

        for cap in registerable:
            print(f"     ✅ {cap.mcp_tool_name} (v{cap.version}) — {cap.required_package}")

        if skipped:
            print(f"  ⏭️  Skipped: {len(skipped)}")
            for cap in skipped:
                print(f"     ⏭️  {cap.mcp_tool_name} — {cap.required_package} not installed")

    print()
    if errors:
        print(f"  ❌ {errors} error(s) found. Fix before starting server.")
        sys.exit(1)
    else:
        print("  ✅ All checks passed. Ready to serve.")
        print(f"     Run: neomem-mcp serve --port {config.port}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="neomem-mcp",
        description="MCP server for neo_memory_hub",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the MCP server")
    serve_parser.add_argument("--port", type=int, help="Port number")
    serve_parser.add_argument("--host", type=str, help="Host to bind to")
    serve_parser.add_argument(
        "--transport",
        choices=["streamable-http", "stdio"],
        help="Transport type",
    )
    serve_parser.add_argument("--capabilities-dir", type=str, help="Capabilities directory")

    # validate
    validate_parser = subparsers.add_parser("validate", help="Validate installation")
    validate_parser.add_argument("--capabilities-dir", type=str, help="Capabilities directory")

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "validate":
        cmd_validate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
