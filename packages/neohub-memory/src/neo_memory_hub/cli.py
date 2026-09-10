"""Command-line interface for Neo Memory Hub.

This CLI is intentionally minimal. It exists primarily to provide a stable
console entrypoint for the packaged library.
"""

from __future__ import annotations

import argparse

from neo_memory_hub import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neo-memory", description="Neo Memory Hub CLI")
    parser.add_argument(
        "--version",
        action="version",
        version=f"neo-memory-hub {__version__}",
        help="Print version and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0


__all__ = ["build_parser", "main"]
