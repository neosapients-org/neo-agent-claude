"""Capability YAML scanner and package detector.

Scans a directory for *.capability.yaml files, parses them into
CapabilityDefinition objects, and detects which memory packages
are installed at runtime.
"""

import importlib
import logging
from pathlib import Path
from typing import Optional

import yaml

from neomem_mcp.models import CapabilityDefinition

logger = logging.getLogger(__name__)


_BUNDLED_DIR = str(Path(__file__).parent / "capabilities")


def get_bundled_capabilities_dir() -> str:
    """Return the path to the capabilities bundled inside the neomem_mcp package."""
    return _BUNDLED_DIR


def resolve_capabilities_dir(capabilities_dir: str | None = None) -> str:
    """Return a valid capabilities directory path.

    If *capabilities_dir* is provided and exists on disk, use it (allows
    users to supply a custom set of YAMLs).  Otherwise fall back to the
    copy that ships inside the ``neomem_mcp`` package.
    """
    if capabilities_dir and Path(capabilities_dir).is_dir():
        return capabilities_dir
    return _BUNDLED_DIR


def load_capabilities(capabilities_dir: str) -> list[CapabilityDefinition]:
    """Scan a directory for *.capability.yaml files and return parsed definitions.

    Args:
        capabilities_dir: Path to directory containing capability YAML files.

    Returns:
        List of parsed CapabilityDefinition objects, sorted by filename.
    """
    caps_path = Path(capabilities_dir)
    if not caps_path.exists():
        logger.warning("Capabilities directory not found: %s", capabilities_dir)
        return []

    capabilities: list[CapabilityDefinition] = []

    for yaml_file in sorted(caps_path.glob("*.capability.yaml")):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)

            if not data or not isinstance(data, dict):
                logger.warning("Empty or invalid YAML: %s", yaml_file)
                continue

            cap = CapabilityDefinition.from_dict(data, source_file=str(yaml_file))
            capabilities.append(cap)
            logger.debug("Loaded capability: %s from %s", cap.name, yaml_file.name)

        except yaml.YAMLError as e:
            logger.error("YAML parse error in %s: %s", yaml_file, e)
        except KeyError as e:
            logger.error("Missing required field %s in %s", e, yaml_file)
        except Exception as e:
            logger.error("Failed to load %s: %s", yaml_file, e)

    logger.info("Loaded %d capabilities from %s", len(capabilities), capabilities_dir)
    return capabilities


def detect_installed_packages() -> set[str]:
    """Detect which memory packages are importable at runtime.

    Returns:
        Set of installed package names (e.g. {"neo_memory_hub", "memory_utils"}).
    """
    installed: set[str] = set()

    for package in ["neo_memory_hub", "memory_utils"]:
        try:
            importlib.import_module(package)
            installed.add(package)
        except ImportError:
            pass

    logger.info("Detected installed packages: %s", installed)
    return installed


def filter_capabilities(
    capabilities: list[CapabilityDefinition],
    installed_packages: set[str],
    mcp_type: Optional[str] = None,
) -> tuple[list[CapabilityDefinition], list[CapabilityDefinition]]:
    """Split capabilities into registerable and skipped based on installed packages.

    Args:
        capabilities: All parsed capability definitions.
        installed_packages: Set of importable package names.
        mcp_type: If set, only return capabilities matching this type ("tool" or "resource").

    Returns:
        Tuple of (registerable, skipped) capability lists.
    """
    registerable: list[CapabilityDefinition] = []
    skipped: list[CapabilityDefinition] = []

    for cap in capabilities:
        if mcp_type and cap.mcp_type != mcp_type:
            continue

        if cap.required_package in installed_packages:
            registerable.append(cap)
        else:
            skipped.append(cap)
            logger.info(
                "Skipped %s — %s not installed",
                cap.mcp_tool_name or cap.name,
                cap.required_package,
            )

    return registerable, skipped
