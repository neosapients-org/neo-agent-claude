"""
Pre-generation API for Neo Guardrail Hub.

This module provides APIs to pre-generate NeMo configuration files
before using the library. This is essential for:
1. Faster startup - configs are already generated
2. Validation - catch config errors early
3. Customization - review and modify generated files

Usage:
    from neo_guardrail_hub.api import initialize, generate_configs
    
    # Option 1: Full initialization (generates configs + validates)
    result = await initialize(
        config_path="./configs",
        agent_id="wealth_advisor",  # Optional
    )
    
    # Option 2: Just generate configs
    result = generate_configs(
        config_path="./configs",
        agent_id="wealth_advisor",
        output_path="./nemo_generated",
    )
    
    print(f"Generated configs at: {result.output_path}")
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..core.config import ConfigLoader
from ..core.exceptions import ConfigurationError
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PreGenerationResult:
    """Result of pre-generation operation.
    
    Attributes:
        success: Whether the operation succeeded
        output_path: Path to generated config files
        agent_id: Agent ID if agent-specific
        errors: List of errors if any
        warnings: List of warnings if any
        generated_files: List of generated file paths
        config_summary: Summary of the configuration
    """
    success: bool
    output_path: Optional[Path] = None
    agent_id: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    generated_files: List[Path] = field(default_factory=list)
    config_summary: Dict[str, Any] = field(default_factory=dict)
    
    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.success


def generate_configs(
    config_path: Union[str, Path] = "./configs",
    agent_id: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    force: bool = False,
    validate: bool = True,
) -> PreGenerationResult:
    """
    Generate NeMo configuration files from YAML config.
    
    This function reads the default.yaml (or agent-specific yaml) and
    generates the necessary NeMo Guardrails configuration files.
    
    Args:
        config_path: Path to the configs directory containing default.yaml
        agent_id: Optional agent ID for agent-specific configuration
        output_path: Directory to save generated files. If None, uses
                    the path from config or defaults to ./neo_configs/
        force: If True, regenerate even if files exist
        validate: If True, validate the generated configuration
        
    Returns:
        PreGenerationResult with details about the generation
        
    Raises:
        ConfigurationError: If config cannot be loaded or is invalid
        
    Example:
        >>> result = generate_configs(config_path="./configs")
        >>> if result.success:
        ...     print(f"Generated at: {result.output_path}")
        ... else:
        ...     print(f"Errors: {result.errors}")
    """
    config_path = Path(config_path)
    result = PreGenerationResult(success=False, agent_id=agent_id)
    
    try:
        # Load configuration
        loader = ConfigLoader(config_path)
        config = loader.load(agent_id=agent_id)
        
        # Check if NeMo is enabled
        nemo_config = config.get("nemo", {})
        if not nemo_config.get("enabled", False):
            result.warnings.append("NeMo is not enabled in configuration")
            result.success = True
            result.config_summary = {"nemo_enabled": False}
            return result
        
        # Import NeMo generator
        from ..providers.nemo import NeMoConfigGenerator
        
        # Determine output path
        if output_path is not None:
            output_path = Path(output_path)
        elif nemo_config.get("output_path"):
            output_path = config_path / nemo_config["output_path"]
        else:
            output_path = None  # Let generator use default
        
        # Check if files already exist
        if output_path and output_path.exists() and not force:
            config_yml = output_path / "config.yml"
            if config_yml.exists():
                result.output_path = output_path
                result.generated_files = list(output_path.glob("*"))
                result.warnings.append(
                    f"Config files already exist at {output_path}. "
                    "Use force=True to regenerate."
                )
                result.success = True
                result.config_summary = _build_config_summary(config)
                return result
        
        # Generate configuration files
        generator = NeMoConfigGenerator(
            config_path=config_path,
            output_base_path=output_path.parent if output_path else None,
        )
        
        generated_path = generator.generate(
            agent_id=agent_id,
            output_path=output_path,
            prompt_style=nemo_config.get("prompt_style", "simple"),
        )
        
        result.output_path = generated_path
        result.generated_files = list(generated_path.glob("*"))
        result.config_summary = _build_config_summary(config)
        
        # Validate if requested
        if validate:
            validation_errors = _validate_generated_config(generated_path)
            if validation_errors:
                result.warnings.extend(validation_errors)
        
        result.success = True
        
        logger.info(
            "configs_generated",
            output_path=str(generated_path),
            agent_id=agent_id,
            files_count=len(result.generated_files),
        )
        
    except ImportError as e:
        result.errors.append(
            f"NeMo Guardrails not installed: {e}. "
            "Install with: pip install neo-guardrail-hub[nemo]"
        )
    except Exception as e:
        result.errors.append(f"Failed to generate configs: {str(e)}")
        logger.error("config_generation_failed", error=str(e))
    
    return result


def generate_configs_sync(
    config_path: Union[str, Path] = "./configs",
    agent_id: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    force: bool = False,
    validate: bool = True,
) -> PreGenerationResult:
    """
    Synchronous wrapper for generate_configs.
    
    See generate_configs for full documentation.
    """
    return generate_configs(
        config_path=config_path,
        agent_id=agent_id,
        output_path=output_path,
        force=force,
        validate=validate,
    )


async def initialize(
    config_path: Union[str, Path] = "./configs",
    agent_id: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    force: bool = False,
    preload_rails: bool = True,
) -> PreGenerationResult:
    """
    Initialize Neo Guardrail Hub with pre-generated configurations.
    
    This function:
    1. Generates NeMo configuration files
    2. Optionally pre-loads NeMo Rails into cache
    3. Validates the setup
    
    This is the recommended way to set up Neo Guardrail Hub before
    using it in production. It ensures all configurations are valid
    and Rails are cached for fast access.
    
    Args:
        config_path: Path to the configs directory
        agent_id: Optional agent ID for agent-specific configuration
        output_path: Directory for generated files
        force: If True, regenerate even if files exist
        preload_rails: If True, pre-load NeMo Rails into cache
        
    Returns:
        PreGenerationResult with initialization details
        
    Example:
        >>> result = await initialize(config_path="./configs")
        >>> if result.success:
        ...     print("Ready to use!")
        ...     # Now use the orchestrator
        ...     orchestrator = NeoGuardrailOrchestrator(config_path="./configs")
    """
    # First generate configs
    result = generate_configs(
        config_path=config_path,
        agent_id=agent_id,
        output_path=output_path,
        force=force,
        validate=True,
    )
    
    if not result.success:
        return result
    
    # Pre-load Rails if requested
    if preload_rails and result.output_path:
        try:
            from .caching import get_rails_cache
            
            cache = get_rails_cache()
            cache_key = agent_id or "default"
            
            # Load Rails into cache
            await cache.get_or_create(
                key=cache_key,
                config_path=result.output_path,
            )
            
            result.config_summary["rails_cached"] = True
            
            logger.info(
                "rails_preloaded",
                cache_key=cache_key,
                config_path=str(result.output_path),
            )
            
        except ImportError as e:
            result.warnings.append(
                f"Could not preload NeMo Rails: {e}"
            )
        except Exception as e:
            result.warnings.append(
                f"Error preloading Rails: {e}"
            )
    
    return result


def initialize_sync(
    config_path: Union[str, Path] = "./configs",
    agent_id: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    force: bool = False,
    preload_rails: bool = True,
) -> PreGenerationResult:
    """
    Synchronous wrapper for initialize.
    
    See initialize for full documentation.
    """
    return asyncio.get_event_loop().run_until_complete(
        initialize(
            config_path=config_path,
            agent_id=agent_id,
            output_path=output_path,
            force=force,
            preload_rails=preload_rails,
        )
    )


def _build_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build a summary of the configuration."""
    nemo_config = config.get("nemo", {})
    guardrails_config = config.get("guardrails", {})
    
    return {
        "nemo_enabled": nemo_config.get("enabled", False),
        "nemo_llm": nemo_config.get("llm", {}).get("model"),
        "nemo_preset": nemo_config.get("preset"),
        "input_rails": nemo_config.get("rails", {}).get("input", {}).get("enabled", False),
        "output_rails": nemo_config.get("rails", {}).get("output", {}).get("enabled", False),
        "dialog_rails": nemo_config.get("rails", {}).get("dialog", {}).get("enabled", False),
        "llm_guard_input": guardrails_config.get("input", {}).get("enabled", False),
        "llm_guard_output": guardrails_config.get("output", {}).get("enabled", False),
    }


def _validate_generated_config(config_path: Path) -> List[str]:
    """Validate generated NeMo configuration files."""
    warnings = []
    
    # Check required files
    config_yml = config_path / "config.yml"
    if not config_yml.exists():
        warnings.append(f"Missing config.yml at {config_path}")
        return warnings
    
    # Try to parse config.yml
    try:
        import yaml
        with open(config_yml) as f:
            config = yaml.safe_load(f)
        
        # Validate structure
        if "models" not in config:
            warnings.append("config.yml missing 'models' section")
        
    except Exception as e:
        warnings.append(f"Error parsing config.yml: {e}")
    
    return warnings
