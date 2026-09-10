"""
NeMo Guardrails Provider for Neo Guardrail Hub.

This module provides integration with NVIDIA's NeMo Guardrails framework,
enabling:
1. Automatic generation of Colang configurations from YAML config (Phase 3A)
2. NeMo-based guardrails using LLM self-checks (Phase 3B)

Features:
- NeMoProvider: Run NeMo guardrails with auto-generated configs
- NeMoConfigGenerator: Generate complete NeMo config from default.yaml
- ConfigToNeMoMapper: Map YAML guardrail config to NeMo structures
- ColangGenerator: Generate Colang 2.x flow definitions
- Topic Library: Pre-built topics for topical rails

Usage:
    # Phase 3B: Using NeMo Provider for guardrails
    from neo_guardrail_hub.providers.nemo import NeMoProvider
    
    provider = NeMoProvider(config_path="./configs")
    await provider.initialize()
    
    # Check input using NeMo's self_check_input
    result = await provider.check_input("user message here")
    
    # Or use guardrail interface
    guardrail = provider.get_guardrail("self_check_input")
    result = await guardrail.check("user message")
    
    # Phase 3A: Generate NeMo config files only
    from neo_guardrail_hub.providers.nemo import generate_nemo_from_yaml
    
    output_dir = generate_nemo_from_yaml(
        config_path="./configs",          # Directory with default.yaml
        agent_id=None,                    # Use default config
        output_path="./nemo_generated"    # Where to save files
    )
"""

# Phase 3A: Config generation
from .config_mapper import ConfigToNeMoMapper, map_yaml_to_colang
from .generators import (
    BaseGenerator,
    ColangConfig,
    ColangGenerator,
    ConfigGenerator,
    InputRailConfig,
    LLMConfig,
    NeMoGuardrailsConfig,
    OutputRailConfig,
    PromptsGenerator,
    TopicalRailConfig,
    TopicLibrary,
    TopicLibraryEntry,
    generate_colang_from_dict,
)
from .nemo_generator import NeMoConfigGenerator, generate_nemo_from_yaml
from .topic_library import (
    get_all_disallowed_topics,
    get_harmful_topics,
    get_sensitive_topics,
    get_topic_by_name,
    get_topics_by_category,
    load_topic_library,
)

# Phase 3B: NeMo Provider and Guardrails
from .config_models import (
    NeMoDialogRailConfig,
    NeMoGenerationConfig,
    NeMoInputRailConfig,
    NeMoLLMConfig,
    NeMoOutputRailConfig,
    NeMoProviderConfig,
    NeMoRailsConfig,
    PromptStyle,
)
from .provider import NeMoProvider, create_nemo_provider

# Import NeMo guardrails from the correct location
from ...guardrails.input.nemo_self_check_input import NeMoSelfCheckInputGuardrail
from ...guardrails.input.nemo_jailbreak_detection_heuristics import NeMoJailbreakDetectionHeuristicsGuardrail

__all__ = [
    # Phase 3B: NeMo Provider (main entry point for guardrails)
    "NeMoProvider",
    "create_nemo_provider",
    # Phase 3B: NeMo Guardrails (from guardrails.input)
    "NeMoSelfCheckInputGuardrail",
    "NeMoJailbreakDetectionHeuristicsGuardrail",
    # Phase 3B: Provider Configuration Models
    "NeMoProviderConfig",
    "NeMoLLMConfig",
    "NeMoGenerationConfig",
    "NeMoRailsConfig",
    "NeMoInputRailConfig",
    "NeMoOutputRailConfig",
    "NeMoDialogRailConfig",
    "PromptStyle",
    # Phase 3A: High-level generators
    "NeMoConfigGenerator",
    "generate_nemo_from_yaml",
    # Phase 3A: Config mapping
    "ConfigToNeMoMapper",
    "map_yaml_to_colang",
    # Phase 3A: Low-level generators
    "BaseGenerator",
    "ColangGenerator",
    "ConfigGenerator",
    "PromptsGenerator",
    "generate_colang_from_dict",
    # Phase 3A: Configuration Models
    "ColangConfig",
    "InputRailConfig",
    "OutputRailConfig",
    "TopicalRailConfig",
    "LLMConfig",
    "NeMoGuardrailsConfig",
    # Phase 3A: Topic Library
    "TopicLibrary",
    "TopicLibraryEntry",
    "load_topic_library",
    "get_topic_by_name",
    "get_topics_by_category",
    "get_sensitive_topics",
    "get_harmful_topics",
    "get_all_disallowed_topics",
]
