"""
NeMo Provider Configuration Models.

This module defines Pydantic models for configuring the NeMo Guardrails provider,
including LLM settings, generation options, and rail configurations.

These models are used by the NeMoProvider to:
1. Configure how NeMo configs are generated
2. Control the LLM used for self-checks
3. Set up input/output rail behavior
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class NeMoLLMConfig(BaseModel):
    """LLM configuration for NeMo Guardrails.
    
    This configures the LLM used by NeMo for self-check tasks like
    self_check_input, self_check_output, self_check_facts, etc.
    
    Example:
        llm_config = NeMoLLMConfig(
            engine="openai",
            model="gpt-4",
            temperature=0.0,
        )
    """
    
    engine: str = Field(
        default="openai",
        description="LLM engine/provider (openai, azure, anthropic, etc.)"
    )
    model: str = Field(
        default="gpt-3.5-turbo",
        description="Model name/identifier"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM generation (0.0 for deterministic)"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens for LLM response"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional LLM parameters"
    )


class NeMoGenerationConfig(BaseModel):
    """Configuration for NeMo config generation.
    
    Controls how and when NeMo configuration files are generated.
    """
    
    auto_generate: bool = Field(
        default=True,
        description="Automatically generate NeMo configs when needed"
    )
    output_path: Optional[Path] = Field(
        default=None,
        description="Directory to store generated configs"
    )
    regenerate_on_change: bool = Field(
        default=True,
        description="Regenerate configs when configuration changes"
    )
    prompt_style: str = Field(
        default="simple",
        description="Prompt style for self-checks ('simple' or 'complex')"
    )


class PromptStyle(str, Enum):
    """Style of prompts for self-check tasks."""
    
    SIMPLE = "simple"  # Concise prompts for faster/cheaper evaluation
    COMPLEX = "complex"  # Detailed prompts for thorough checking


class NeMoInputRailConfig(BaseModel):
    """Configuration for NeMo input rails.
    
    Input rails validate user messages before they reach the LLM.
    """
    
    enabled: bool = Field(
        default=True,
        description="Enable input rails"
    )
    self_check_input: bool = Field(
        default=True,
        description="Enable self_check_input for jailbreak/injection detection"
    )


class NeMoOutputRailConfig(BaseModel):
    """Configuration for NeMo output rails.
    
    Output rails validate LLM responses before returning to users.
    """
    
    enabled: bool = Field(
        default=True,
        description="Enable output rails"
    )
    self_check_output: bool = Field(
        default=True,
        description="Enable self_check_output for response validation"
    )
    self_check_facts: bool = Field(
        default=False,
        description="Enable fact-checking rail (requires $relevant_chunks)"
    )
    self_check_hallucination: bool = Field(
        default=False,
        description="Enable hallucination detection rail"
    )


class NeMoDialogRailConfig(BaseModel):
    """Configuration for NeMo dialog rails.
    
    Dialog rails control conversation flow and require user-provided Colang files.
    Users must create their own .co files for use-case specific dialog management.
    """
    
    enabled: bool = Field(
        default=False,
        description="Enable dialog rails (requires user-provided Colang files)"
    )


class NeMoRailsConfig(BaseModel):
    """Combined rails configuration for NeMo.
    
    Groups all rail configurations (input, output, dialog) together.
    """
    
    input: NeMoInputRailConfig = Field(
        default_factory=NeMoInputRailConfig,
        description="Input rail configuration"
    )
    output: NeMoOutputRailConfig = Field(
        default_factory=NeMoOutputRailConfig,
        description="Output rail configuration"
    )
    dialog: NeMoDialogRailConfig = Field(
        default_factory=NeMoDialogRailConfig,
        description="Dialog rail configuration"
    )


class NeMoProviderConfig(BaseModel):
    """Complete configuration for the NeMo Guardrails provider.
    
    This is the top-level configuration model that combines all NeMo
    settings needed to run guardrails using NeMo's rails.
    
    Example:
        config = NeMoProviderConfig(
            llm=NeMoLLMConfig(model="gpt-4"),
            rails=NeMoRailsConfig(
                input=NeMoInputRailConfig(self_check_input=True),
                output=NeMoOutputRailConfig(self_check_output=True),
            ),
        )
    """
    
    # Basic settings
    enabled: bool = Field(
        default=True,
        description="Enable NeMo Guardrails provider"
    )
    
    # LLM configuration
    llm: NeMoLLMConfig = Field(
        default_factory=NeMoLLMConfig,
        description="LLM configuration for self-checks"
    )
    
    # Generation settings
    generation: NeMoGenerationConfig = Field(
        default_factory=NeMoGenerationConfig,
        description="Config generation settings"
    )
    
    # Rails configuration
    rails: NeMoRailsConfig = Field(
        default_factory=NeMoRailsConfig,
        description="Rails configuration"
    )
    
    # Streaming mode
    streaming: bool = Field(
        default=False,
        description="Enable streaming mode"
    )
    
    # Custom refusal message
    refusal_message: str = Field(
        default="I'm sorry, I can't respond to that.",
        description="Default message when input/output is blocked"
    )
    
    # Config path settings
    config_path: Optional[Path] = Field(
        default=None,
        description="Path to YAML configuration directory"
    )
    
    @classmethod
    def from_yaml_config(cls, nemo_config: Dict[str, Any]) -> "NeMoProviderConfig":
        """
        Create NeMoProviderConfig from a YAML nemo section.
        
        Args:
            nemo_config: Dictionary from YAML 'nemo' section
            
        Returns:
            NeMoProviderConfig instance
        """
        # Extract LLM config
        llm_dict = nemo_config.get("llm", {})
        llm = NeMoLLMConfig(
            engine=llm_dict.get("engine", "openai"),
            model=llm_dict.get("model", "gpt-3.5-turbo"),
            temperature=llm_dict.get("temperature", 0.0),
            max_tokens=llm_dict.get("max_tokens"),
        )
        
        # Extract rails config
        rails_dict = nemo_config.get("rails", {})
        
        input_dict = rails_dict.get("input", {})
        input_rails = NeMoInputRailConfig(
            enabled=input_dict.get("enabled", True),
            self_check_input=input_dict.get("self_check_input", True),
        )
        
        output_dict = rails_dict.get("output", {})
        output_rails = NeMoOutputRailConfig(
            enabled=output_dict.get("enabled", True),
            self_check_output=output_dict.get("self_check_output", True),
            self_check_facts=output_dict.get("self_check_facts", False),
            self_check_hallucination=output_dict.get("self_check_hallucination", False),
        )
        
        dialog_dict = rails_dict.get("dialog", {})
        dialog_rails = NeMoDialogRailConfig(
            enabled=dialog_dict.get("enabled", False),
        )
        
        # Extract generation config
        generation = NeMoGenerationConfig(
            output_path=Path(nemo_config["output_path"]) if nemo_config.get("output_path") else None,
            prompt_style=nemo_config.get("prompt_style", "simple"),
        )
        
        return cls(
            enabled=nemo_config.get("enabled", True),
            llm=llm,
            generation=generation,
            rails=NeMoRailsConfig(
                input=input_rails,
                output=output_rails,
                dialog=dialog_rails,
            ),
            streaming=nemo_config.get("streaming", False),
        )
