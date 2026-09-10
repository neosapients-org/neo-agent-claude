"""
Configuration Mapper for NeMo Guardrails.

This module maps Neo Guardrail Hub's YAML configuration to NeMo Colang
configuration structure. It reads from the 'nemo' section of the YAML
config to produce ColangConfig for generation.

IMPORTANT: NeMo Guardrails and LLM Guard are SEPARATE implementations:
- LLM Guard (providers/llm_guard/) - Uses LLM Guard library scanners
- NeMo Guardrails (providers/nemo/) - Uses NeMo's Colang flows and LLM-based self-checks

The NeMo-specific configuration is defined in the 'nemo' section of YAML:
- nemo.rails.input - NeMo input rails configuration
- nemo.rails.output - NeMo output rails configuration  
- nemo.rails.dialog - NeMo dialog/topical rails configuration

Both providers can coexist and provide different capabilities.
"""

from pathlib import Path
from typing import Any, Optional

from neo_guardrail_hub.core.config import ConfigLoader

from .generators.models import (
    ColangConfig,
    InputRailConfig,
    JailbreakDetectionConfig,
    LLMConfig,
    NeMoGuardrailsConfig,
    OutputRailConfig,
    TopicalRailConfig,
    WealthManagementDomainCheckConfig,
)


class ConfigToNeMoMapper:
    """
    Maps Neo Guardrail Hub YAML config to NeMo Colang configuration.
    
    This mapper reads the 'nemo' section of configuration files and produces
    NeMo-compatible configuration objects for Colang generation.
    
    NeMo's self_check_input and self_check_output are NeMo's native features
    that use LLM prompts (defined in prompts.yml) - they are NOT related to
    LLM Guard's scanners.
    
    Example:
        >>> mapper = ConfigToNeMoMapper(config_path="./configs")
        >>> colang_config = mapper.map_to_colang()  # Uses default.yaml
        >>> colang_config = mapper.map_to_colang(agent_id="wealth_advisor")
    """
    
    
    def __init__(self, config_path: str | Path = "./configs"):
        """
        Initialize the mapper.
        
        Args:
            config_path: Path to the configuration directory
        """
        self.config_path = Path(config_path)
        self.config_loader = ConfigLoader(config_path)
        
    def map_to_colang(
        self,
        agent_id: Optional[str] = None,
        config_override: Optional[dict[str, Any]] = None,
    ) -> ColangConfig:
        """
        Map YAML configuration to ColangConfig.
        
        Reads from the 'nemo.rails' section of the configuration.
        
        Args:
            agent_id: Optional agent ID for agent-specific config
            config_override: Optional runtime configuration overrides
            
        Returns:
            ColangConfig ready for Colang generation
        """
        # Load the configuration using existing loader
        config = self.config_loader.load(
            agent_id=agent_id,
            config_override=config_override,
        )
        
        return self._map_config(config)
    
    def map_to_nemo_config(
        self,
        agent_id: Optional[str] = None,
        config_override: Optional[dict[str, Any]] = None,
    ) -> NeMoGuardrailsConfig:
        """
        Map YAML configuration to full NeMo configuration.
        
        Args:
            agent_id: Optional agent ID for agent-specific config
            config_override: Optional runtime configuration overrides
            
        Returns:
            NeMoGuardrailsConfig for complete NeMo setup
        """
        config = self.config_loader.load(
            agent_id=agent_id,
            config_override=config_override,
        )
        
        colang_config = self._map_config(config)
        
        # Extract NeMo-specific settings from config
        nemo_section = config.get("nemo", {})
        llm_section = nemo_section.get("llm", {})
        
        # Build LLM config
        llm_config = LLMConfig(
            engine=llm_section.get("engine", "openai"),
            model=llm_section.get("model", "gpt-3.5-turbo"),
            temperature=llm_section.get("temperature", 0.0),
        )
        
        # Determine if rails are enabled based on guardrails section
        guardrails = config.get("guardrails", {})
        
        # Check if any NeMo checks are enabled in each layer
        enable_input_rails = self._has_enabled_nemo_checks(guardrails.get("input", {}))
        enable_output_rails = self._has_enabled_nemo_checks(guardrails.get("output", {}))
        enable_dialog_rails = self._has_enabled_nemo_checks(guardrails.get("context", {}))
        
        return NeMoGuardrailsConfig(
            name=agent_id or "default",
            llm=llm_config,
            colang=colang_config,
            enable_input_rails=enable_input_rails,
            enable_output_rails=enable_output_rails,
            enable_dialog_rails=enable_dialog_rails,
            streaming=nemo_section.get("streaming", False),
        )
    
    def _has_enabled_nemo_checks(self, layer_config: dict[str, Any]) -> bool:
        """Check if a layer has any enabled NeMo checks."""
        if not layer_config.get("enabled", True):
            return False
        
        checks = layer_config.get("checks", [])
        return any(
            c.get("provider") == "nemo" and c.get("enabled", True)
            for c in checks
        )
    
    def get_output_path(
        self,
        agent_id: Optional[str] = None,
        config_override: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Get the configured output path for NeMo files.
        
        Args:
            agent_id: Optional agent ID
            config_override: Optional configuration overrides
            
        Returns:
            Output path from config or None to use default
        """
        config = self.config_loader.load(
            agent_id=agent_id,
            config_override=config_override,
        )
        
        nemo_section = config.get("nemo", {})
        return nemo_section.get("output_path")
    
    def _map_config(self, config: dict[str, Any]) -> ColangConfig:
        """
        Internal method to map config dict to ColangConfig.
        
        Reads from the 'guardrails' section which contains all provider checks
        unified under input/output/context layers.
        
        Args:
            config: Loaded configuration dictionary
            
        Returns:
            ColangConfig object
        """
        # Get guardrails configuration
        guardrails = config.get("guardrails", {})
        
        # Map input rails from guardrails.input.checks (NeMo provider)
        input_rails = self._map_input_rails(guardrails.get("input", {}))
        
        # Map output rails from guardrails.output.checks (NeMo provider)
        output_rails = self._map_output_rails(guardrails.get("output", {}))
        
        # Map context rails from guardrails.context.checks (NeMo provider)
        topical_rails = self._map_context_rails(guardrails.get("context", {}))
        
        # Map wealth management domain check from guardrails.context.checks (NeMo provider)
        wealth_management_domain_check = self._map_wealth_management_domain_check(guardrails.get("context", {}))
        
        return ColangConfig(
            input_rails=input_rails,
            output_rails=output_rails,
            topical_rails=topical_rails,
            wealth_management_domain_check=wealth_management_domain_check,
        )
    
    def _map_input_rails(self, input_config: dict[str, Any]) -> Optional[InputRailConfig]:
        """
        Map input rails config to InputRailConfig.
        
        Reads from guardrails.input.checks where provider == 'nemo'.
        Looks for nemo_self_check_input and nemo_jailbreak_detection_heuristics check types.
        """
        if not input_config.get("enabled", True):
            return None
        
        # Find NeMo checks from the checks array
        checks = input_config.get("checks", [])
        nemo_checks = [c for c in checks if c.get("provider") == "nemo" and c.get("enabled", True)]
        
        # Check if nemo_self_check_input is enabled
        self_check_input = any(c.get("type") == "nemo_self_check_input" for c in nemo_checks)
        
        # Check if jailbreak detection heuristics is enabled
        jailbreak_config = None
        for check in nemo_checks:
            if check.get("type") == "nemo_jailbreak_detection_heuristics":
                config = check.get("config", {})
                jailbreak_config = JailbreakDetectionConfig(
                    enabled=True,
                    length_per_perplexity_threshold=config.get("length_per_perplexity_threshold", 89.79),
                    prefix_suffix_perplexity_threshold=config.get("prefix_suffix_perplexity_threshold", 1845.65),
                    server_endpoint=config.get("server_endpoint"),
                )
                break
        
        if not self_check_input and not jailbreak_config:
            return None
        
        return InputRailConfig(
            self_check_input=self_check_input,
            jailbreak_detection_heuristics=jailbreak_config,
        )
    
    def _map_output_rails(self, output_config: dict[str, Any]) -> Optional[OutputRailConfig]:
        """
        Map output rails config to OutputRailConfig.
        
        Reads from guardrails.output.checks where provider == 'nemo'.
        Looks for nemo_self_check_output, nemo_self_check_facts, nemo_self_check_hallucination.
        """
        if not output_config.get("enabled", True):
            return None
        
        # Find NeMo checks from the checks array
        checks = output_config.get("checks", [])
        nemo_checks = [c for c in checks if c.get("provider") == "nemo" and c.get("enabled", True)]
        
        # Check which output rails are enabled
        self_check_output = any(c.get("type") == "nemo_self_check_output" for c in nemo_checks)
        self_check_facts = any(c.get("type") == "nemo_self_check_facts" for c in nemo_checks)
        self_check_hallucination = any(c.get("type") == "nemo_self_check_hallucination" for c in nemo_checks)
        
        if not (self_check_output or self_check_facts or self_check_hallucination):
            return None
        
        return OutputRailConfig(
            self_check_output=self_check_output,
            self_check_facts=self_check_facts,
            self_check_hallucination=self_check_hallucination,
        )
    
    def _map_context_rails(self, context_config: dict[str, Any]) -> Optional[TopicalRailConfig]:
        """
        Map context rails config to TopicalRailConfig.
        
        Reads from guardrails.context.checks where provider == 'nemo'.
        Looks for nemo_topical_rail check type and extracts its config.
        
        Supports:
        - enabled: Enable/disable context rails
        - allowed_topics: List of allowed topics (whitelist)
        - allow_only_listed_topics: Only allow whitelisted topics
        - disallowed_topics: List of topic names to block (from topic library)
        - custom_topics: Custom topic definitions
        - off_topic_response: Default response for off-topic requests
        """
        if not context_config.get("enabled", False):
            return None
        
        # Find NeMo topical rail check from the checks array
        checks = context_config.get("checks", [])
        for check in checks:
            if (check.get("provider") == "nemo" and 
                check.get("type") == "nemo_topical_rail" and 
                check.get("enabled", True)):
                
                # Extract config from the check
                rail_config = check.get("config", {})
                
                return TopicalRailConfig(
                    enabled=True,
                    allowed_topics=rail_config.get("allowed_topics", []),
                    allow_only_listed_topics=rail_config.get("allow_only_listed_topics", False),
                    disallowed_topics=rail_config.get("disallowed_topics", []),
                    custom_topics=rail_config.get("custom_topics", {}),
                    off_topic_response=rail_config.get(
                        "off_topic_response",
                        "I'm not able to discuss that topic. Is there something else I can help with?"
                    ),
                )
        
        return None
    
    def _map_wealth_management_domain_check(self, context_config: dict[str, Any]) -> Optional[WealthManagementDomainCheckConfig]:
        """
        Map wealth management domain check config to WealthManagementDomainCheckConfig.
        
        Reads from guardrails.context.checks where type == 'wealth_management_domain_check'.
        """
        if not context_config.get("enabled", False):
            return None
        
        # Find wealth_management_domain_check from the checks array
        checks = context_config.get("checks", [])
        for check in checks:
            if (check.get("provider") == "nemo" and 
                check.get("type") == "wealth_management_domain_check" and 
                check.get("enabled", False)):
                
                return WealthManagementDomainCheckConfig(
                    enabled=True,
                )
        
        return None


def map_yaml_to_colang(
    config_path: str | Path = "./configs",
    agent_id: Optional[str] = None,
) -> ColangConfig:
    """
    Convenience function to map YAML config to ColangConfig.
    
    Args:
        config_path: Path to configuration directory
        agent_id: Optional agent ID for agent-specific config
        
    Returns:
        ColangConfig ready for generation
    """
    mapper = ConfigToNeMoMapper(config_path)
    return mapper.map_to_colang(agent_id)
