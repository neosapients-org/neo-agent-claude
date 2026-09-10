"""
Pydantic models for NeMo Guardrails configuration.

This module defines the data models used by the generators to create
NeMo Guardrails configuration files (Colang, config.yml, prompts.yml).
"""

from typing import Optional

from pydantic import BaseModel, Field


class JailbreakDetectionConfig(BaseModel):
    """Configuration for jailbreak detection heuristics."""
    
    enabled: bool = Field(
        default=False,
        description="Enable jailbreak detection heuristics"
    )
    length_per_perplexity_threshold: float = Field(
        default=89.79,
        description="Threshold for length per perplexity heuristic (detects long, garbled prompts)"
    )
    prefix_suffix_perplexity_threshold: float = Field(
        default=1845.65,
        description="Threshold for prefix/suffix perplexity heuristic (detects adversarial suffixes)"
    )
    server_endpoint: Optional[str] = Field(
        default=None,
        description="Optional server endpoint for jailbreak detection (for production use)"
    )


class InputRailConfig(BaseModel):
    """Configuration for input rails."""
    
    self_check_input: bool = Field(
        default=False,
        description="Enable self_check_input rail for jailbreak/injection detection"
    )
    jailbreak_detection_heuristics: Optional[JailbreakDetectionConfig] = Field(
        default=None,
        description="Configuration for perplexity-based jailbreak detection heuristics"
    )
    
    
class OutputRailConfig(BaseModel):
    """Configuration for output rails."""
    
    self_check_output: bool = Field(
        default=False,
        description="Enable self_check_output rail for response validation"
    )
    self_check_facts: bool = Field(
        default=False,
        description="Enable fact checking rail"
    )
    self_check_hallucination: bool = Field(
        default=False,
        description="Enable hallucination detection rail"
    )
    

class TopicalRailConfig(BaseModel):
    """Configuration for topical/dialog rails.
    
    Supports auto-generation of dialog.co files with allowed/blocked topics.
    Topics can be specified by name (using the built-in topic library)
    or with custom definitions.
    """
    
    enabled: bool = Field(
        default=True,
        description="Enable dialog rails"
    )
    
    # List of allowed topics (whitelist) - only these topics can be discussed
    allowed_topics: list[str] = Field(
        default_factory=list,
        description="List of allowed topic names (whitelist mode)"
    )
    
    # Whether to allow only the listed topics (whitelist mode)
    allow_only_listed_topics: bool = Field(
        default=False,
        description="If True, only allowed_topics can be discussed"
    )
    
    # List of topic names to block (e.g., ["politics", "medical_advice"])
    # These will be looked up in the topic library
    disallowed_topics: list[str] = Field(
        default_factory=list,
        description="List of topic names to block (from topic library)"
    )
    
    # Custom topic definitions for domain-specific blocking
    # Format: {"topic_name": {"examples": [...], "response": "..."}}
    custom_topics: dict[str, dict] = Field(
        default_factory=dict,
        description="Custom topic definitions with examples and responses"
    )
    
    # Off-topic refusal message
    off_topic_response: str = Field(
        default="I'm not able to discuss that topic. Is there something else I can help with?",
        description="Default response for off-topic requests"
    )


class WealthManagementDomainCheckConfig(BaseModel):
    """Configuration for wealth management domain check (prompt-based).
    
    Specific prompt-based domain validation for wealth management domain.
    """
    
    enabled: bool = Field(
        default=False,
        description="Enable wealth management domain check"
    )


class LLMConfig(BaseModel):
    """Configuration for the LLM used by NeMo Guardrails."""
    
    model: str = Field(
        default="gpt-3.5-turbo",
        description="Model name/identifier"
    )
    engine: str = Field(
        default="openai",
        description="LLM engine/provider"
    )
    temperature: float = Field(
        default=0.0,
        description="Temperature for LLM generation"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens for LLM response"
    )


class ColangConfig(BaseModel):
    """
    Complete configuration for Colang generation.
    
    This model represents the simplified YAML configuration that users
    provide to generate Colang 2.x files.
    """
    
    # Rail configurations
    input_rails: Optional[InputRailConfig] = Field(
        default=None,
        description="Input rail configuration"
    )
    output_rails: Optional[OutputRailConfig] = Field(
        default=None,
        description="Output rail configuration"
    )
    topical_rails: Optional[TopicalRailConfig] = Field(
        default=None,
        description="Topical/dialog rail configuration"
    )
    wealth_management_domain_check: Optional[WealthManagementDomainCheckConfig] = Field(
        default=None,
        description="Wealth management domain check configuration (prompt-based)"
    )
    
    # Custom messages
    custom_user_messages: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Custom user message definitions: name -> examples"
    )
    custom_bot_messages: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Custom bot message definitions: name -> responses"
    )
    
    # General settings
    refusal_message: Optional[str] = Field(
        default=None,
        description="Custom refusal message for blocked content"
    )


class NeMoGuardrailsConfig(BaseModel):
    """
    Complete NeMo Guardrails configuration.
    
    This is the top-level configuration model that includes all settings
    needed to generate a complete NeMo Guardrails configuration directory.
    """
    
    # Basic settings
    name: str = Field(
        default="neo_guardrail_hub",
        description="Name of the guardrails configuration"
    )
    
    # LLM configuration
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM configuration"
    )
    
    # Colang configuration (used for .co file generation)
    colang: ColangConfig = Field(
        default_factory=ColangConfig,
        description="Colang generation configuration"
    )
    
    # Enabled rails
    enable_input_rails: bool = Field(
        default=True,
        description="Enable input rails"
    )
    enable_output_rails: bool = Field(
        default=True,
        description="Enable output rails"
    )
    enable_dialog_rails: bool = Field(
        default=False,
        description="Enable dialog rails"
    )
    
    # Advanced settings
    streaming: bool = Field(
        default=False,
        description="Enable streaming mode"
    )
    lowest_temperature: float = Field(
        default=0.0,
        description="Lowest temperature for guardrail LLM calls"
    )


class TopicLibraryEntry(BaseModel):
    """A pre-built topic with examples for the topic library."""
    
    name: str = Field(
        description="Topic name (e.g., 'politics', 'medical_advice')"
    )
    category: str = Field(
        default="general",
        description="Topic category for organization"
    )
    description: str = Field(
        default="",
        description="Description of what this topic covers"
    )
    user_examples: list[str] = Field(
        default_factory=list,
        description="Example user utterances about this topic"
    )
    bot_responses: list[str] = Field(
        default_factory=list,
        description="Example bot responses for this topic"
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords associated with this topic"
    )


class TopicLibrary(BaseModel):
    """Collection of pre-built topics for topical rails."""
    
    version: str = Field(
        default="1.0.0",
        description="Library version"
    )
    topics: list[TopicLibraryEntry] = Field(
        default_factory=list,
        description="List of topic entries"
    )
    
    def get_topic(self, name: str) -> Optional[TopicLibraryEntry]:
        """Get a topic by name."""
        for topic in self.topics:
            if topic.name.lower() == name.lower():
                return topic
        return None
    
    def get_topics_by_category(self, category: str) -> list[TopicLibraryEntry]:
        """Get all topics in a category."""
        return [t for t in self.topics if t.category.lower() == category.lower()]
