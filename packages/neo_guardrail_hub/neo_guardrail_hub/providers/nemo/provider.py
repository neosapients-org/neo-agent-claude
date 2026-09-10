"""
NeMo Guardrails Provider for Neo Guardrail Hub.

This module provides the NeMoProvider class that integrates NVIDIA NeMo Guardrails
with Neo Guardrail Hub. It enables LLM-based guardrails using NeMo's rails system.

The provider handles:
1. Auto-generation of NeMo config files (config.yml, rails.co, prompts.yml)
2. Loading and initializing NeMo's LLMRails
3. Executing input rails (self_check_input) for jailbreak/injection detection
4. Executing output rails (self_check_output) for response validation

Usage:
    from neo_guardrail_hub.providers.nemo import NeMoProvider
    
    # Create provider
    provider = NeMoProvider(config_path="./configs")
    await provider.initialize()
    
    # Check input using NeMo's self_check_input
    result = await provider.check_input("user message here")
    
    # Or get a guardrail instance
    guardrail = provider.get_guardrail("self_check_input")
    result = await guardrail.check("user message")
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, Type, TYPE_CHECKING

from ...core.exceptions import ProviderError
from ...core.interfaces import BaseGuardrail
from ...core.models import GuardrailContext
from ...utils.logging import get_logger
from ..base import ProviderBase
from .config_models import NeMoProviderConfig, PromptStyle
from .nemo_generator import NeMoConfigGenerator


class NeMoProvider(ProviderBase):
    """
    Provider for NeMo Guardrails.
    
    This provider integrates NVIDIA NeMo Guardrails with Neo Guardrail Hub,
    enabling LLM-based input and output validation through NeMo's rails system.
    
    Features:
    - Auto-generates NeMo config files from default.yaml
    - Uses NeMo's self_check_input for jailbreak/injection detection
    - Uses NeMo's self_check_output for response validation
    - Supports fact-checking and hallucination detection
    
    Supported Guardrails:
        - self_check_input: LLM-based input validation
        - self_check_output: LLM-based output validation (Sprint 3B.2)
        - self_check_facts: Fact-checking rail (Sprint 3C)
        - self_check_hallucination: Hallucination detection (Sprint 3C)
    
    Example:
        provider = NeMoProvider(config_path="./configs")
        await provider.initialize()
        
        # Check user input
        result = await provider.check_input("Hello, how are you?")
        if result["allowed"]:
            print("Input is safe")
        else:
            print("Input blocked:", result["message"])
    """
    
    name = "nemo"
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str | Path] = None,
        nemo_config: Optional[NeMoProviderConfig] = None,
        agent_id: Optional[str] = None,
        output_path: Optional[str | Path] = None,
        prompt_style: str = "simple",
    ) -> None:
        """
        Initialize the NeMo Guardrails provider.
        
        Args:
            config: Dict-based configuration (used by orchestrator's create_provider)
            config_path: Path to the Neo Guardrail Hub configs directory
                        (containing default.yaml or agent configs)
            nemo_config: Pre-configured NeMoProviderConfig (alternative to config_path)
            agent_id: Agent ID for agent-specific configuration
            output_path: Directory for generated NeMo config files
            prompt_style: Prompt style for self-checks ("simple" or "complex")
        """
        super().__init__()
        
        # Store the raw config for override purposes
        self._config_override: Optional[Dict[str, Any]] = None
        
        # Handle dict-based config (from orchestrator)
        if config:
            config_path = config.get("config_path", config_path)
            nemo_config = config.get("nemo_config", nemo_config)
            agent_id = config.get("agent_id", agent_id)
            output_path = config.get("output_path", output_path)
            prompt_style = config.get("prompt_style", prompt_style)
            # Store the full config for override
            self._config_override = config
        
        self.config_path = Path(config_path) if config_path else Path("./configs")
        # Handle nemo_config being a dict (from YAML) or NeMoProviderConfig
        if isinstance(nemo_config, dict):
            self._nemo_config_dict = nemo_config  # Store raw dict for lazy loading
            self.nemo_config = None
        else:
            self._nemo_config_dict = None
            self.nemo_config = nemo_config
        self.agent_id = agent_id
        self.output_path = Path(output_path) if output_path else None
        self.prompt_style = prompt_style
        
        self.logger = get_logger(self.__class__.__name__)
        
        # NeMo components (initialized lazily)
        self._rails = None
        self._rails_config = None
        self._generated_config_path: Optional[Path] = None
        
        # Generator for creating NeMo configs
        self._generator: Optional[NeMoConfigGenerator] = None
        
        # Register guardrails
        self._register_guardrails()
    
    def _register_guardrails(self) -> None:
        """Register available NeMo guardrails."""
        # Import from the correct guardrails location
        from ...guardrails.input.nemo_self_check_input import NeMoSelfCheckInputGuardrail
        from ...guardrails.input.nemo_jailbreak_detection_heuristics import NeMoJailbreakDetectionHeuristicsGuardrail
        from ...guardrails.output.nemo_self_check_output import NeMoSelfCheckOutputGuardrail
        from ...guardrails.output.nemo_self_check_facts import NeMoSelfCheckFactsGuardrail
        from ...guardrails.output.nemo_self_check_hallucination import NeMoSelfCheckHallucinationGuardrail
        from ...guardrails.context.nemo_topical_rail import NeMoTopicalRailGuardrail
        from ...guardrails.context.wealth_management_domain_check import WealthManagementDomainCheckGuardrail
        
        self._guardrails: Dict[str, Type[BaseGuardrail]] = {
            "nemo_self_check_input": NeMoSelfCheckInputGuardrail,
            "nemo_jailbreak_detection_heuristics": NeMoJailbreakDetectionHeuristicsGuardrail,
            "nemo_self_check_output": NeMoSelfCheckOutputGuardrail,
            "nemo_self_check_facts": NeMoSelfCheckFactsGuardrail,
            "nemo_self_check_hallucination": NeMoSelfCheckHallucinationGuardrail,
            "nemo_topical_rail": NeMoTopicalRailGuardrail,
            "wealth_management_domain_check": WealthManagementDomainCheckGuardrail,
        }
    
    async def initialize(self) -> None:
        """
        Initialize the NeMo provider.
        
        This method:
        1. Generates NeMo config files from YAML configuration
        2. Loads NeMo's RailsConfig from generated files
        3. Creates the LLMRails instance for executing rails
        
        Raises:
            ProviderError: If NeMo Guardrails is not installed or initialization fails
        """
        if self._initialized:
            return
            
        self.logger.info(
            "nemo_provider_initializing",
            config_path=str(self.config_path),
            agent_id=self.agent_id,
        )
        
        try:
            # Step 1: Generate NeMo configuration files
            await self._generate_nemo_configs()
            
            # Step 2: Load NeMo Rails
            await self._load_nemo_rails()
            
            self._initialized = True
            
            self.logger.info(
                "nemo_provider_initialized",
                config_path=str(self._generated_config_path),
            )
            
        except ImportError as e:
            error_msg = (
                "NeMo Guardrails not installed. "
                "Install with: pip install neo-guardrail-hub[nemo]"
            )
            self.logger.error("nemo_import_error", error=str(e))
            raise ProviderError(
                error_msg,
                provider_name=self.name,
            ) from e
            
        except Exception as e:
            self.logger.error("nemo_init_error", error=str(e))
            raise ProviderError(
                f"Failed to initialize NeMo provider: {str(e)}",
                provider_name=self.name,
            ) from e
    
    async def _generate_nemo_configs(self) -> None:
        """Generate NeMo configuration files from YAML config."""
        self._generator = NeMoConfigGenerator(
            config_path=self.config_path,
            output_base_path=self.output_path.parent if self.output_path else None,
        )
        
        # Generate all config files, passing any config overrides
        self._generated_config_path = self._generator.generate(
            agent_id=self.agent_id,
            output_path=self.output_path,
            prompt_style=self.prompt_style,
            config_override=self._config_override,
        )
        
        self.logger.info(
            "nemo_configs_generated",
            output_path=str(self._generated_config_path),
        )
    
    async def _load_nemo_rails(self) -> None:
        """Load NeMo Rails from generated config directory, using cache if available."""
        try:
            from nemoguardrails import RailsConfig
        except ImportError:
            raise ImportError(
                "nemoguardrails package not installed. "
                "Install with: pip install neo-guardrail-hub[nemo]"
            )
        
        # Try to use cached Rails first
        try:
            from ...api.caching import get_rails_cache
            
            cache = get_rails_cache()
            cache_key = self.agent_id or "default"
            
            # Get or create Rails from cache
            self._rails = await cache.get_or_create(
                key=cache_key,
                config_path=self._generated_config_path,
            )
            
            # Also load the config for reference
            self._rails_config = RailsConfig.from_path(str(self._generated_config_path))
            
            self.logger.info(
                "nemo_rails_loaded_from_cache",
                cache_key=cache_key,
                config_path=str(self._generated_config_path),
            )
            
        except ImportError:
            # Fallback: Create Rails directly without caching
            from nemoguardrails import LLMRails
            
            # Load Rails config from generated directory
            self._rails_config = RailsConfig.from_path(str(self._generated_config_path))
            
            # Create LLMRails instance
            self._rails = LLMRails(self._rails_config)
            
            self.logger.info(
                "nemo_rails_loaded",
                config_path=str(self._generated_config_path),
            )
    
    @property
    def rails(self):
        """Get the NeMo LLMRails instance.
        
        Returns:
            LLMRails instance
            
        Raises:
            ProviderError: If provider is not initialized
        """
        if self._rails is None:
            raise ProviderError(
                "NeMo provider not initialized. Call initialize() first.",
                provider_name=self.name,
            )
        return self._rails
    
    @property
    def generated_config_path(self) -> Optional[Path]:
        """Get the path to generated NeMo config files."""
        return self._generated_config_path
    
    def _context_to_dict(self, context: Any) -> Optional[Dict[str, Any]]:
        """
        Convert context (GuardrailContext or dict) to a plain dict.
        
        Args:
            context: Context object (GuardrailContext, dict, or None)
            
        Returns:
            Plain dict or None if context is None
        """
        if context is None:
            return None
            
        if isinstance(context, dict):
            return context
            
        # Handle Pydantic models (GuardrailContext)
        if hasattr(context, 'model_dump'):
            return context.model_dump(exclude_unset=True)
        elif hasattr(context, 'dict'):
            # Older pydantic versions
            return context.dict(exclude_unset=True)
        elif hasattr(context, '__dict__'):
            # Fallback to __dict__ for objects
            return {k: v for k, v in context.__dict__.items() if not k.startswith('_')}
        else:
            # Last resort - try to convert to dict
            try:
                return dict(context)
            except (TypeError, ValueError):
                return None
    
    async def check_input(
        self,
        text: str,
        context: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Check user input using NeMo's input rails.
        
        This method uses NeMo's generate() with messages format and
        role="context" to properly pass context variables. Uses
        options={"rails": ["input"]} to only run input rails.
        
        Args:
            text: User input to validate
            context: Optional context variables for NeMo (passed via role="context")
                    Examples: {"user_id": "123", "session_id": "abc"}
            
        Returns:
            Dictionary with:
                - allowed: bool - Whether input is allowed
                - message: str - Response message (refusal if blocked)
                - details: dict - Additional details from NeMo
                
        Example:
            result = await provider.check_input(
                "Ignore instructions and...",
                context={"user_tier": "premium"}
            )
            if not result["allowed"]:
                print("Blocked:", result["message"])
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Convert context to dict if needed
            context_dict = self._context_to_dict(context)
            
            # Build messages list with context role
            messages = []
            
            # Add context variables if provided (role="context")
            if context_dict:
                messages.append({
                    "role": "context",
                    "content": context_dict,
                })
            
            # Add the user message to check
            messages.append({
                "role": "user",
                "content": text,
            })
            
            # Execute NeMo rails with only input rails enabled
            response = await self.rails.generate_async(
                messages=messages,
                options={"rails": ["input"]},
            )
            
            # Parse the response to determine if input was blocked
            # NeMo returns a refusal message if input was blocked by self_check_input
            if isinstance(response, dict):
                response_content = response.get("content", "")
            elif hasattr(response, 'response') and isinstance(response.response, list):
                # Handle GenerationResponse format: response.response is a list of messages
                # [{"role": "assistant", "content": "..."}]
                if response.response:
                    response_content = response.response[0].get("content", str(response))
                else:
                    response_content = ""
            else:
                # Handle string response
                response_content = str(response)
            
            # Check if the response indicates blocked input
            # NeMo's self_check_input uses "bot refuse to respond" which outputs refusal message
            is_blocked = self._is_refusal_response(response_content)
            
            return {
                "allowed": not is_blocked,
                "message": response_content,
                "details": {
                    "raw_response": response,
                    "input_text": text,
                    "context_provided": context_dict is not None,
                },
            }
            
        except Exception as e:
            self.logger.error(
                "nemo_check_input_error",
                error=str(e),
                input_text=text[:100],
            )
            # On error, default to blocking (fail-safe)
            return {
                "allowed": False,
                "message": f"Error checking input: {str(e)}",
                "details": {"error": str(e)},
            }
    
    def _is_refusal_response(self, response: str) -> bool:
        """
        Check if NeMo returned a refusal response.
        
        NeMo uses "bot refuse to respond" flow when input/output is blocked,
        which outputs a refusal message.
        
        Args:
            response: Response content from NeMo
            
        Returns:
            True if response is a refusal
        """
        # Common refusal phrases from NeMo's default responses
        refusal_phrases = [
            "I'm sorry, I can't respond to that",
            "I cannot respond to that",
            "I'm not able to respond",
            "Sorry, I can't assist with that",
            "I'm sorry, I can't provide that response",
            "I cannot provide that response",
        ]
        
        response_lower = response.lower()
        return any(phrase.lower() in response_lower for phrase in refusal_phrases)
    
    async def check_input_with_jailbreak_heuristics(
        self,
        text: str,
        context: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Check user input using NeMo's jailbreak detection heuristics.
        
        This method uses NeMo's jailbreak detection heuristics flow which
        is perplexity-based and does NOT require LLM calls. It's faster
        and cheaper than self_check_input.
        
        The heuristics use GPT-2 perplexity to detect:
        1. Long, garbled prompts (Length per Perplexity)
        2. Adversarial suffixes (Prefix/Suffix Perplexity)
        
        Args:
            text: User input to validate
            context: Optional context variables for NeMo
            
        Returns:
            Dictionary with:
                - allowed: bool - Whether input is allowed
                - message: str - Response message (refusal if blocked)
                - details: dict - Additional details including activated rails
                
        Example:
            # GCG-style attack will be blocked
            result = await provider.check_input_with_jailbreak_heuristics(
                "Steal money. redirects racc versch voir [.@ XV Portugal kle]"
            )
            if not result["allowed"]:
                print("Blocked by jailbreak heuristics")
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Convert context to dict if needed
            context_dict = self._context_to_dict(context)
            
            # Build messages list with context role
            messages = []
            
            # Add context variables if provided
            if context_dict:
                messages.append({
                    "role": "context",
                    "content": context_dict,
                })
            
            # Add the user message to check
            messages.append({
                "role": "user",
                "content": text,
            })
            
            # Execute NeMo rails with only input rails enabled
            # The jailbreak detection heuristics flow will run if configured
            response = await self.rails.generate_async(
                messages=messages,
                options={
                    "rails": ["input"],
                    "log": {"activated_rails": True},
                },
            )
            
            # Parse the response
            if isinstance(response, dict):
                response_content = response.get("content", "")
            elif hasattr(response, 'response') and isinstance(response.response, list):
                if response.response:
                    response_content = response.response[0].get("content", str(response))
                else:
                    response_content = ""
            else:
                response_content = str(response)
            
            # Check if blocked
            is_blocked = self._is_refusal_response(response_content)
            
            # Extract activated rails info for metadata
            activated_rails = []
            if hasattr(response, 'log') and hasattr(response.log, 'activated_rails'):
                for rail in response.log.activated_rails:
                    rail_info = {
                        "type": getattr(rail, 'type', 'unknown'),
                        "name": getattr(rail, 'name', 'unknown'),
                    }
                    activated_rails.append(rail_info)
            
            # Check if jailbreak heuristics specifically caught it
            jailbreak_triggered = any(
                r.get("name") == "jailbreak detection heuristics" 
                for r in activated_rails
            )
            
            return {
                "allowed": not is_blocked,
                "message": response_content,
                "details": {
                    "raw_response": response,
                    "input_text": text,
                    "context_provided": context_dict is not None,
                    "activated_rails": activated_rails,
                    "jailbreak_heuristics_triggered": jailbreak_triggered,
                    "heuristic": "jailbreak_detection" if jailbreak_triggered else None,
                },
            }
            
        except Exception as e:
            self.logger.error(
                "nemo_check_input_jailbreak_error",
                error=str(e),
                input_text=text[:100],
            )
            # On error, default to blocking (fail-safe)
            return {
                "allowed": False,
                "message": f"Error checking input with jailbreak heuristics: {str(e)}",
                "details": {"error": str(e)},
            }

    async def check_output(
        self,
        text: str,
        context: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Check bot output using NeMo's output rails.
        
        This method uses NeMo's generate() with messages format and
        role="context" to properly pass context variables including the
        bot_message to be judged. Uses options={"rails": ["output"]} 
        to only run output rails.
        
        Args:
            text: Bot response to validate
            context: Optional context variables for NeMo. Can include:
                - user_input: Original user query (recommended)
                - relevant_chunks: Evidence for fact-checking
                - Any other context variables needed by rails
            
        Returns:
            Dictionary with:
                - allowed: bool - Whether output is allowed
                - message: str - Response message (refusal if blocked)
                - details: dict - Additional details from NeMo
                
        Example:
            result = await provider.check_output(
                "Here's how to hack...",
                context={
                    "user_input": "How do I improve security?",
                    "relevant_chunks": "Security best practices..."
                }
            )
            if not result["allowed"]:
                print("Blocked:", result["message"])
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Convert context to dict if needed
            context_dict = self._context_to_dict(context)
            
            # Build context content with bot_message for the rail to judge
            context_content = {
                "bot_message": text,  # The rail judges this variable
            }
            
            # Add any additional context from caller
            if context_dict:
                # Add user_input if provided
                if "user_input" in context_dict:
                    context_content["user_message"] = context_dict["user_input"]
                
                # Add any other context variables
                for key, value in context_dict.items():
                    if key not in ["user_input"]:
                        context_content[key] = value
            
            # Build messages list with context role
            messages = [
                {
                    "role": "context",
                    "content": context_content,
                },
                {
                    "role": "user",
                    "content": context_dict.get("user_input", "Please respond.") if context_dict else "Please respond.",
                },
                {
                    "role": "assistant",
                    "content": text,
                },
            ]
            
            # Execute NeMo rails with only output rails enabled
            response = await self.rails.generate_async(
                messages=messages,
                options={"rails": ["output"]},
            )
            
            # Parse the response
            if isinstance(response, dict):
                response_content = response.get("content", "")
            elif hasattr(response, 'response') and isinstance(response.response, list):
                # Handle GenerationResponse format: response.response is a list of messages
                if response.response:
                    response_content = response.response[0].get("content", str(response))
                else:
                    response_content = ""
            else:
                response_content = str(response)
            
            # Check if the response indicates blocked output
            is_blocked = self._is_refusal_response(response_content)
            
            # If NeMo replaced the output with a refusal, it was blocked
            if not is_blocked and response_content != text:
                is_blocked = self._is_refusal_response(response_content)
            
            return {
                "allowed": not is_blocked,
                "message": response_content if is_blocked else text,
                "details": {
                    "raw_response": response,
                    "output_text": text,
                    "checked_response": response_content,
                    "context_provided": context is not None,
                },
            }
        except Exception as e:
            self.logger.error(
                "nemo_check_output_error",
                error=str(e),
                output_text=text[:100],
            )
            return {
                "allowed": False,
                "message": f"Error checking output: {str(e)}",
                "details": {"error": str(e)},
            }
    
    async def check_facts(
        self,
        response: str,
        evidence: str | list,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Check if bot response is grounded in provided evidence.
        
        This method uses NeMo's self_check_facts flow with messages format to verify
        that the response is factually supported by the provided evidence.
        Designed for RAG systems where responses should be grounded in retrieved documents.
        
        The check works by:
        1. Passing evidence via role="context" with relevant_chunks key
        2. Using the self check facts flow which calls execute self_check_facts
        3. Getting accuracy from output_vars to determine if grounded
        
        Args:
            response: Bot response to validate
            evidence: Evidence/context to check against (string or list of chunks)
            context: Optional additional context variables (dict or GuardrailContext)
            
        Returns:
            Dictionary with:
                - accuracy: float - Score between 0.0 (inaccurate) and 1.0 (accurate)
                - grounded: bool - Whether response is grounded in evidence
                - message: str - Human-readable result message
                - details: dict - Additional details
                
        Example:
            result = await provider.check_facts(
                response="Paris is the capital of France.",
                evidence="France is a country in Europe. Its capital is Paris."
            )
            if result["grounded"]:
                print("Response is factually accurate")
            else:
                print(f"Accuracy: {result['accuracy']:.2f}")
        
        Reference:
            https://docs.nvidia.com/nemo/guardrails/latest/getting-started/7-rag/README.html
            https://docs.nvidia.com/nemo/guardrails/latest/user-guides/guardrails-library.html#fact-checking
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Handle evidence as list of chunks - join with newline
            if isinstance(evidence, list):
                evidence_text = "\n".join(str(chunk) for chunk in evidence)
            else:
                evidence_text = str(evidence)
            
            # Build context content with relevant_chunks for the fact check rail
            context_content = {
                "relevant_chunks": evidence_text,
            }
            
            # Add any additional context from caller - handle both dict and GuardrailContext
            if context:
                # Convert GuardrailContext to dict if needed
                context_dict = self._context_to_dict(context)
                # If it's a GuardrailContext, extract metadata
                if context_dict and 'metadata' in context_dict:
                    context_dict = context_dict.get('metadata', {})
                # Add non-evidence keys to context content
                if context_dict:
                    for key, value in context_dict.items():
                        if key not in ["relevant_chunks", "evidence"]:
                            context_content[key] = value
            
            # Build messages list with context role
            messages = [
                {
                    "role": "context",
                    "content": context_content,
                },
                {
                    "role": "user",
                    "content": "Please verify this information.",
                },
                {
                    "role": "assistant",
                    "content": response,  # The response to fact-check
                },
            ]
            
            # Execute with output_vars to get accuracy score
            result = await self.rails.generate_async(
                messages=messages,
                options={"output_vars": True, "rails": ["output"]},
            )
            
            # Get accuracy from output_data
            accuracy = 0.0
            if hasattr(result, 'output_data') and result.output_data:
                accuracy = result.output_data.get("accuracy", 0.0)
                if accuracy is None:
                    accuracy = 0.0
            
            # Determine if grounded based on accuracy threshold
            grounded = accuracy >= 0.5
            
            # Parse response content
            if hasattr(result, 'response') and isinstance(result.response, list):
                response_content = result.response[0].get("content", "") if result.response else ""
            elif isinstance(result, dict):
                response_content = result.get("content", "")
            else:
                response_content = str(result)
            
            # Check if the response indicates not grounded
            is_blocked = self._is_refusal_response(response_content) or "cannot verify" in response_content.lower()
            if is_blocked:
                grounded = False
                accuracy = min(accuracy, 0.4)  # If blocked, ensure accuracy is below threshold
            
            return {
                "accuracy": accuracy,
                "grounded": grounded,
                "message": "Response is grounded in evidence" if grounded else "Response is NOT grounded in evidence",
                "details": {
                    "response": response,
                    "evidence_length": len(evidence_text),
                    "accuracy_score": accuracy,
                    "raw_response": response_content,
                },
            }
            
        except Exception as e:
            self.logger.error(
                "nemo_check_facts_error",
                error=str(e),
                response=response[:100],
            )
            # On error, default to low accuracy (fail-safe)
            return {
                "accuracy": 0.0,
                "grounded": False,
                "message": f"Error checking facts: {str(e)}",
                "details": {"error": str(e)},
            }
    
    async def check_hallucination(
        self,
        response: str,
        user_input: str,
        num_samples: int = 2,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Check if bot response contains hallucinations.
        
        This method uses NeMo's self_check_hallucination approach based on
        SelfCheckGPT methodology to detect when the LLM generates false claims.
        
        The check works by:
        1. Generating alternative responses for the same user query
        2. Using LLM to check if original response agrees with alternatives
        3. Inconsistent responses across samples indicate hallucination
        
        Args:
            response: Bot response to validate
            user_input: Original user query (used to generate alternatives)
            num_samples: Number of alternative responses to generate (default: 2)
            context: Optional additional context variables
            
        Returns:
            Dictionary with:
                - consistency: float - Score between 0.0 (hallucinated) and 1.0 (consistent)
                - is_hallucination: bool - Whether hallucination was detected
                - message: str - Human-readable result message
                - details: dict - Additional details
                
        Example:
            result = await provider.check_hallucination(
                response="Einstein invented the telephone.",
                user_input="What did Einstein invent?"
            )
            if result["is_hallucination"]:
                print("Hallucination detected!")
            else:
                print("Response appears consistent")
        
        Reference:
            https://docs.nvidia.com/nemo/guardrails/latest/user-guides/guardrails-library.html#hallucination-detection
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Step 1: Generate alternative responses for the same query
            alternatives = []
            for _ in range(num_samples):
                alt_result = await self.rails.generate_async(prompt=user_input)
                if isinstance(alt_result, dict):
                    alt_text = alt_result.get("content", "")
                else:
                    alt_text = str(alt_result)
                alternatives.append(alt_text)
            
            if not alternatives:
                # No alternatives generated, cannot check
                return {
                    "consistency": 1.0,
                    "is_hallucination": False,
                    "message": "Could not generate alternatives for comparison",
                    "details": {"alternatives_generated": 0},
                }
            
            # Step 2: Combine alternatives as context (paragraph)
            paragraph = "\n\n".join(alternatives)
            
            # Step 3: Use the hallucination check prompt (NeMo format)
            hallucination_prompt = f'''You are given a task to identify if the hypothesis is in agreement with the context below.
You will only use the contents of the context and not rely on external knowledge.
Answer with yes/no. "context": {paragraph} "hypothesis": {response} "agreement":'''
            
            # Call LLM to check consistency
            result = await self.rails.generate_async(prompt=hallucination_prompt)
            
            # Parse the response
            if isinstance(result, dict):
                answer = result.get("content", "").lower().strip()
            else:
                answer = str(result).lower().strip()
            
            # Extract yes/no from the response
            # "yes" means agreement (not hallucinated), "no" means disagreement (hallucinated)
            agrees = answer.startswith("yes") or "yes" in answer[:10]
            disagrees = answer.startswith("no") or "no" in answer[:10]
            
            if agrees and not disagrees:
                consistency = 1.0
                is_hallucination = False
            elif disagrees:
                consistency = 0.0
                is_hallucination = True
            else:
                # Ambiguous, fail-safe to hallucination
                consistency = 0.0
                is_hallucination = True
            
            return {
                "consistency": consistency,
                "is_hallucination": is_hallucination,
                "message": "Response appears consistent" if not is_hallucination else "Potential hallucination detected",
                "details": {
                    "response": response,
                    "user_input": user_input,
                    "num_samples": num_samples,
                    "alternatives_generated": len(alternatives),
                    "llm_answer": answer,
                },
            }
            
        except Exception as e:
            self.logger.error(
                "nemo_check_hallucination_error",
                error=str(e),
                response=response[:100],
            )
            # On error, default to hallucination (fail-safe)
            return {
                "consistency": 0.0,
                "is_hallucination": True,
                "message": f"Error checking hallucination: {str(e)}",
                "details": {"error": str(e)},
            }
    
    async def check_topic(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> Dict[str, Any]:
        """
        Check if user message is on-topic using NeMo's dialog rails.
        
        This method uses NeMo's generate() with dialog rails to detect off-topic
        messages. The dialog rails are defined in Colang files and include:
        - User intent patterns for allowed/blocked topics
        - Flows that block certain topics (e.g., stock_tips, medical_advice)
        
        We rely entirely on NeMo's Colang flows and LLM-based intent recognition
        to determine if a topic should be blocked. No additional keyword matching.
        
        Args:
            text: User message to check for topic compliance
            context: Optional context with conversation history
            
        Returns:
            Dictionary with:
                - on_topic: True if message is on-topic or allowed
                - response: NeMo's response (refusal if blocked)
                - details: Additional execution details including colang_history
        
        Example:
            >>> result = await provider.check_topic("What's my account balance?")
            >>> # result = {"on_topic": True, "response": "Your account balance is...", ...}
            
            >>> result = await provider.check_topic("Give me a stock tip")
            >>> # result = {"on_topic": False, "response": "I can't provide stock tips...", ...}
        """
        if not self._initialized or not self._rails:
            raise ProviderError(
                "NeMo provider not initialized. Call initialize() first.",
                provider_name=self.name,
            )
        
        try:
            # Build messages for NeMo
            messages = [{"role": "user", "content": text}]
            
            # Add conversation history from context if available
            if context and context.metadata:
                history = context.metadata.get("conversation_history", [])
                if history:
                    # Prepend history to messages
                    messages = history + messages
            
            # Generate response using NeMo rails (includes dialog rails)
            response = await self._rails.generate_async(messages=messages)
            
            # Get explanation for debugging
            info = self._rails.explain()
            colang_history = getattr(info, 'colang_history', '')
            
            # Extract response content
            response_content = response.get("content", "") if isinstance(response, dict) else str(response)
            
            # Trust NeMo's refusal detection
            # If NeMo triggers a refusal flow, it will use the standard refusal response
            # is_blocked = self._is_refusal_response(response_content)
            is_blocked = "BLOCKED" in response_content
            
            return {
                "on_topic": not is_blocked,
                "response": response_content,
                "details": {
                    "colang_history": colang_history[:500] if colang_history else "",
                    "raw_response": response,
                },
            }
            
        except Exception as e:
            self.logger.error(
                "nemo_check_topic_error",
                error=str(e),
                text=text[:100],
            )
            # On error, default to on-topic (fail-open for topic check)
            return {
                "on_topic": True,
                "response": "",
                "details": {"error": str(e)},
            }
    
    async def check_domain(
        self,
        text: str,
        domain: str = "Wealth Management",
        prompt_template: Optional[str] = None,
        context: Optional[GuardrailContext] = None,
    ) -> Dict[str, Any]:
        """
        Check if user message is allowed in a specific domain using prompt-based classification.
        
        This method uses a custom prompt template to classify user queries as ALLOWED or
        BLOCKED based on domain-specific rules. The LLM response determines if the
        query should be allowed or blocked.
        
        Args:
            text: User message to check for domain compliance
            domain: Domain name for context (e.g., "Wealth Management", "Finance")
            prompt_template: Custom prompt template with {{ user_input }} placeholder
            context: Optional context with conversation history
            
        Returns:
            Dictionary with:
                - allowed: True if message is allowed, False if blocked
                - response: LLM response (includes "BLOCKED: reason" if blocked)
                - details: Additional execution details
        
        Example:
            >>> result = await provider.check_domain(
            ...     "What is asset allocation?",
            ...     domain="Wealth Management"
            ... )
            >>> # result = {"allowed": True, "response": "Asset allocation is...", ...}
            
            >>> result = await provider.check_domain(
            ...     "Should I buy Tesla stock?",
            ...     domain="Wealth Management"
            ... )
            >>> # result = {"allowed": False, "response": "BLOCKED: Cannot provide...", ...}
        """
        if not self._initialized or not self._rails:
            raise ProviderError(
                "NeMo provider not initialized. Call initialize() first.",
                provider_name=self.name,
            )
        
        try:
            # Use default prompt template if not provided
            if prompt_template is None:
                prompt_template = """You are a {domain} Assistant. Analyze the user's message and determine if it should be ALLOWED or BLOCKED.

ALLOW if the query is about:
- General domain education (concepts, strategies, planning)
- Explaining terms and processes
- Providing step-by-step instructions for account-related tasks
- General market trends and economic discussions
- Casual conversation

BLOCK if the query asks for:
- Specific investment recommendations or buy/sell advice on particular assets
- Guaranteed returns or performance predictions
- Anything outside the domain (recipes, sports, etc.)
- Illegal, harmful, or unethical activities
- Personal opinions on the user's specific decisions

Response format:
- If BLOCKED: Start with "BLOCKED: [reason]"
- If ALLOWED: Provide helpful, educational guidance

User message: "{{{{ user_input }}}}"
""".format(domain=domain)
            
            # Render the prompt with user input
            rendered_prompt = prompt_template.replace("{{ user_input }}", text)
            
            # Build messages for NeMo
            messages = [{"role": "user", "content": rendered_prompt}]
            
            # Add conversation history from context if available
            if context and context.metadata:
                history = context.metadata.get("conversation_history", [])
                if history:
                    # Prepend history to messages
                    messages = history + messages
            
            # Generate response using NeMo
            response = await self._rails.generate_async(messages=messages)
            
            # Extract response content
            response_content = response.get("content", "") if isinstance(response, dict) else str(response)
            
            # Check if response indicates blocked
            is_blocked = response_content.strip().startswith("BLOCKED:")
            
            return {
                "allowed": not is_blocked,
                "response": response_content,
                "details": {
                    "domain": domain,
                    "raw_response": response,
                },
            }
            
        except Exception as e:
            self.logger.error(
                "nemo_check_domain_error",
                error=str(e),
                text=text[:100],
                domain=domain,
            )
            # On error, default to allowed (fail-open for domain check)
            return {
                "allowed": True,
                "response": "",
                "details": {"error": str(e)},
            }
    
    async def check_wealth_management_domain(
        self,
        text: str,
        context: Optional[GuardrailContext] = None,
    ) -> Dict[str, Any]:
        """
        Check if user message is allowed in wealth management domain using prompt-based classification.
        
        This method uses the wealth_management_domain_check prompt with task_manager to classify
        user queries as ALLOWED or BLOCKED based on wealth management domain rules.
        
        Args:
            text: User message to check for domain compliance
            context: Optional context with conversation history
            
        Returns:
            Dictionary with:
                - allowed: True if message is allowed, False if blocked
                - response: LLM response (includes "BLOCKED: reason" if blocked)
                - details: Additional execution details
        
        Example:
            >>> result = await provider.check_wealth_management_domain(
            ...     "What is asset allocation?"
            ... )
            >>> # result = {"allowed": True, "response": "Asset allocation is...", ...}
            
            >>> result = await provider.check_wealth_management_domain(
            ...     "Should I buy Tesla stock?"
            ... )
            >>> # result = {"allowed": False, "response": "BLOCKED: Cannot provide...", ...}
        """
        if not self._initialized or not self._rails:
            raise ProviderError(
                "NeMo provider not initialized. Call initialize() first.",
                provider_name=self.name,
            )
        
        try:
            # Get task manager
            task_manager = self._rails.runtime.llm_task_manager
            
            # Render the prompt using task_manager (just like in the notebook)
            rendered_prompt = task_manager.render_task_prompt(
                task="wealth_management_domain_check", 
                context={"user_input": text}
            )
            
            # Build messages for NeMo
            messages = [{"role": "user", "content": rendered_prompt}]
            
            # Add conversation history from context if available
            if context and context.metadata:
                history = context.metadata.get("conversation_history", [])
                if history:
                    # Prepend history to messages
                    messages = history + messages
            
            # Generate response using NeMo
            response = await self._rails.generate_async(messages=messages)
            
            # Extract response content
            response_content = response.get("content", "") if isinstance(response, dict) else str(response)
            
            # Check if response indicates blocked and check if contains is_refusal_response
            is_blocked = response_content.strip().startswith("BLOCKED:") or self._is_refusal_response(response_content)

            
            return {
                "allowed": not is_blocked,
                "response": response_content,
                "details": {
                    "domain": "Wealth Management",
                    "raw_response": response,
                    "rendered_prompt_preview": rendered_prompt[:200],
                },
            }
            
        except Exception as e:
            self.logger.error(
                "nemo_check_wealth_management_domain_error",
                error=str(e),
                text=text[:100],
            )
            # On error, default to allowed (fail-open for domain check)
            return {
                "allowed": True,
                "response": "",
                "details": {"error": str(e)},
            }
    
    def get_guardrail(
        self,
        guardrail_type: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseGuardrail:
        """
        Get a guardrail implementation for the given type.
        
        Args:
            guardrail_type: Type of guardrail (e.g., "self_check_input")
            config: Optional configuration for the guardrail
            
        Returns:
            Configured guardrail instance
            
        Raises:
            ProviderError: If guardrail type is not supported
        """
        if guardrail_type not in self._guardrails:
            raise ProviderError(
                f"Guardrail type '{guardrail_type}' not supported by NeMo provider. "
                f"Supported types: {list(self._guardrails.keys())}",
                provider_name=self.name,
                guardrail_type=guardrail_type,
            )
        
        guardrail_class = self._guardrails[guardrail_type]
        
        # Merge config with provider reference
        guardrail_config = config.copy() if config else {}
        guardrail_config["provider"] = self
        
        return guardrail_class(guardrail_config)
    
    async def cleanup(self) -> None:
        """Clean up NeMo resources."""
        self._rails = None
        self._rails_config = None
        self._initialized = False
        
        self.logger.info("nemo_provider_cleanup_complete")


# Factory function for easy provider creation
def create_nemo_provider(
    config_path: Optional[str | Path] = None,
    agent_id: Optional[str] = None,
    output_path: Optional[str | Path] = None,
    prompt_style: str = "simple",
) -> NeMoProvider:
    """
    Create and return a NeMo provider instance.
    
    This is a convenience function for creating NeMoProvider instances.
    
    Args:
        config_path: Path to configs directory (default: ./configs)
        agent_id: Optional agent ID for agent-specific config
        output_path: Directory for generated NeMo configs
        prompt_style: Prompt style ("simple" or "complex")
        
    Returns:
        Configured NeMoProvider instance (not yet initialized)
        
    Example:
        provider = create_nemo_provider(config_path="./configs")
        await provider.initialize()
        
        result = await provider.check_input("Hello!")
    """
    return NeMoProvider(
        config_path=config_path,
        agent_id=agent_id,
        output_path=output_path,
        prompt_style=prompt_style,
    )
