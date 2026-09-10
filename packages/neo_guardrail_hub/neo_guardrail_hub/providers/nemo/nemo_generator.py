"""
NeMo Configuration Generator for Neo Guardrail Hub.

This module provides the high-level interface for generating NeMo Guardrails
configuration files from Neo Guardrail Hub YAML configuration.

The generator reads from:
- configs/default.yaml (if no agent specified)
- configs/agents/{agent_id}.yaml or configs/agents/{agent_id}/guardrails.yaml

And generates:
- {output_path}/config.yml - NeMo main configuration
- {output_path}/rails.co - Colang flow definitions
- {output_path}/prompts.yml - LLM prompts for self-check tasks
"""

from pathlib import Path
from typing import Any, Literal, Optional

from .config_mapper import ConfigToNeMoMapper
from .generators import ColangGenerator, ConfigGenerator, PromptsGenerator
from .generators.models import NeMoGuardrailsConfig


# Type for prompt style
PromptStyle = Literal["simple", "complex"]


class NeMoConfigGenerator:
    """
    High-level generator for NeMo Guardrails configuration.
    
    This class orchestrates the generation of all NeMo configuration files
    from Neo Guardrail Hub's YAML configuration using dedicated generators:
    - ConfigGenerator for config.yml
    - PromptsGenerator for prompts.yml
    - ColangGenerator for rails.co
    
    Example:
        >>> generator = NeMoConfigGenerator(config_path="./configs")
        >>> 
        >>> # Generate from default.yaml, save to ./nemo_config/
        >>> generator.generate()
        >>> 
        >>> # Generate for specific agent, save to custom path
        >>> generator.generate(
        ...     agent_id="wealth_advisor",
        ...     output_path="./agents/wealth_advisor/nemo"
        ... )
        >>>
        >>> # Generate with complex prompts for more thorough checking
        >>> generator.generate(prompt_style="complex")
    """
    
    # Default output directory - created in current working directory
    DEFAULT_OUTPUT_DIR = "neo_configs"
    
    def __init__(
        self, 
        config_path: str | Path = "./configs",
        templates_path: Optional[Path] = None,
        presets_path: Optional[Path] = None,
        output_base_path: Optional[str | Path] = None,
    ):
        """
        Initialize the generator.
        
        Args:
            config_path: Path to the configuration directory containing
                         default.yaml and optionally agents/ subdirectory
            templates_path: Optional path to templates directory
            presets_path: Optional path to presets directory
            output_base_path: Base path for output files. If None, uses
                             current working directory
        """
        self.config_path = Path(config_path)
        self.mapper = ConfigToNeMoMapper(config_path)
        
        # Set output base path (where neo_configs will be created)
        self.output_base_path = Path(output_base_path) if output_base_path else Path.cwd()
        
        # Set default paths relative to package
        package_dir = Path(__file__).parent
        self.templates_path = templates_path or (package_dir / "templates")
        self.presets_path = presets_path or (package_dir / "presets")
        
    def generate(
        self,
        agent_id: Optional[str] = None,
        output_path: Optional[str | Path] = None,
        config_override: Optional[dict[str, Any]] = None,
        prompt_style: PromptStyle = "simple",
    ) -> Path:
        """
        Generate NeMo configuration files from YAML config.
        
        Generated files are saved to a persistent location:
        - Default: {cwd}/neo_configs/
        - With agent: {cwd}/neo_configs/{agent_id}/
        - Custom: Specified output_path
        
        Args:
            agent_id: Optional agent ID for agent-specific configuration.
                     If None, uses default.yaml
            output_path: Directory to save generated files.
                        If None, uses path from config or defaults to
                        {output_base_path}/{DEFAULT_OUTPUT_DIR}/
            config_override: Optional runtime configuration overrides
            prompt_style: Prompt style ("simple" or "complex")
            
        Returns:
            Path to the output directory containing generated files
            (e.g., /path/to/project/neo_configs/)
        """
        # Determine output path
        if output_path is None:
            # Check if output_path is specified in YAML config
            config_output_path = self.mapper.get_output_path(
                agent_id=agent_id,
                config_override=config_override,
            )
            if config_output_path:
                output_path = self.output_base_path / config_output_path
            elif agent_id:
                output_path = self.output_base_path / self.DEFAULT_OUTPUT_DIR / agent_id
            else:
                output_path = self.output_base_path / self.DEFAULT_OUTPUT_DIR
        else:
            output_path = Path(output_path)
            # If relative path, make it relative to output_base_path
            if not output_path.is_absolute():
                output_path = self.output_base_path / output_path
            
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Map YAML config to NeMo config
        nemo_config = self.mapper.map_to_nemo_config(
            agent_id=agent_id,
            config_override=config_override,
        )
        
        # Generate config.yml WITH prompts included (NeMo works best this way)
        # Only self_check_input, self_check_output, self_check_facts, self_check_hallucination
        # are supported - these are built into NeMo
        self._generate_config_yml_with_prompts(nemo_config, output_path, prompt_style)
        
        # Generate rails/*.co files for Colang flows
        self._generate_rails_files(nemo_config, output_path)
        
        return output_path
    
    def generate_colang_only(
        self,
        agent_id: Optional[str] = None,
        output_path: Optional[str | Path] = None,
        config_override: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Generate only the Colang files in rails/ directory and return combined content.
        
        Note: This generates individual .co files in rails/ subdirectory, not a single file.
        Returns the combined content as a string for display/inspection purposes.
        
        If output_path is not specified, saves to {output_base_path}/neo_configs/rails/
        
        Args:
            agent_id: Optional agent ID
            output_path: Optional base path. Rails files will be created in
                        {output_path}/rails/ directory. If None, uses neo_configs/
            config_override: Optional configuration overrides
            
        Returns:
            Combined Colang content as string (for display/inspection)
        """
        # Map config
        colang_config = self.mapper.map_to_colang(
            agent_id=agent_id,
            config_override=config_override,
        )
        
        # Generate Colang
        generator = ColangGenerator(colang_config)
        content = generator.generate()  # Combined content for return value
        
        # Determine base output path
        if output_path is None:
            # Default to neo_configs/
            if agent_id:
                base_path = self.output_base_path / self.DEFAULT_OUTPUT_DIR / agent_id
            else:
                base_path = self.output_base_path / self.DEFAULT_OUTPUT_DIR
        else:
            base_path = Path(output_path)
            if not base_path.is_absolute():
                base_path = self.output_base_path / base_path
        
        # Generate individual rail files in rails/ subdirectory
        rails_dir = base_path / "rails"
        generator.generate_rails_to_directory(rails_dir)
            
        return content
    
    def generate_config_only(
        self,
        agent_id: Optional[str] = None,
        output_path: Optional[str | Path] = None,
        config_override: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Generate only the config.yml and return its content as dictionary.
        
        If output_path is not specified, saves to {output_base_path}/neo_configs/config.yml
        
        Args:
            agent_id: Optional agent ID
            output_path: Optional path to save the file. If None, saves to
                        neo_configs/config.yml in current directory
            config_override: Optional configuration overrides
            
        Returns:
            Generated config as dictionary
        """
        # Map YAML config to NeMo config
        nemo_config = self.mapper.map_to_nemo_config(
            agent_id=agent_id,
            config_override=config_override,
        )
        
        # Generate config
        generator = ConfigGenerator(
            nemo_config=nemo_config,
            presets_path=self.presets_path,
        )
        config_dict = generator.generate()
        
        # Determine output path
        if output_path is None:
            # Default to neo_configs/config.yml
            if agent_id:
                output_path = self.output_base_path / self.DEFAULT_OUTPUT_DIR / agent_id / "config.yml"
            else:
                output_path = self.output_base_path / self.DEFAULT_OUTPUT_DIR / "config.yml"
        else:
            output_path = Path(output_path)
            if not output_path.is_absolute():
                output_path = self.output_base_path / output_path
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generator.write(output_path)
            
        return config_dict
    
    def generate_config_yaml(
        self,
        agent_id: Optional[str] = None,
        config_override: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Generate config.yml content as YAML string.
        
        Args:
            agent_id: Optional agent ID
            config_override: Optional configuration overrides
            
        Returns:
            Generated config as YAML string
        """
        nemo_config = self.mapper.map_to_nemo_config(
            agent_id=agent_id,
            config_override=config_override,
        )
        generator = ConfigGenerator(
            nemo_config=nemo_config,
            presets_path=self.presets_path,
        )
        return generator.generate_yaml()
    
    def generate_prompts_only(
        self,
        agent_id: Optional[str] = None,
        output_path: Optional[str | Path] = None,
        config_override: Optional[dict[str, Any]] = None,
        prompt_style: PromptStyle = "simple",
    ) -> dict[str, Any]:
        """
        Generate only the prompts.yml and return its content as dictionary.
        
        If output_path is not specified, saves to {output_base_path}/neo_configs/prompts.yml
        
        Args:
            agent_id: Optional agent ID
            output_path: Optional path to save the file. If None, saves to
                        neo_configs/prompts.yml in current directory
            config_override: Optional configuration overrides
            prompt_style: Prompt style ("simple" or "complex")
            
        Returns:
            Generated prompts as dictionary
        """
        # Map YAML config to NeMo config
        nemo_config = self.mapper.map_to_nemo_config(
            agent_id=agent_id,
            config_override=config_override,
        )
        
        if not nemo_config.colang:
            return {"prompts": []}
        
        # Generate prompts
        generator = PromptsGenerator(
            colang_config=nemo_config.colang,
            style=prompt_style,
            templates_path=self.templates_path,
        )
        prompts_dict = generator.generate()
        
        # Only save if there are prompts to write
        if prompts_dict.get("prompts"):
            # Determine output path
            if output_path is None:
                # Default to neo_configs/prompts.yml
                if agent_id:
                    output_path = self.output_base_path / self.DEFAULT_OUTPUT_DIR / agent_id / "prompts.yml"
                else:
                    output_path = self.output_base_path / self.DEFAULT_OUTPUT_DIR / "prompts.yml"
            else:
                output_path = Path(output_path)
                if not output_path.is_absolute():
                    output_path = self.output_base_path / output_path
            
            # Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            generator.write(output_path)
            
        return prompts_dict
    
    def generate_prompts_yaml(
        self,
        agent_id: Optional[str] = None,
        config_override: Optional[dict[str, Any]] = None,
        prompt_style: PromptStyle = "simple",
    ) -> str:
        """
        Generate prompts.yml content as YAML string.
        
        Args:
            agent_id: Optional agent ID
            config_override: Optional configuration overrides
            prompt_style: Prompt style ("simple" or "complex")
            
        Returns:
            Generated prompts as YAML string
        """
        nemo_config = self.mapper.map_to_nemo_config(
            agent_id=agent_id,
            config_override=config_override,
        )
        
        if not nemo_config.colang:
            return "prompts: []\n"
        
        generator = PromptsGenerator(
            colang_config=nemo_config.colang,
            style=prompt_style,
            templates_path=self.templates_path,
        )
        return generator.generate_yaml()
    
    def _generate_config_yml(
        self,
        nemo_config: NeMoGuardrailsConfig,
        output_path: Path,
    ) -> None:
        """Generate NeMo config.yml file using ConfigGenerator."""
        generator = ConfigGenerator(
            nemo_config=nemo_config,
            presets_path=self.presets_path,
        )
        generator.write(output_path / "config.yml")
    
    def _generate_config_yml_with_prompts(
        self,
        nemo_config: NeMoGuardrailsConfig,
        output_path: Path,
        prompt_style: PromptStyle = "simple",
    ) -> None:
        """Generate NeMo config.yml with prompts included inline.
        
        NeMo works more reliably when prompts are embedded in config.yml
        rather than being in a separate prompts.yml file. This method
        generates a single config.yml with both rails and prompts.
        
        Args:
            nemo_config: NeMo configuration
            output_path: Directory to save config.yml
            prompt_style: Prompt style ("simple" or "complex")
        """
        import yaml
        
        # Generate base config
        generator = ConfigGenerator(
            nemo_config=nemo_config,
            presets_path=self.presets_path,
        )
        config_dict = generator.generate()
        
        # Generate prompts and add them to config
        if nemo_config.colang:
            prompts_generator = PromptsGenerator(
                colang_config=nemo_config.colang,
                style=prompt_style,
                templates_path=self.templates_path,
            )
            prompts_dict = prompts_generator.generate()
            
            if prompts_dict.get("prompts"):
                config_dict["prompts"] = prompts_dict["prompts"]
        
        # Write combined config
        config_file = output_path / "config.yml"
        
        lines = [
            "# Auto-generated by Neo Guardrail Hub",
            "# NeMo Guardrails configuration with prompts",
            "",
        ]
        
        yaml_content = yaml.dump(
            config_dict, 
            default_flow_style=False, 
            sort_keys=False,
            allow_unicode=True,
        )
        
        config_file.write_text("\n".join(lines) + yaml_content)
            
    def _generate_colang(
        self,
        nemo_config: NeMoGuardrailsConfig,
        output_path: Path,
    ) -> None:
        """Generate Colang rails.co file using ColangGenerator."""
        generator = ColangGenerator(nemo_config.colang)
        content = generator.generate()
        
        colang_file = output_path / "rails.co"
        colang_file.write_text(content)
    
    def _generate_rails_files(
        self,
        nemo_config: NeMoGuardrailsConfig,
        output_path: Path,
    ) -> None:
        """Generate individual Colang files in rails/ subdirectory.
        
        Uses ColangGenerator to create separate .co files for each enabled rail type:
        - rails/input.co - self_check_input flow
        - rails/output.co - self_check_output flow  
        - rails/fact_check.co - self_check_facts flow
        - rails/hallucination.co - self_check_hallucination flow
        - rails/dialog.co - topical/dialog rails
        
        NeMo's RailsConfig.from_path() will automatically load all .co files
        from the directory and subdirectories.
        
        Args:
            nemo_config: NeMo configuration with enabled rails
            output_path: Base output directory (e.g., neo_configs/)
        """
        if not nemo_config.colang:
            return
        
        # Use ColangGenerator to generate individual rail files
        generator = ColangGenerator(nemo_config.colang)
        rails_dir = output_path / "rails"
        
        # Generate all enabled rails to the directory
        generated_files = generator.generate_rails_to_directory(rails_dir)
        
        # Log what was generated (optional, for debugging)
        if generated_files:
            pass  # Could add logging here if needed
        
    def _generate_prompts_yml(
        self,
        nemo_config: NeMoGuardrailsConfig,
        output_path: Path,
        prompt_style: PromptStyle = "simple",
    ) -> None:
        """Generate prompts.yml file using PromptsGenerator."""
        if not nemo_config.colang:
            return
            
        generator = PromptsGenerator(
            colang_config=nemo_config.colang,
            style=prompt_style,
            templates_path=self.templates_path,
        )
        
        # Only write if there are prompts to generate
        prompts = generator.generate()
        if prompts.get("prompts"):
            generator.write(output_path / "prompts.yml")


def generate_nemo_from_yaml(
    config_path: str | Path = "./configs",
    agent_id: Optional[str] = None,
    output_path: Optional[str | Path] = None,
) -> Path:
    """
    Convenience function to generate NeMo config from YAML.
    
    Args:
        config_path: Path to configuration directory
        agent_id: Optional agent ID
        output_path: Optional output directory
        
    Returns:
        Path to output directory
    """
    generator = NeMoConfigGenerator(config_path)
    return generator.generate(agent_id=agent_id, output_path=output_path)
