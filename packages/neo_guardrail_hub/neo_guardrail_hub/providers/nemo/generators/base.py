"""Base generator class for NeMo config generation.

This module provides the abstract base class that all generators
must inherit from.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import structlog

logger = structlog.get_logger(__name__)


class BaseGenerator(ABC):
    """Abstract base class for NeMo config generators.
    
    All generators (Colang, Config, Prompts) inherit from this class
    and implement the generate() method.
    
    Attributes:
        templates_path: Path to the templates directory
        
    Example:
        class MyGenerator(BaseGenerator):
            def generate(self, config: Dict) -> str:
                return "generated content"
    """
    
    def __init__(self, templates_path: Optional[Path] = None) -> None:
        """Initialize the generator.
        
        Args:
            templates_path: Path to templates directory. If None, uses
                           the default templates bundled with the package.
        """
        if templates_path is None:
            # Use default templates from package
            self.templates_path = Path(__file__).parent.parent / "templates"
        else:
            self.templates_path = Path(templates_path)
        
        self.logger = logger.bind(generator=self.__class__.__name__)
    
    @abstractmethod
    def generate(self, config: Dict[str, Any]) -> str:
        """Generate content from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Generated content as string
        """
        pass
    
    def write(self, content: str, output_path: Union[str, Path]) -> Path:
        """Write generated content to a file.
        
        Args:
            content: Content to write
            output_path: Path to output file
            
        Returns:
            Path to the written file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        self.logger.info(
            "generator_wrote_file",
            path=str(output_path),
            size_bytes=len(content),
        )
        
        return output_path
    
    def load_template(self, template_name: str) -> str:
        """Load a template file from the templates directory.
        
        Args:
            template_name: Relative path to template file
            
        Returns:
            Template content as string
            
        Raises:
            FileNotFoundError: If template doesn't exist
        """
        template_path = self.templates_path / template_name
        
        if not template_path.exists():
            raise FileNotFoundError(
                f"Template not found: {template_name} "
                f"(looked in {self.templates_path})"
            )
        
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def template_exists(self, template_name: str) -> bool:
        """Check if a template file exists.
        
        Args:
            template_name: Relative path to template file
            
        Returns:
            True if template exists
        """
        template_path = self.templates_path / template_name
        return template_path.exists()
