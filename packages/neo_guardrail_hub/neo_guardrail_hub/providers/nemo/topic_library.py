"""
Topic Library Loader for Neo Guardrail Hub.

This module provides utilities for loading and using the pre-built topic library.
"""

from pathlib import Path
from typing import Optional

import yaml

from .generators.models import TopicLibrary, TopicLibraryEntry


def get_default_topics_path() -> Path:
    """Get the path to the default topics library."""
    return Path(__file__).parent / "templates" / "topics" / "examples.yml"


def load_topic_library(path: Optional[Path] = None) -> TopicLibrary:
    """
    Load the topic library from a YAML file.
    
    Args:
        path: Path to the topics YAML file. If None, uses the default library.
        
    Returns:
        TopicLibrary containing all pre-built topics
    """
    if path is None:
        path = get_default_topics_path()
    
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    
    return TopicLibrary.model_validate(data)


def get_topic_by_name(name: str, library: Optional[TopicLibrary] = None) -> Optional[TopicLibraryEntry]:
    """
    Get a topic by name from the library.
    
    Args:
        name: Name of the topic (e.g., "politics", "medical_advice")
        library: TopicLibrary to search. If None, loads the default library.
        
    Returns:
        TopicLibraryEntry if found, None otherwise
    """
    if library is None:
        library = load_topic_library()
    
    return library.get_topic(name)


def get_topics_by_category(
    category: str,
    library: Optional[TopicLibrary] = None
) -> list[TopicLibraryEntry]:
    """
    Get all topics in a category.
    
    Args:
        category: Category name (e.g., "sensitive", "harmful", "business")
        library: TopicLibrary to search. If None, loads the default library.
        
    Returns:
        List of TopicLibraryEntry objects in the category
    """
    if library is None:
        library = load_topic_library()
    
    return library.get_topics_by_category(category)


def get_sensitive_topics(library: Optional[TopicLibrary] = None) -> list[str]:
    """
    Get names of all sensitive topics.
    
    Args:
        library: TopicLibrary to use. If None, loads the default library.
        
    Returns:
        List of topic names in the "sensitive" category
    """
    topics = get_topics_by_category("sensitive", library)
    return [t.name for t in topics]


def get_harmful_topics(library: Optional[TopicLibrary] = None) -> list[str]:
    """
    Get names of all harmful topics.
    
    Args:
        library: TopicLibrary to use. If None, loads the default library.
        
    Returns:
        List of topic names in the "harmful" category
    """
    topics = get_topics_by_category("harmful", library)
    return [t.name for t in topics]


def get_all_disallowed_topics(library: Optional[TopicLibrary] = None) -> list[str]:
    """
    Get names of all topics typically disallowed (sensitive + harmful).
    
    Args:
        library: TopicLibrary to use. If None, loads the default library.
        
    Returns:
        List of topic names that are typically disallowed
    """
    if library is None:
        library = load_topic_library()
    
    sensitive = get_topics_by_category("sensitive", library)
    harmful = get_topics_by_category("harmful", library)
    
    return [t.name for t in sensitive] + [t.name for t in harmful]
