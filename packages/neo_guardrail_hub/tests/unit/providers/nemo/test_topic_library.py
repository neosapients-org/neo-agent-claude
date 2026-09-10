"""
Unit tests for NeMo Topic Library.

Tests the topic library loader and utility functions.
"""

import pytest
from pathlib import Path

from neo_guardrail_hub.providers.nemo import (
    TopicLibrary,
    TopicLibraryEntry,
    load_topic_library,
    get_topic_by_name,
    get_topics_by_category,
    get_sensitive_topics,
    get_harmful_topics,
    get_all_disallowed_topics,
)


class TestTopicLibraryLoader:
    """Tests for loading the topic library."""
    
    def test_load_default_library(self):
        """Test loading the default topic library."""
        library = load_topic_library()
        
        assert isinstance(library, TopicLibrary)
        assert library.version == "1.0.0"
        assert len(library.topics) > 0
        
    def test_library_has_expected_categories(self):
        """Test that the library contains expected categories."""
        library = load_topic_library()
        
        categories = set(t.category for t in library.topics)
        
        assert "sensitive" in categories
        assert "harmful" in categories
        assert "business" in categories
        
    def test_library_has_expected_topics(self):
        """Test that the library contains key topics."""
        library = load_topic_library()
        
        topic_names = [t.name for t in library.topics]
        
        assert "politics" in topic_names
        assert "medical_advice" in topic_names
        assert "violence" in topic_names
        assert "customer_support" in topic_names


class TestGetTopicByName:
    """Tests for get_topic_by_name function."""
    
    def test_get_existing_topic(self):
        """Test getting a topic that exists."""
        topic = get_topic_by_name("politics")
        
        assert topic is not None
        assert topic.name == "politics"
        assert topic.category == "sensitive"
        assert len(topic.user_examples) > 0
        
    def test_get_nonexistent_topic(self):
        """Test getting a topic that doesn't exist."""
        topic = get_topic_by_name("nonexistent_topic_xyz")
        
        assert topic is None
        
    def test_case_insensitive_lookup(self):
        """Test that topic lookup is case insensitive."""
        topic = get_topic_by_name("POLITICS")
        
        assert topic is not None
        assert topic.name == "politics"


class TestGetTopicsByCategory:
    """Tests for get_topics_by_category function."""
    
    def test_get_sensitive_category(self):
        """Test getting topics in the sensitive category."""
        topics = get_topics_by_category("sensitive")
        
        assert len(topics) > 0
        assert all(t.category == "sensitive" for t in topics)
        
    def test_get_harmful_category(self):
        """Test getting topics in the harmful category."""
        topics = get_topics_by_category("harmful")
        
        assert len(topics) > 0
        assert all(t.category == "harmful" for t in topics)
        
    def test_get_business_category(self):
        """Test getting topics in the business category."""
        topics = get_topics_by_category("business")
        
        assert len(topics) > 0
        assert all(t.category == "business" for t in topics)
        
    def test_get_nonexistent_category(self):
        """Test getting topics from a nonexistent category."""
        topics = get_topics_by_category("nonexistent_category")
        
        assert topics == []


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_get_sensitive_topics(self):
        """Test getting sensitive topic names."""
        topics = get_sensitive_topics()
        
        assert isinstance(topics, list)
        assert len(topics) > 0
        assert "politics" in topics
        assert "medical_advice" in topics
        
    def test_get_harmful_topics(self):
        """Test getting harmful topic names."""
        topics = get_harmful_topics()
        
        assert isinstance(topics, list)
        assert len(topics) > 0
        assert "violence" in topics
        assert "illegal_activities" in topics
        
    def test_get_all_disallowed_topics(self):
        """Test getting all disallowed topic names."""
        topics = get_all_disallowed_topics()
        
        sensitive = get_sensitive_topics()
        harmful = get_harmful_topics()
        
        # Should contain both sensitive and harmful topics
        for t in sensitive:
            assert t in topics
        for t in harmful:
            assert t in topics
            
        # Total should be the sum
        assert len(topics) == len(sensitive) + len(harmful)


class TestTopicLibraryEntry:
    """Tests for TopicLibraryEntry model."""
    
    def test_entry_structure(self):
        """Test that topic entries have the expected structure."""
        topic = get_topic_by_name("politics")
        
        assert hasattr(topic, "name")
        assert hasattr(topic, "category")
        assert hasattr(topic, "description")
        assert hasattr(topic, "user_examples")
        assert hasattr(topic, "bot_responses")
        assert hasattr(topic, "keywords")
        
    def test_entry_has_examples(self):
        """Test that topic entries have user examples."""
        topic = get_topic_by_name("politics")
        
        assert len(topic.user_examples) > 0
        assert all(isinstance(ex, str) for ex in topic.user_examples)
        
    def test_entry_has_keywords(self):
        """Test that topic entries have keywords."""
        topic = get_topic_by_name("politics")
        
        assert len(topic.keywords) > 0
        assert all(isinstance(kw, str) for kw in topic.keywords)


class TestTopicLibraryMethods:
    """Tests for TopicLibrary model methods."""
    
    def test_get_topic_method(self):
        """Test TopicLibrary.get_topic method."""
        library = load_topic_library()
        
        topic = library.get_topic("politics")
        
        assert topic is not None
        assert topic.name == "politics"
        
    def test_get_topics_by_category_method(self):
        """Test TopicLibrary.get_topics_by_category method."""
        library = load_topic_library()
        
        topics = library.get_topics_by_category("sensitive")
        
        assert len(topics) > 0
        assert all(t.category == "sensitive" for t in topics)
