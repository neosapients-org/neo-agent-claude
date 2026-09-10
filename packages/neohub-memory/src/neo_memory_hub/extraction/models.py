"""Data models for fact extraction results."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExtractedFact:
    """A single fact extracted from a conversation exchange."""

    category: str  # persona, preference, episodic, procedural
    content: str  # The fact value text
    investor_name: Optional[str] = None  # Entity name (client, user, etc.)
    profile_key: str = ""  # KV key (e.g., "risk_behavior")
    detail: str = ""  # Optional context for persona/preference
    behavioral_note: str = ""  # Psychological insight for episodic
    trigger: str = ""  # When/condition for procedural
    salience: float = 0.5  # 0.0-1.0, importance score
    salience_reasoning: str = ""  # Why this salience score
    pool: Optional[str] = None  # private | team | org (visibility)


@dataclass
class ExtractionResult:
    """Result of a fact extraction call."""

    facts: List[ExtractedFact] = field(default_factory=list)
    investor_names_found: set = field(default_factory=set)
    raw_llm_response: str = ""
    error: Optional[str] = None
