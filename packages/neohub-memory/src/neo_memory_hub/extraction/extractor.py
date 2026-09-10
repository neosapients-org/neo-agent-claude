"""LLM-based fact extraction pipeline.

Extracts structured key-value facts from conversation exchanges using a single
LLM call. Each fact includes category, entity name, profile key, value, salience
score, and visibility pool.

The extraction prompt is customizable:
1. Pass a custom prompt via config (recommended for domain-specific extraction)
2. Fall back to DEFAULT_EXTRACTION_PROMPT if no custom prompt provided
"""
from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

from neo_memory_hub.extraction.models import ExtractedFact, ExtractionResult
from neo_memory_hub.extraction.prompts import (
    DEFAULT_CATEGORIES_DESCRIPTION,
    DEFAULT_EXTRACTION_PROMPT,
)

logger = logging.getLogger(__name__)

# Names that the LLM sometimes returns as "investor_name" but should be treated as null
_NULL_NAME_STRINGS = frozenset(
    {
        "null",
        "none",
        "unknown",
        "n/a",
        "",
        "unknown client",
        "client",
        "the client",
        "user",
        "the user",
    }
)


class FactExtractor:
    """Extracts structured facts from conversation exchanges via LLM.

    Usage:
        extractor = FactExtractor(
            llm_model="gpt-4o-mini",
            extraction_prompt=my_custom_prompt,  # optional
            valid_categories=["persona", "preference", "episodic", "procedural"],
        )
        result = await extractor.extract(
            query="What about Senthil's portfolio?",
            response="Based on his conservative profile...",
        )
        for fact in result.facts:
            print(f"{fact.category}: {fact.content} (salience={fact.salience})")
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        extraction_prompt: Optional[str] = None,
        valid_categories: Optional[List[str]] = None,
        default_salience: float = 0.5,
        categories_description: Optional[str] = None,
    ) -> None:
        self._llm_model = llm_model
        self._temperature = temperature
        self._extraction_prompt = extraction_prompt
        self._valid_categories = set(
            valid_categories or ["persona", "preference", "episodic", "procedural"]
        )
        self._default_salience = default_salience
        self._categories_description = (
            categories_description or DEFAULT_CATEGORIES_DESCRIPTION
        )

    def _build_system_prompt(self) -> str:
        """Build the system prompt for extraction."""
        if self._extraction_prompt:
            return self._extraction_prompt
        return DEFAULT_EXTRACTION_PROMPT.format(
            categories_description=self._categories_description,
        )

    def _validate_category(self, raw: Optional[str]) -> str:
        """Validate category string. Returns validated category or 'persona' default."""
        if not raw or not isinstance(raw, str):
            return "persona"
        cat = raw.strip().lower()
        return cat if cat in self._valid_categories else "persona"

    @staticmethod
    def _clean_entity_name(raw: Any) -> Optional[str]:
        """Clean entity/investor name from LLM output.

        Filters out null-like strings the LLM sometimes returns.
        Returns None if name is empty or a null sentinel.
        """
        if not raw or not isinstance(raw, str):
            return None
        cleaned = raw.strip()
        if cleaned.lower() in _NULL_NAME_STRINGS:
            return None
        return cleaned

    async def extract(
        self,
        query: str,
        response: str,
        max_input_chars: int = 12000,
        prompt: str | None = None,
    ) -> ExtractionResult:
        """Extract structured facts from a conversation exchange.

        Args:
            query: User's query/message.
            response: Assistant's response.
            max_input_chars: Max combined chars for query+response. Truncates if exceeded.
            prompt: Per-call extraction prompt override. When provided, replaces
                the instance-level extraction prompt for this call only.

        Returns:
            ExtractionResult with list of ExtractedFact objects.
        """
        from openai import AsyncOpenAI

        # Truncate if needed
        if max_input_chars and (len(query) + len(response)) > max_input_chars:
            total_len = max(len(query) + len(response), 1)
            query_budget = max(200, int(max_input_chars * (len(query) / total_len)))
            response_budget = max(200, max_input_chars - query_budget)
            query = query[:query_budget]
            response = response[:response_budget]

        system_prompt = prompt if prompt is not None else self._build_system_prompt()
        categories_str = ", ".join(sorted(self._valid_categories))

        user_content = (
            f"Conversation to analyze:\n"
            f"User: {query}\n"
            f"Assistant: {response}\n\n"
            f"Extract structured key-value memories. Valid categories: {categories_str}\n"
            f'Return JSON with "memories" array.'
        )

        try:
            client = AsyncOpenAI()
            llm_response = await client.chat.completions.create(
                model=self._llm_model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
            )

            raw_text = llm_response.choices[0].message.content or "{}"
            raw_data = json.loads(raw_text)
            raw_facts = raw_data.get("memories", [])

            facts: List[ExtractedFact] = []
            names_found: set = set()

            for item in raw_facts:
                if not isinstance(item, dict):
                    continue

                # Extract value (KV format)
                fact_text = str(item.get("value") or "").strip()
                if not fact_text:
                    continue

                # Entity name
                entity_name = self._clean_entity_name(item.get("investor_name"))
                if entity_name:
                    names_found.add(entity_name)

                # Category validation
                category = self._validate_category(item.get("category"))

                # Salience
                salience = item.get("salience")
                if salience is not None:
                    try:
                        salience = max(0.0, min(1.0, float(salience)))
                    except (ValueError, TypeError):
                        salience = None
                if salience is None:
                    salience = self._default_salience

                # Pool
                pool_raw = (item.get("pool") or "").strip().lower()
                pool = (
                    pool_raw
                    if pool_raw in ("private", "team", "org", "shared")
                    else None
                )

                facts.append(
                    ExtractedFact(
                        category=category,
                        content=fact_text,
                        investor_name=entity_name,
                        profile_key=(item.get("key") or "").strip(),
                        detail=(item.get("detail") or "").strip(),
                        behavioral_note=(item.get("behavioral_note") or "").strip(),
                        trigger=(item.get("trigger") or "").strip(),
                        salience=salience,
                        salience_reasoning=item.get("reasoning", ""),
                        pool=pool,
                    )
                )

            logger.info(
                f"[FactExtractor] Extracted {len(facts)} facts, "
                f"entities: {names_found or 'none'}"
            )
            return ExtractionResult(
                facts=facts,
                investor_names_found=names_found,
                raw_llm_response=raw_text,
            )

        except Exception as e:
            logger.error(f"[FactExtractor] LLM extraction failed: {e}")
            return ExtractionResult(error=str(e))
