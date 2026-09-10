"""Default LLM functions for the summary layer.

These mirror the OpenAI usage in ``extraction/extractor.py`` (AsyncOpenAI,
key resolved from the environment). They are deliberately plain async callables
so callers can inject their own ``merge_fn`` / ``summarize_fn`` for any provider
without subclassing — see ``SummaryStore``.

Signatures (the injection contract):
    async def merge_fn(old_summary: str, new_facts: list[str]) -> str
    async def summarize_fn(sorted_facts: list[str]) -> str
"""

from __future__ import annotations

import logging
from typing import List

from .prompts import MERGE_PROMPT, REBUILD_PROMPT

logger = logging.getLogger(__name__)


def _format_facts(facts: List[str]) -> str:
    return "\n".join(f"- {f}" for f in facts if f and f.strip())


async def llm_merge(
    old_summary: str,
    new_facts: List[str],
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
) -> str:
    """Fold ``new_facts`` (more recent) into ``old_summary``; newest wins on conflict.

    Returns the updated summary text. On any failure returns ``old_summary``
    unchanged (callers treat the summary layer as best-effort/non-fatal).
    """
    from openai import AsyncOpenAI

    facts_block = _format_facts(new_facts)
    if not facts_block:
        return old_summary or ""

    user_content = (
        f"CURRENT SUMMARY:\n{old_summary or '(none)'}\n\n"
        f"NEW FACTS (most recent):\n{facts_block}"
    )
    try:
        client = AsyncOpenAI()
        resp = await client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": MERGE_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        return (resp.choices[0].message.content or "").strip() or (old_summary or "")
    except Exception as e:  # pragma: no cover - network/credentials
        logger.warning("[summary.llm_merge] failed: %s", e)
        return old_summary or ""


async def llm_summarize(
    sorted_facts: List[str],
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
) -> str:
    """Build a fresh summary from ALL facts (oldest → newest); later facts override.

    Returns the summary text, or "" if there are no facts / on failure.
    """
    from openai import AsyncOpenAI

    facts_block = _format_facts(sorted_facts)
    if not facts_block:
        return ""

    user_content = f"FACTS (oldest → newest):\n{facts_block}"
    try:
        client = AsyncOpenAI()
        resp = await client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": REBUILD_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:  # pragma: no cover - network/credentials
        logger.warning("[summary.llm_summarize] failed: %s", e)
        return ""
