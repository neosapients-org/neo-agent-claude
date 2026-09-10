"""Default prompts for fact extraction and dedup resolution.

These are generic defaults suitable for any agentic application.
Applications should provide domain-specific prompts via config for
better extraction quality.
"""

DEFAULT_EXTRACTION_PROMPT = '''You are an expert AI assistant extracting structured memories from conversations.

Extract STRUCTURED KEY-VALUE MEMORIES from the conversation below.
Each memory must have a category, entity name, key, value, and salience score.

## Categories
{categories_description}

## Visibility Pool
Each memory gets a "pool" that controls access:
- "private" (DEFAULT): Scoped to the specific user.
- "team": Visible to the team/department.
- "org": Visible to the entire organization.

Rule: When in doubt, use "private". Team/org memories are RARE.

## RULES
- ALWAYS include the entity name (person, client, user) when identifiable.
- DO NOT extract greetings, filler, transient data.
- Keys: lowercase_snake_case, descriptive, consistent.
- Salience: 0.0 (trivial) to 1.0 (critical).

## OUTPUT FORMAT
Return JSON: {{"memories": [
  {{
    "category": "persona",
    "pool": "private",
    "investor_name": "Full Name or null",
    "key": "descriptive_key",
    "value": "concise value",
    "detail": "optional context",
    "behavioral_note": "optional for episodic",
    "trigger": "optional for procedural",
    "salience": 0.75,
    "reasoning": "brief justification"
  }}
]}}
'''

DEFAULT_DEDUP_PROMPT = '''You manage long-term memory for an AI assistant.
Each memory may be tagged with an entity name. Memories for different entities should NEVER be merged.

Given EXISTING memories and a NEW fact, choose one action:
- REPLACE: new fact updates or contradicts an existing fact FOR THE SAME ENTITY
- MERGE: new fact adds meaningful NEW detail to an existing fact FOR THE SAME ENTITY
- KEEP_EXISTING: new fact is redundant or less specific than existing (PREFER THIS for near-duplicates)
- NONE: facts are about DIFFERENT entities, or are completely unrelated

CRITICAL RULES:
1. NEVER merge memories for different entities
2. Check entity names FIRST before deciding action
3. Episodic memories with different dates → NONE (keep both)
4. If the new fact says the same thing in different words → KEEP_EXISTING
5. When in doubt → KEEP_EXISTING to avoid bloat
6. RECENCY MATTERS: If the new fact provides more current information, prefer REPLACE

Return JSON:
{
  "action": "REPLACE|MERGE|KEEP_EXISTING|NONE",
  "target_memory_id": "id of existing memory acted upon",
  "updated_memory": "the merged or replacement text (for REPLACE/MERGE)",
  "reason": "short explanation"
}
'''

# Category description template for the extraction prompt
DEFAULT_CATEGORIES_DESCRIPTION = '''### persona — WHO the entity is
Identity, background, emotional drivers, relationships, behavioral patterns.
Keys: risk_behavior, family_context, career_background, life_stage, etc.

### preference — HOW they want to be served
Communication style, format preferences, exclusions.
Keys: communication_channel, report_format, etc.

### episodic — Significant EVENTS, DECISIONS & EMOTIONAL MOMENTS
What happened, why it matters, decision reasoning.
Keys: descriptive event keys.

### procedural — SYSTEM-LEARNED BEHAVIORS (reminders, recurring tasks)
Things the system should remember to do.
Keys: descriptive action keys.'''
