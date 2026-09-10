"""Prompts for the per-user preference SUMMARY layer.

Both prompts encode RECENCY-based conflict resolution: when a newer fact
contradicts something already in the summary, the NEWER fact wins and the older
preference is replaced. Non-conflicting information is always preserved, and no
duplicates are emitted. The output is always a plain-text preference summary
(no JSON, no markdown headers) so it can be embedded and stored directly.
"""

# Used on every turn: fold the turn's newly extracted facts INTO the existing
# summary. `old_summary` may be empty (first turn for a user).
MERGE_PROMPT = """You maintain a concise, durable PREFERENCE SUMMARY for a single user.

You are given the CURRENT SUMMARY and a list of NEW FACTS that were just learned
(the new facts are more recent than anything in the current summary).

Produce an UPDATED summary by folding the new facts into the current one:
- KEEP every non-conflicting statement from the current summary.
- ADD the new facts.
- When a new fact CONFLICTS with an existing statement (same preference/dimension,
  different value), the NEW fact WINS — replace the old value, do not keep both.
- Never list the same preference twice; de-duplicate restatements.
- Be concise and factual. Prefer short declarative sentences or bullet points.
- Do NOT invent anything that is not in the current summary or the new facts.
- Do NOT include point-in-time financial values (amounts, balances, %, NAV, prices).

Return ONLY the updated summary text — no preamble, no JSON, no markdown headers."""

# Used by rebuild_summary: regenerate the whole summary from ALL of a user's
# facts, sorted oldest → newest. Later facts override earlier conflicting ones.
REBUILD_PROMPT = """You write a concise, durable PREFERENCE SUMMARY for a single user.

You are given ALL known FACTS about the user, ordered OLDEST → NEWEST.

Write a single summary that captures the user's durable preferences and traits:
- Where two facts CONFLICT (same preference/dimension, different value), the LATER
  (more recent) fact OVERRIDES the earlier one — keep only the latest value.
- KEEP all non-conflicting information.
- Never repeat the same preference; de-duplicate restatements.
- Be concise and factual. Prefer short declarative sentences or bullet points.
- Do NOT invent anything not present in the facts.
- Do NOT include point-in-time financial values (amounts, balances, %, NAV, prices).

Return ONLY the summary text — no preamble, no JSON, no markdown headers."""
