"""Skill data model (M3)."""

from dataclasses import dataclass, field


@dataclass
class Skill:
    """One reusable domain playbook. Mirrors the platform's skill registry fields."""
    skill_id: str
    name: str
    domain: str                       # e.g. wealth_management
    skill_type: str                   # reasoning_methodology | output_format | tool_playbook | domain_knowledge
    sensitivity: str = "internal"     # internal | public
    description: str = ""
    tags: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)          # phrases that activate it
    sample_questions: list[str] = field(default_factory=list)
    content: str = ""                 # the Markdown methodology the agent follows (Option A)
    # Option B compile target: ordered sub-question templates ({client} is filled at runtime).
    # When present on a reasoning_methodology skill, the skill is compiled into a decomposition
    # plan the Execution Engine runs, instead of relying on the planner LLM.
    plan_steps: list[dict] = field(default_factory=list)
    synthesis: str = ""               # how Generate should assemble the final answer
    requires_client: bool = False     # skill needs a specific client to run (rule-based gate)
    # Dual-scope: the methodology handles BOTH a specific named client AND the whole book
    # ("my clients"). When True, compile_skill_plan fills {client} with the resolved name for a
    # single-client query, or with "each client" for a book-wide query — one skill, both scopes.
    book_capable: bool = False
    # Option B rule-based activation: regex patterns that VETO this skill even when a trigger
    # matched (e.g. logistics/interrogative/past-tense uses of a trigger word). Evaluated
    # case-insensitively against the user's query. Empty = no exclusions.
    exclusions: list[str] = field(default_factory=list)
    version: str = "v1"
    status: str = "active"            # active | draft
    created_by: str = "seed"

    def brief(self) -> dict:
        """Lightweight dict carried in agent state for the Generate node (no heavy fields)."""
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "skill_type": self.skill_type,
            "content": self.content,
            "synthesis": self.synthesis,
        }
