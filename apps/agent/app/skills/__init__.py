"""Skills framework (M3) — reusable domain playbooks that shape how the agent reasons.

A skill is an instruction file (metadata + Markdown) the agent selects per question and weaves
into its reasoning: either compiled into a decomposition plan (Option B, for step-by-step
methodologies) or injected into the answer prompt (Option A). For M3 skills are bundled as a
static seed; the backend-backed registry + sidebar UI come in M5/M6.
"""

from .models import Skill
from .registry import registry

__all__ = ["Skill", "registry"]
