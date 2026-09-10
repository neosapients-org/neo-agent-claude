"""Tests for ProfileAssembler and ContextFormatter."""
import pytest

from neo_memory_hub.retrieval.formatter import ContextFormatter, ProfileAssembler


def _make_memory(content, category, metadata=None, memory_id="m1"):
    """Helper to create test memory dicts."""
    return {
        "id": memory_id,
        "memory": content,
        "metadata": metadata or {},
        "_retrieval_reason": category,
    }


class TestProfileAssembler:
    """Test ProfileAssembler."""

    @pytest.fixture
    def assembler(self):
        return ProfileAssembler()

    def test_empty_memories(self, assembler):
        profile = assembler.assemble([], entity_name="Test")
        assert profile["entity_name"] == "Test"
        assert "persona" not in profile
        assert "episodic" not in profile

    def test_persona_single_key(self, assembler):
        memories = [
            _make_memory(
                "Conservative investor", "persona",
                {"profile_key": "risk_behavior", "salience_score": 0.9},
                "m1",
            ),
        ]
        profile = assembler.assemble(memories, entity_name="Senthil")
        assert "persona" in profile
        assert profile["persona"]["risk_behavior"] == "Conservative investor"

    def test_persona_with_detail(self, assembler):
        memories = [
            _make_memory(
                "Conservative", "persona",
                {"profile_key": "risk", "salience_score": 0.9, "detail": "Prefers low risk"},
                "m1",
            ),
        ]
        profile = assembler.assemble(memories)
        assert isinstance(profile["persona"]["risk"], dict)
        assert profile["persona"]["risk"]["value"] == "Conservative"
        assert profile["persona"]["risk"]["detail"] == "Prefers low risk"

    def test_persona_multiple_keys(self, assembler):
        memories = [
            _make_memory(
                "Conservative", "persona",
                {"profile_key": "risk", "salience_score": 0.9},
                "m1",
            ),
            _make_memory(
                "Tech background", "persona",
                {"profile_key": "career", "salience_score": 0.7},
                "m2",
            ),
        ]
        profile = assembler.assemble(memories)
        assert profile["persona"]["risk"] == "Conservative"
        assert profile["persona"]["career"] == "Tech background"

    def test_persona_multiple_values_same_key(self, assembler):
        memories = [
            _make_memory(
                "Value 1", "persona",
                {"profile_key": "trait", "salience_score": 0.5},
                "m1",
            ),
            _make_memory(
                "Value 2", "persona",
                {"profile_key": "trait", "salience_score": 0.9},
                "m2",
            ),
        ]
        profile = assembler.assemble(memories)
        # Should be a list since multiple entries for same key
        assert isinstance(profile["persona"]["trait"], list)
        assert len(profile["persona"]["trait"]) == 2
        # Sorted by salience descending
        assert profile["persona"]["trait"][0] == "Value 2"

    def test_preference_assembly(self, assembler):
        memories = [
            _make_memory(
                "Email only", "preference",
                {"profile_key": "comm_channel", "salience_score": 0.8},
                "m1",
            ),
        ]
        profile = assembler.assemble(memories)
        assert profile["preference"]["comm_channel"] == "Email only"

    def test_episodic_assembly(self, assembler):
        memories = [
            _make_memory(
                "Sold house in 2024", "episodic",
                {
                    "profile_key": "housing_decision",
                    "behavioral_note": "Shows caution",
                    "stored_at": "2024-06-15 10:00 UTC",
                    "salience_score": 0.8,
                },
                "m1",
            ),
        ]
        profile = assembler.assemble(memories)
        assert "episodic" in profile
        events = profile["episodic"]
        assert len(events) == 1
        assert events[0]["event"] == "Sold house in 2024"
        assert events[0]["behavioral_note"] == "Shows caution"
        assert events[0]["date"] == "2024-06-15"
        assert events[0]["key"] == "housing_decision"

    def test_procedural_assembly(self, assembler):
        memories = [
            _make_memory(
                "Send quarterly report", "procedural",
                {
                    "trigger": "Every quarter end",
                    "stored_at": "2024-01-15 10:00 UTC",
                    "salience_score": 0.9,
                },
                "m1",
            ),
        ]
        profile = assembler.assemble(memories)
        assert "procedural" in profile
        actions = profile["procedural"]
        assert len(actions) == 1
        assert actions[0]["action"] == "Send quarterly report"
        assert actions[0]["trigger"] == "Every quarter end"

    def test_team_pool_assembly(self, assembler):
        memories = [
            _make_memory(
                "All RMs must follow KYC", "team",
                {"profile_key": "kyc", "stored_at": "2024-03-01", "salience_score": 0.95},
                "m1",
            ),
        ]
        profile = assembler.assemble(memories)
        assert "team" in profile
        assert profile["team"][0]["directive"] == "All RMs must follow KYC"
        assert profile["team"][0]["pool"] == "team"

    def test_org_pool_assembly(self, assembler):
        memories = [
            _make_memory(
                "Firm-wide risk limit", "org",
                {"salience_score": 0.99},
                "m1",
            ),
        ]
        profile = assembler.assemble(memories)
        assert "org" in profile
        assert profile["org"][0]["directive"] == "Firm-wide risk limit"

    def test_empty_content_skipped(self, assembler):
        memories = [
            _make_memory("", "episodic", {}, "m1"),
            _make_memory("Valid event", "episodic", {"salience_score": 0.5}, "m2"),
        ]
        profile = assembler.assemble(memories)
        assert len(profile.get("episodic", [])) == 1

    def test_sorted_by_salience(self, assembler):
        memories = [
            _make_memory(
                "Low importance", "episodic",
                {"salience_score": 0.2}, "m1",
            ),
            _make_memory(
                "High importance", "episodic",
                {"salience_score": 0.9}, "m2",
            ),
        ]
        profile = assembler.assemble(memories)
        assert profile["episodic"][0]["event"] == "High importance"

    def test_mixed_categories(self, assembler):
        memories = [
            _make_memory("Conservative", "persona", {"profile_key": "risk", "salience_score": 0.9}, "m1"),
            _make_memory("Email me", "preference", {"profile_key": "comm", "salience_score": 0.8}, "m2"),
            _make_memory("Sold house", "episodic", {"salience_score": 0.7}, "m3"),
            _make_memory("Send report", "procedural", {"salience_score": 0.6}, "m4"),
        ]
        profile = assembler.assemble(memories, entity_name="John")
        assert "persona" in profile
        assert "preference" in profile
        assert "episodic" in profile
        assert "procedural" in profile
        assert profile["entity_name"] == "John"


class TestContextFormatter:
    """Test ContextFormatter."""

    @pytest.fixture
    def formatter(self):
        return ContextFormatter(header="---MEMORY CONTEXT---")

    def test_empty_profile(self, formatter):
        profile = {"entity_name": "Test"}
        result = formatter.format(profile)
        assert result == ""

    def test_persona_section(self, formatter):
        profile = {
            "entity_name": "Senthil",
            "persona": {"risk_behavior": "Conservative"},
        }
        result = formatter.format(profile, entity_name="Senthil")
        assert "---MEMORY CONTEXT---" in result
        assert "CLIENT PROFILE (Senthil)" in result
        assert "**Risk Behavior:**" in result
        assert "Conservative" in result

    def test_persona_with_detail_dict(self, formatter):
        profile = {
            "persona": {
                "risk": {"value": "Conservative", "detail": "Low risk only", "salience": 0.9},
            },
        }
        result = formatter.format(profile, entity_name="Test")
        assert "Conservative" in result
        assert "Low risk only" in result

    def test_preference_section(self, formatter):
        profile = {
            "preference": {"communication": "Email only"},
        }
        result = formatter.format(profile)
        assert "SERVICE PREFERENCES" in result
        assert "Email only" in result

    def test_episodic_section(self, formatter):
        profile = {
            "episodic": [
                {"event": "Sold house", "behavioral_note": "Risk averse", "date": "2024-06-15"},
            ],
        }
        result = formatter.format(profile)
        assert "CLIENT EVENTS" in result
        assert "Sold house" in result
        assert "Risk averse" in result
        assert "2024-06-15" in result

    def test_procedural_section(self, formatter):
        profile = {
            "procedural": [
                {"action": "Send report", "trigger": "quarterly", "date": "2024-01-15"},
            ],
        }
        result = formatter.format(profile)
        assert "REMINDERS & PROCEDURES" in result
        assert "Send report" in result
        assert "[when: quarterly]" in result

    def test_procedural_without_trigger(self, formatter):
        profile = {
            "procedural": [
                {"action": "Remember preference", "date": "2024-01-01"},
            ],
        }
        result = formatter.format(profile)
        assert "Remember preference" in result
        assert "[when:" not in result

    def test_team_section(self, formatter):
        profile = {
            "team": [
                {"directive": "Follow KYC rules", "pool": "team", "date": "2024-03-01"},
            ],
        }
        result = formatter.format(profile)
        assert "DEPARTMENT DIRECTIVES" in result
        assert "Follow KYC rules" in result

    def test_org_section(self, formatter):
        profile = {
            "org": [
                {"directive": "Risk limit 10%", "pool": "org"},
            ],
        }
        result = formatter.format(profile)
        assert "FIRM-WIDE DIRECTIVES" in result
        assert "Risk limit 10%" in result

    def test_truncation(self):
        formatter = ContextFormatter(max_chars=100)
        profile = {
            "persona": {f"key_{i}": f"{'x' * 50}" for i in range(10)},
        }
        result = formatter.format(profile, entity_name="Test")
        assert len(result) <= 100
        assert "truncated" in result

    def test_custom_header(self):
        formatter = ContextFormatter(header="---CUSTOM HEADER---")
        profile = {"persona": {"key": "value"}}
        result = formatter.format(profile)
        assert "---CUSTOM HEADER---" in result

    def test_entity_name_from_profile(self, formatter):
        profile = {
            "entity_name": "FromProfile",
            "persona": {"key": "value"},
        }
        result = formatter.format(profile)
        assert "FromProfile" in result

    def test_persona_list_values(self, formatter):
        profile = {
            "persona": {
                "traits": ["Cautious", "Analytical"],
            },
        }
        result = formatter.format(profile, entity_name="Test")
        assert "Cautious" in result
        assert "Analytical" in result
