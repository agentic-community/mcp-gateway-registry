"""Metadata limits on the agent and skill request models (issue #1811).

Agents and skills take metadata through Pydantic request bodies, so the limits
live on those models: POST /api/agents/register, PUT /api/agents/{path}, the
register and replace items of POST /api/agents/batch, POST /api/skills and
PUT /api/skills/{path}. The stored models (AgentCard, SkillCard) are left
unchecked on purpose, so records saved before the limits existed still load.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from registry.schemas.agent_models import (
    AgentBatchRequest,
    AgentCard,
    AgentCardPatch,
    AgentRegistrationRequest,
)
from registry.schemas.skill_models import (
    SkillCard,
    SkillRegistrationRequest,
    skill_metadata_storage_shape,
)
from registry.utils.metadata_limits import (
    MAX_METADATA_BYTES,
    metadata_depth,
    metadata_serialized_size,
)


def _metadata_of_size(total_bytes: int) -> dict[str, str]:
    """Return flat metadata whose compact UTF-8 JSON is exactly ``total_bytes``."""
    return {"k": "x" * (total_bytes - 8)}


def _metadata_of_depth(depth: int) -> dict[str, Any]:
    doc: dict[str, Any] = {"leaf": "value"}
    for _ in range(depth - 1):
        doc = {"n": doc}
    return doc


def _agent(**overrides: Any) -> AgentRegistrationRequest:
    fields: dict[str, Any] = {
        "name": "limits-agent",
        "url": "https://agent.example.net",
        "supportedProtocol": "a2a",
    }
    fields.update(overrides)
    return AgentRegistrationRequest(**fields)


def _skill(**overrides: Any) -> SkillRegistrationRequest:
    fields: dict[str, Any] = {
        "name": "limits-skill",
        "description": "Metadata limits test",
        "skill_md_url": "https://github.com/org/skills/SKILL.md",
    }
    fields.update(overrides)
    return SkillRegistrationRequest(**fields)


# The stored skill wrapper for flat metadata {"k": v} is
# {"author":null,"version":null,"extra":{"k":v}}: 46 bytes of envelope.
_SKILL_ENVELOPE_BYTES = metadata_serialized_size(skill_metadata_storage_shape({"k": ""}))


@pytest.mark.unit
class TestAgentRegistrationRequestLimits:
    def test_missing_metadata_is_fine(self):
        assert _agent().metadata == {}

    def test_exactly_the_byte_limit_is_accepted(self):
        metadata = _metadata_of_size(MAX_METADATA_BYTES)

        assert _agent(metadata=metadata).metadata == metadata

    def test_one_byte_over_is_rejected(self):
        with pytest.raises(ValidationError, match="10241 bytes"):
            _agent(metadata=_metadata_of_size(MAX_METADATA_BYTES + 1))

    def test_exactly_eight_levels_is_accepted(self):
        _agent(metadata=_metadata_of_depth(8))

    def test_nine_levels_is_rejected(self):
        with pytest.raises(ValidationError, match="nests 9 levels deep"):
            _agent(metadata=_metadata_of_depth(9))

    def test_batch_register_and_replace_items_inherit_the_limit(self):
        """POST /api/agents/batch validates each item's card at submit time."""
        oversized = {
            "name": "limits-agent",
            "url": "https://agent.example.net",
            "supportedProtocol": "a2a",
            "metadata": _metadata_of_size(MAX_METADATA_BYTES + 1),
        }
        for item in (
            {"op": "register", "card": oversized},
            {"op": "replace", "path": "/limits-agent", "card": oversized},
        ):
            with pytest.raises(ValidationError, match="10241 bytes"):
                AgentBatchRequest(items=[item])

    def test_patch_cannot_carry_metadata(self):
        """PATCH /api/agents/{path} has no metadata field, so it cannot grow metadata."""
        with pytest.raises(ValidationError, match="metadata"):
            AgentCardPatch(metadata={"k": "v"})

    def test_stored_agent_card_is_not_checked(self):
        """An agent stored before the limits existed must still load."""
        card = AgentCard(
            name="legacy",
            description="stored before #1811",
            version="1.0.0",
            url="https://agent.example.net",
            metadata=_metadata_of_size(MAX_METADATA_BYTES * 2),
        )

        assert metadata_serialized_size(card.metadata) == MAX_METADATA_BYTES * 2


@pytest.mark.unit
class TestSkillMetadataStorageShape:
    def test_lifts_author_and_version_and_nests_the_rest_under_extra(self):
        shape = skill_metadata_storage_shape(
            {"author": "a", "version": "1.0.0", "team": "platform"},
        )

        assert shape == {"author": "a", "version": "1.0.0", "extra": {"team": "platform"}}

    def test_explicit_version_wins_over_metadata_version(self):
        shape = skill_metadata_storage_shape({"version": "1.0.0"}, version="2.0.0")

        assert shape["version"] == "2.0.0"

    def test_version_alone_still_stores_a_wrapper(self):
        assert skill_metadata_storage_shape(None, version="1.0.0") == {
            "author": None,
            "version": "1.0.0",
            "extra": {},
        }

    @pytest.mark.parametrize("metadata", [None, {}], ids=["none", "empty"])
    def test_nothing_is_stored_without_metadata_or_version(self, metadata):
        assert skill_metadata_storage_shape(metadata) is None

    def test_matches_what_skill_card_persists(self):
        """The shape the limits check is the shape SkillCard.metadata dumps to."""
        flat = {"author": "a", "version": "1.0.0", "team": "platform", "cfg": {"x": [1]}}
        card = SkillCard(
            name="limits-skill",
            description="Metadata limits test",
            path="/skills/limits-skill",
            skill_md_url="https://github.com/org/skills/SKILL.md",
            metadata=skill_metadata_storage_shape(flat),
        )

        assert card.metadata.model_dump(mode="json") == skill_metadata_storage_shape(flat)


@pytest.mark.unit
class TestSkillRegistrationRequestLimits:
    def test_missing_metadata_is_fine(self):
        assert _skill().metadata is None

    def test_stored_shape_exactly_at_the_byte_limit_is_accepted(self):
        metadata = {"k": "x" * (MAX_METADATA_BYTES - _SKILL_ENVELOPE_BYTES)}
        assert metadata_serialized_size(skill_metadata_storage_shape(metadata)) == (
            MAX_METADATA_BYTES
        )

        _skill(metadata=metadata)

    def test_wrapper_overhead_counts_toward_the_byte_limit(self):
        """Flat metadata under 10,240 bytes is rejected once the stored wrapper pushes it over."""
        metadata = {"k": "x" * (MAX_METADATA_BYTES - _SKILL_ENVELOPE_BYTES + 1)}
        assert metadata_serialized_size(metadata) < MAX_METADATA_BYTES

        with pytest.raises(ValidationError, match="10241 bytes"):
            _skill(metadata=metadata)

    def test_explicit_version_counts_toward_the_byte_limit(self):
        """A version field replaces the null in the wrapper, adding its length in bytes."""
        metadata = {"k": "x" * (MAX_METADATA_BYTES - _SKILL_ENVELOPE_BYTES)}
        _skill(metadata=metadata)

        # "1.0.0" in quotes replaces null: 7 bytes in, 4 out, so 3 bytes over.
        with pytest.raises(ValidationError, match="10243 bytes"):
            _skill(metadata=metadata, version="1.0.0")

    def test_extra_nesting_counts_as_a_level(self):
        """Flat metadata 7 levels deep is 8 once stored under extra; 8 levels becomes 9."""
        seven = _metadata_of_depth(7)
        eight = _metadata_of_depth(8)
        assert metadata_depth(skill_metadata_storage_shape(seven)) == 8

        _skill(metadata=seven)
        with pytest.raises(ValidationError, match="nests 9 levels deep"):
            _skill(metadata=eight)

    def test_error_explains_the_stored_shape(self):
        with pytest.raises(ValidationError, match="as stored: author, version and extra"):
            _skill(metadata=_metadata_of_depth(8))

    def test_ordinary_skill_metadata_is_accepted(self):
        request = _skill(
            metadata={
                "author": "platform-team",
                "version": "1.2.0",
                "owner_team": "platform",
                "links": {"docs": "https://docs.example.com/skill"},
            },
        )

        assert request.metadata["owner_team"] == "platform"
