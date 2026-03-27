"""Unit tests for Registry Card fields added to SkillCard and SkillInfo."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import HttpUrl

from registry.schemas.registry_card import LifecycleStatus
from registry.schemas.skill_models import (
    SkillCard,
    SkillInfo,
    SkillRegistrationRequest,
    SkillTier1_Metadata,
)


@pytest.mark.unit
class TestSkillCardRegistryCardFields:
    """Tests for Registry Card fields in SkillCard model."""

    def test_default_lifecycle_status(self):
        """Test that default lifecycle status is ACTIVE."""
        skill = SkillCard(
            path="/skills/test-skill",
            name="test-skill",
            description="Test skill",
            skill_md_url=HttpUrl("https://example.com/SKILL.md"),
        )
        assert skill.status == LifecycleStatus.ACTIVE

    def test_custom_lifecycle_status(self):
        """Test setting custom lifecycle status."""
        skill = SkillCard(
            path="/skills/test-skill",
            name="test-skill",
            description="Test skill",
            skill_md_url=HttpUrl("https://example.com/SKILL.md"),
            status=LifecycleStatus.DEPRECATED,
        )
        assert skill.status == LifecycleStatus.DEPRECATED

    def test_all_lifecycle_statuses(self):
        """Test all lifecycle status values."""
        statuses = [
            LifecycleStatus.ACTIVE,
            LifecycleStatus.DEPRECATED,
            LifecycleStatus.DRAFT,
            LifecycleStatus.BETA,
        ]

        for status in statuses:
            skill = SkillCard(
                path="/skills/test-skill",
                name="test-skill",
                description="Test skill",
                skill_md_url=HttpUrl("https://example.com/SKILL.md"),
                status=status,
            )
            assert skill.status == status

    def test_source_timestamps_default_none(self):
        """Test that source timestamps default to None."""
        skill = SkillCard(
            path="/skills/test-skill",
            name="test-skill",
            description="Test skill",
            skill_md_url=HttpUrl("https://example.com/SKILL.md"),
        )

        assert skill.source_created_at is None
        assert skill.source_updated_at is None

    def test_source_timestamps_with_values(self):
        """Test setting source timestamps."""
        created = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        updated = datetime(2024, 1, 15, 0, 0, 0, tzinfo=UTC)

        skill = SkillCard(
            path="/skills/test-skill",
            name="test-skill",
            description="Test skill",
            skill_md_url=HttpUrl("https://example.com/SKILL.md"),
            source_created_at=created,
            source_updated_at=updated,
        )

        assert skill.source_created_at == created
        assert skill.source_updated_at == updated

    def test_external_tags_default_empty(self):
        """Test that external_tags defaults to empty list."""
        skill = SkillCard(
            path="/skills/test-skill",
            name="test-skill",
            description="Test skill",
            skill_md_url=HttpUrl("https://example.com/SKILL.md"),
        )

        assert skill.external_tags == []

    def test_external_tags_with_values(self):
        """Test setting external tags."""
        skill = SkillCard(
            path="/skills/test-skill",
            name="test-skill",
            description="Test skill",
            skill_md_url=HttpUrl("https://example.com/SKILL.md"),
            external_tags=["federated", "external", "verified"],
        )

        assert skill.external_tags == ["federated", "external", "verified"]
        assert len(skill.external_tags) == 3

    def test_all_registry_card_fields_together(self):
        """Test setting all registry card fields together."""
        created = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        updated = datetime(2024, 1, 15, 0, 0, 0, tzinfo=UTC)

        skill = SkillCard(
            path="/skills/test-skill",
            name="test-skill",
            description="Test skill",
            skill_md_url=HttpUrl("https://example.com/SKILL.md"),
            status=LifecycleStatus.BETA,
            source_created_at=created,
            source_updated_at=updated,
            external_tags=["tag1", "tag2"],
        )

        assert skill.status == LifecycleStatus.BETA
        assert skill.source_created_at == created
        assert skill.source_updated_at == updated
        assert skill.external_tags == ["tag1", "tag2"]

    def test_json_serialization_with_registry_card_fields(self):
        """Test JSON serialization of registry card fields."""
        created = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        updated = datetime(2024, 1, 15, 0, 0, 0, tzinfo=UTC)

        skill = SkillCard(
            path="/skills/test-skill",
            name="test-skill",
            description="Test skill",
            skill_md_url=HttpUrl("https://example.com/SKILL.md"),
            status=LifecycleStatus.DEPRECATED,
            source_created_at=created,
            source_updated_at=updated,
            external_tags=["federated"],
        )

        json_data = skill.model_dump(mode="json")

        assert json_data["status"] == "deprecated"
        assert "source_created_at" in json_data
        assert "source_updated_at" in json_data
        assert json_data["external_tags"] == ["federated"]

        # Round-trip
        restored = SkillCard(**json_data)
        assert restored.status == LifecycleStatus.DEPRECATED
        assert restored.external_tags == ["federated"]

    def test_backwards_compatibility_without_new_fields(self):
        """Test that old data without new fields loads successfully."""
        old_data = {
            "path": "/skills/old-skill",
            "name": "old-skill",
            "description": "Old skill without registry card fields",
            "skill_md_url": "https://example.com/SKILL.md",
            "tags": ["old"],
        }

        # Should load successfully with defaults
        skill = SkillCard(**old_data)

        assert skill.status == LifecycleStatus.ACTIVE
        assert skill.source_created_at is None
        assert skill.source_updated_at is None
        assert skill.external_tags == []


@pytest.mark.unit
class TestSkillInfoRegistryCardFields:
    """Tests for Registry Card fields in SkillInfo model."""

    def test_default_lifecycle_status(self):
        """Test that default lifecycle status is ACTIVE."""
        skill = SkillInfo(
            id=uuid4(),
            path="/skills/test-skill",
            name="test-skill",
            description="Test skill",
            skill_md_url="https://example.com/SKILL.md",
        )
        assert skill.status == LifecycleStatus.ACTIVE

    def test_custom_lifecycle_status(self):
        """Test setting custom lifecycle status."""
        skill = SkillInfo(
            id=uuid4(),
            path="/skills/test-skill",
            name="test-skill",
            description="Test skill",
            skill_md_url="https://example.com/SKILL.md",
            status=LifecycleStatus.DRAFT,
        )
        assert skill.status == LifecycleStatus.DRAFT

    def test_source_timestamps_default_none(self):
        """Test that source timestamps default to None."""
        skill = SkillInfo(
            id=uuid4(),
            path="/skills/test-skill",
            name="test-skill",
            description="Test skill",
            skill_md_url="https://example.com/SKILL.md",
        )

        assert skill.source_created_at is None
        assert skill.source_updated_at is None

    def test_source_timestamps_with_values(self):
        """Test setting source timestamps."""
        created = datetime(2024, 2, 1, 0, 0, 0, tzinfo=UTC)
        updated = datetime(2024, 2, 15, 0, 0, 0, tzinfo=UTC)

        skill = SkillInfo(
            id=uuid4(),
            path="/skills/test-skill",
            name="test-skill",
            description="Test skill",
            skill_md_url="https://example.com/SKILL.md",
            source_created_at=created,
            source_updated_at=updated,
        )

        assert skill.source_created_at == created
        assert skill.source_updated_at == updated

    def test_external_tags_default_empty(self):
        """Test that external_tags defaults to empty list."""
        skill = SkillInfo(
            id=uuid4(),
            path="/skills/test-skill",
            name="test-skill",
            description="Test skill",
            skill_md_url="https://example.com/SKILL.md",
        )

        assert skill.external_tags == []

    def test_external_tags_with_values(self):
        """Test setting external tags."""
        skill = SkillInfo(
            id=uuid4(),
            path="/skills/test-skill",
            name="test-skill",
            description="Test skill",
            skill_md_url="https://example.com/SKILL.md",
            external_tags=["federated", "verified"],
        )

        assert skill.external_tags == ["federated", "verified"]

    def test_all_registry_card_fields_together(self):
        """Test setting all registry card fields together."""
        created = datetime(2024, 2, 1, 0, 0, 0, tzinfo=UTC)
        updated = datetime(2024, 2, 15, 0, 0, 0, tzinfo=UTC)

        skill = SkillInfo(
            id=uuid4(),
            path="/skills/test-skill",
            name="test-skill",
            description="Test skill",
            skill_md_url="https://example.com/SKILL.md",
            status=LifecycleStatus.BETA,
            source_created_at=created,
            source_updated_at=updated,
            external_tags=["tag1", "tag2"],
        )

        assert skill.status == LifecycleStatus.BETA
        assert skill.source_created_at == created
        assert skill.source_updated_at == updated
        assert skill.external_tags == ["tag1", "tag2"]

    def test_backwards_compatibility_without_new_fields(self):
        """Test that old data without new fields loads successfully."""
        old_data = {
            "id": str(uuid4()),
            "path": "/skills/old-skill",
            "name": "old-skill",
            "description": "Old skill without registry card fields",
            "skill_md_url": "https://example.com/SKILL.md",
            "tags": ["old"],
        }

        # Should load successfully with defaults
        skill = SkillInfo(**old_data)

        assert skill.status == LifecycleStatus.ACTIVE
        assert skill.source_created_at is None
        assert skill.source_updated_at is None
        assert skill.external_tags == []


@pytest.mark.unit
class TestSkillRegistrationRequestStatus:
    """Tests for status field in SkillRegistrationRequest."""

    def test_default_status(self):
        """Test that default status is ACTIVE."""
        request = SkillRegistrationRequest(
            name="test-skill",
            description="Test skill",
            skill_md_url=HttpUrl("https://example.com/SKILL.md"),
        )
        assert request.status == LifecycleStatus.ACTIVE

    def test_custom_status(self):
        """Test setting custom status during registration."""
        request = SkillRegistrationRequest(
            name="test-skill",
            description="Test skill",
            skill_md_url=HttpUrl("https://example.com/SKILL.md"),
            status=LifecycleStatus.DRAFT,
        )
        assert request.status == LifecycleStatus.DRAFT

    def test_all_statuses_allowed(self):
        """Test that all lifecycle statuses can be set during registration."""
        statuses = [
            LifecycleStatus.ACTIVE,
            LifecycleStatus.DEPRECATED,
            LifecycleStatus.DRAFT,
            LifecycleStatus.BETA,
        ]

        for status in statuses:
            request = SkillRegistrationRequest(
                name="test-skill",
                description="Test skill",
                skill_md_url=HttpUrl("https://example.com/SKILL.md"),
                status=status,
            )
            assert request.status == status


@pytest.mark.unit
class TestSkillTier1MetadataStatus:
    """Tests for status field in SkillTier1_Metadata."""

    def test_default_status(self):
        """Test that default status is ACTIVE."""
        metadata = SkillTier1_Metadata(
            path="/skills/test-skill",
            name="test-skill",
            description="Test skill",
            skill_md_url="https://example.com/SKILL.md",
        )
        assert metadata.status == LifecycleStatus.ACTIVE

    def test_custom_status(self):
        """Test setting custom status in tier 1 metadata."""
        metadata = SkillTier1_Metadata(
            path="/skills/test-skill",
            name="test-skill",
            description="Test skill",
            skill_md_url="https://example.com/SKILL.md",
            status=LifecycleStatus.BETA,
        )
        assert metadata.status == LifecycleStatus.BETA
