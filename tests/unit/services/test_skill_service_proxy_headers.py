"""Unit tests for skill_service upstream custom-header handling.

Covers three proxy-header paths on registry.services.skill_service:

* ``_build_skill_card`` encrypts valid ``custom_headers`` into the persisted
  ``custom_headers_encrypted`` blob and derives ``custom_header_names``.
* An invalid custom header (a gateway-reserved name) raises ``ValueError`` in
  ``_build_skill_card`` and is mapped to ``SkillValidationError`` by
  ``register_skill``.
* An ``update_skill`` that repoints the proxy target clears the stored upstream
  headers (credential-misdirection guard) before persisting.
"""

from unittest.mock import AsyncMock, patch

import pytest

from registry.exceptions import SkillValidationError
from registry.schemas.skill_models import SkillCard, SkillRegistrationRequest
from registry.services.skill_service import SkillService, _build_skill_card

_SKILL_MD_URL = "https://raw.githubusercontent.com/acme/skills/main/SKILL.md"


def _make_request(**overrides) -> SkillRegistrationRequest:
    """Construct a minimal valid registration request."""
    data = {
        "name": "demo-skill",
        "description": "A demo skill for header tests.",
        "skill_md_url": _SKILL_MD_URL,
    }
    data.update(overrides)
    return SkillRegistrationRequest(**data)


class TestBuildSkillCardCustomHeaders:
    """_build_skill_card validates + encrypts upstream custom headers."""

    def test_valid_custom_headers_are_encrypted(self) -> None:
        """A valid header populates the encrypted blob and the name list."""
        request = _make_request(
            custom_headers=[{"name": "X-Api-Key", "value": "s3cret"}],
        )

        card = _build_skill_card(
            request=request,
            path="/skills/demo-skill",
            owner="alice@example.com",
            content_version=None,
            content_updated_at=None,
        )

        assert isinstance(card, SkillCard)
        # Ciphertext is populated and is NOT the plaintext value.
        assert card.custom_headers_encrypted is not None
        assert "s3cret" not in str(card.custom_headers_encrypted)
        # The (non-secret) header name is retained for rendering/UX.
        assert card.custom_header_names == ["X-Api-Key"]
        assert card.custom_headers_updated_at is not None

    def test_overridable_reserved_authorization_records_override_name(self) -> None:
        """A caller-overridable Authorization header is accepted + tracked."""
        request = _make_request(
            custom_headers=[{"name": "Authorization", "overridable": True}],
        )

        card = _build_skill_card(
            request=request,
            path="/skills/demo-skill",
            owner=None,
            content_version=None,
            content_updated_at=None,
        )

        assert "Authorization" in card.custom_header_overridable_names

    def test_reserved_header_name_raises_value_error(self) -> None:
        """A gateway-managed header name is rejected with ValueError."""
        request = _make_request(
            custom_headers=[{"name": "Host", "value": "evil.example.com"}],
        )

        with pytest.raises(ValueError, match="managed by the gateway"):
            _build_skill_card(
                request=request,
                path="/skills/demo-skill",
                owner=None,
                content_version=None,
                content_updated_at=None,
            )


class TestRegisterSkillMapsValidationError:
    """register_skill maps the _build_skill_card ValueError to a 4xx error."""

    @pytest.mark.asyncio
    async def test_invalid_custom_header_becomes_skill_validation_error(self) -> None:
        """A reserved custom header surfaces as SkillValidationError, not 500."""
        service = SkillService()
        # Reject the id-conflict/create path if we ever reach it (we should not).
        service._repo = AsyncMock()
        service._search_repo = AsyncMock()

        request = _make_request(
            custom_headers=[{"name": "Cookie", "value": "sid=abc"}],
        )

        with pytest.raises(SkillValidationError):
            await service.register_skill(request, owner=None, validate_url=False)

        # Failure happens during card build, before any persistence.
        service._repo.create.assert_not_called()


class TestUpdateSkillRepointClearsHeaders:
    """update_skill clears stored upstream headers when the target repoints."""

    @pytest.mark.asyncio
    async def test_target_change_clears_upstream_headers(self) -> None:
        service = SkillService()

        existing = SkillCard(
            path="/skills/demo-skill",
            id="demo-skill",
            name="demo-skill",
            description="A demo skill.",
            skill_md_url=_SKILL_MD_URL,
            is_proxied=True,
            proxy_target_url="https://old.example.com/v1",
            custom_headers_encrypted=[{"name": "X-Api-Key", "value_encrypted": "old-ct"}],
            custom_header_names=["X-Api-Key"],
        )

        repo = AsyncMock()
        repo.get.return_value = existing
        repo.update.return_value = existing
        service._repo = repo
        service._search_repo = AsyncMock()

        pin = {
            "proxy_resolved_ips": ["203.0.113.10"],
            "proxy_target_host": "new.example.com",
        }

        with patch(
            "registry.services.skill_service.validate_and_pin_proxy_target",
            new=AsyncMock(return_value=pin),
        ):
            await service.update_skill(
                "/skills/demo-skill",
                {"proxy_target_url": "https://new.example.com/v1"},
            )

        # The dict actually persisted must have the upstream headers wiped.
        persisted = repo.update.await_args.args[1]
        assert persisted["custom_headers_encrypted"] is None
        assert persisted["custom_header_names"] == []
        assert persisted["custom_header_overridable_names"] == []
        assert persisted["custom_headers_updated_at"] is None
        # Pin bookkeeping refreshed for the new target.
        assert persisted["proxy_target_host"] == "new.example.com"
        assert persisted["proxy_resolved_ips"] == ["203.0.113.10"]

    @pytest.mark.asyncio
    async def test_same_target_keeps_headers(self) -> None:
        """A no-op repoint (identical target) leaves headers untouched."""
        service = SkillService()

        existing = SkillCard(
            path="/skills/demo-skill",
            id="demo-skill",
            name="demo-skill",
            description="A demo skill.",
            skill_md_url=_SKILL_MD_URL,
            is_proxied=True,
            proxy_target_url="https://same.example.com/v1",
        )

        repo = AsyncMock()
        repo.get.return_value = existing
        repo.update.return_value = existing
        service._repo = repo
        service._search_repo = AsyncMock()

        pin = {
            "proxy_resolved_ips": ["203.0.113.20"],
            "proxy_target_host": "same.example.com",
        }

        with patch(
            "registry.services.skill_service.validate_and_pin_proxy_target",
            new=AsyncMock(return_value=pin),
        ):
            await service.update_skill(
                "/skills/demo-skill",
                {"proxy_target_url": "https://same.example.com/v1"},
            )

        persisted = repo.update.await_args.args[1]
        # No header keys were added by the repoint guard.
        assert "custom_headers_encrypted" not in persisted
