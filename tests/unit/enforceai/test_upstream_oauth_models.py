"""
Unit tests for EnforceAI upstream OAuth models (Phase 1).
"""

from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)

import pytest

from auth_server.enforceai.models.upstream_oauth import (
    UpstreamOAuthServerStartRequest,
    UpstreamOAuthStateRecord,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


@pytest.mark.unit
class TestUpstreamOAuthModels:
    def test_ui_return_url_allows_relative_paths(
        self,
    ) -> None:
        parsed = UpstreamOAuthServerStartRequest(
            credential_type="oauth2",
            credential_binding="user",
            provider="github",
            ui_return_url="/credentials/upstream/oauth/callback?server=fininfo#done",
        )
        assert parsed.ui_return_url.startswith("/")

    def test_ui_return_url_rejects_absolute_urls(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="relative path"):
            UpstreamOAuthServerStartRequest(
                credential_type="oauth2",
                credential_binding="user",
                provider="github",
                ui_return_url="https://evil.example/steal",
            )

    def test_ui_return_url_rejects_protocol_relative_urls(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="relative path"):
            UpstreamOAuthServerStartRequest(
                credential_type="oauth2",
                credential_binding="user",
                provider="github",
                ui_return_url="//evil.example/steal",
            )

    def test_state_record_validates_ui_return_url(
        self,
    ) -> None:
        now = _utc_now()
        record = UpstreamOAuthStateRecord(
            state_id="state-1",
            server_path="/fininfo",
            credential_type="oauth2",
            credential_binding="user",
            user_id="user-1",
            agent_id=None,
            provider="github",
            redirect_uri="http://localhost/enforceai/upstream/oauth/callback",
            ui_return_url="/credentials/upstream/oauth/callback",
            created_at=now,
            expires_at=now + timedelta(seconds=60),
        )
        assert record.ui_return_url == "/credentials/upstream/oauth/callback"
