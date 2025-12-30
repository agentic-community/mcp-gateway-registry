"""
Unit tests for upstream OAuth routes helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from auth_server.enforceai.api.upstream_oauth_routes import (
    _build_secret_payload,
    _revoke_matching_credentials,
)
from auth_server.enforceai.upstream.oauth_client import (
    OAuthTokenSet,
)


@dataclass(frozen=True)
class _CredentialRecord:
    credential_id: str
    credential_type: str
    credential_binding: str
    provider: str


class _FakeUpstreamCredentialStore:
    def __init__(
        self,
        records: list[_CredentialRecord],
    ) -> None:
        self._records = records
        self.revoked_ids: list[str] = []

    def list_credentials(
        self,
        *,
        server_path: str,
        user_id: str,
        agent_id: Optional[str],
        include_revoked: bool,
    ) -> list[_CredentialRecord]:
        return list(self._records)

    def revoke_credential(
        self,
        *,
        credential_id: str,
    ) -> None:
        self.revoked_ids.append(credential_id)


@pytest.mark.unit
class TestUpstreamOauthRoutesHelpers:
    def test_build_secret_payload_includes_refresh_and_id_when_present(
        self,
    ) -> None:
        tokens = OAuthTokenSet(
            access_token="access",
            token_type="Bearer",
            refresh_token="refresh",
            id_token="id",
            expires_at=None,
            scopes=None,
        )

        payload = _build_secret_payload(tokens)

        assert payload == {
            "access_token": "access",
            "refresh_token": "refresh",
            "id_token": "id",
        }

    def test_build_secret_payload_omits_optional_tokens_when_missing(
        self,
    ) -> None:
        tokens = OAuthTokenSet(
            access_token="access",
            token_type="Bearer",
            refresh_token=None,
            id_token=None,
            expires_at=None,
            scopes=None,
        )

        payload = _build_secret_payload(tokens)

        assert payload == {"access_token": "access"}

    def test_revoke_matching_credentials_only_revokes_matching_records(
        self,
    ) -> None:
        store = _FakeUpstreamCredentialStore(
            [
                _CredentialRecord(
                    credential_id="1",
                    credential_type="oauth2",
                    credential_binding="user",
                    provider="provider-a",
                ),
                _CredentialRecord(
                    credential_id="2",
                    credential_type="oauth2",
                    credential_binding="user",
                    provider="provider-b",
                ),
                _CredentialRecord(
                    credential_id="3",
                    credential_type="oidc",
                    credential_binding="user",
                    provider="provider-a",
                ),
            ]
        )

        revoked = _revoke_matching_credentials(
            upstream_store=store,
            server_path="/server",
            user_id="user",
            agent_id=None,
            credential_type="oauth2",
            credential_binding="user",
            provider="provider-a",
        )

        assert revoked == 1
        assert store.revoked_ids == ["1"]

