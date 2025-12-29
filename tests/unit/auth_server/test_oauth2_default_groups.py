import pytest
from unittest.mock import Mock

from fastapi import Request

import auth_server.routes.oauth2_routes as oauth2_routes_module


@pytest.mark.unit
class TestOAuth2DefaultGroups:
    @pytest.mark.asyncio
    async def test_oauth2_callback_applies_default_groups_when_missing(self, monkeypatch):
        captured_payloads: list[dict] = []

        def fake_dumps(payload: dict) -> str:
            captured_payloads.append(payload)
            return "signed-session"

        monkeypatch.setattr(
            oauth2_routes_module,
            "OAUTH2_CONFIG",
            {
                "providers": {
                    "google": {
                        "enabled": True,
                        "client_id": "cid",
                        "client_secret": "secret",
                        "auth_url": "https://accounts.google.com/o/oauth2/auth",
                        "token_url": "https://oauth2.googleapis.com/token",
                        "user_info_url": "https://www.googleapis.com/oauth2/v2/userinfo",
                        "scopes": ["openid", "email", "profile"],
                        "response_type": "code",
                        "grant_type": "authorization_code",
                        "username_claim": "email",
                        "groups_claim": None,
                        "email_claim": "email",
                        "name_claim": "name",
                        "default_groups": ["registry-users-lob1"],
                    }
                },
                "session": {"max_age_seconds": 28800, "secure": False, "httponly": True, "samesite": "lax"},
                "registry": {"success_redirect": "/"},
            },
        )

        monkeypatch.setattr(
            oauth2_routes_module.signer,
            "loads",
            lambda *_args, **_kwargs: {"state": "expected", "provider": "google", "redirect_uri": "/"},
        )
        monkeypatch.setattr(oauth2_routes_module.signer, "dumps", fake_dumps)

        async def fake_exchange_code_for_token(*_args, **_kwargs) -> dict:
            return {"access_token": "token"}

        async def fake_get_user_info(*_args, **_kwargs) -> dict:
            return {"email": "user@example.com", "name": "User Name", "id": "12345"}

        monkeypatch.setattr(oauth2_routes_module, "exchange_code_for_token", fake_exchange_code_for_token)
        monkeypatch.setattr(oauth2_routes_module, "get_user_info", fake_get_user_info)

        request = Mock(spec=Request)
        request.headers = {"host": "localhost:8888"}
        request.url = Mock()
        request.url.scheme = "http"

        response = await oauth2_routes_module.oauth2_callback(
            provider="google",
            request=request,
            code="code",
            state="expected",
            oauth2_temp_session="temp-session",
        )

        assert response.status_code == 302
        assert captured_payloads, "Expected session cookie payload to be signed"
        assert captured_payloads[-1]["groups"] == ["registry-users-lob1"]
