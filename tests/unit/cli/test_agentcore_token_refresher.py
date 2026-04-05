"""Unit tests for cli.agentcore.token_refresher.

Tests IdP vendor detection, client secret resolution, OIDC discovery,
token requests, registry updates, and end-to-end refresh_all flow.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from cli.agentcore.token_refresher import (
    _detect_idp_vendor,
    _get_client_secret,
    _get_cognito_client_secret,
    _get_token_endpoint,
    _load_registry_token,
    _read_manifest,
    _request_token,
    _trigger_security_scan,
    _update_registry_credential,
    refresh_all,
)


# ---------------------------------------------------------------------------
# _detect_idp_vendor
# ---------------------------------------------------------------------------


class TestDetectIdpVendor:
    """Tests for IdP vendor detection from discovery URL."""

    def test_cognito(self):
        url = "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_abc/.well-known/openid-configuration"
        assert _detect_idp_vendor(url) == "cognito"

    def test_auth0(self):
        url = "https://myorg.auth0.com/.well-known/openid-configuration"
        assert _detect_idp_vendor(url) == "auth0"

    def test_okta(self):
        url = "https://myorg.okta.com/.well-known/openid-configuration"
        assert _detect_idp_vendor(url) == "okta"

    def test_entra(self):
        url = "https://login.microsoftonline.com/tenant-id/.well-known/openid-configuration"
        assert _detect_idp_vendor(url) == "entra"

    def test_keycloak(self):
        url = "https://keycloak.example.com/realms/myrealm/.well-known/openid-configuration"
        assert _detect_idp_vendor(url) == "keycloak"

    def test_unknown(self):
        url = "https://custom-idp.example.com/.well-known/openid-configuration"
        assert _detect_idp_vendor(url) == "unknown"


# ---------------------------------------------------------------------------
# _read_manifest
# ---------------------------------------------------------------------------


class TestReadManifest:
    """Tests for manifest reading."""

    def test_reads_valid_manifest(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        entries = [{"server_path": "/test", "discovery_url": "https://example.com"}]
        manifest.write_text(json.dumps(entries))

        result = _read_manifest(str(manifest))
        assert len(result) == 1
        assert result[0]["server_path"] == "/test"

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _read_manifest(str(tmp_path / "nonexistent.json"))

    def test_raises_on_invalid_json(self, tmp_path):
        manifest = tmp_path / "bad.json"
        manifest.write_text("not valid json{{{")

        with pytest.raises(ValueError, match="Invalid JSON"):
            _read_manifest(str(manifest))

    def test_raises_on_non_array(self, tmp_path):
        manifest = tmp_path / "obj.json"
        manifest.write_text('{"not": "an array"}')

        with pytest.raises(ValueError, match="JSON array"):
            _read_manifest(str(manifest))


# ---------------------------------------------------------------------------
# _get_cognito_client_secret
# ---------------------------------------------------------------------------


class TestGetCognitoClientSecret:
    """Tests for Cognito client secret auto-retrieval."""

    @patch("cli.agentcore.token_refresher.boto3")
    def test_retrieves_secret(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.describe_user_pool_client.return_value = {
            "UserPoolClient": {"ClientSecret": "super-secret"}
        }

        discovery_url = (
            "https://cognito-idp.us-east-1.amazonaws.com/"
            "us-east-1_pnikLWYzO/.well-known/openid-configuration"
        )
        result = _get_cognito_client_secret(discovery_url, "my-client-id")

        assert result == "super-secret"
        mock_boto3.client.assert_called_once_with("cognito-idp", region_name="us-east-1")
        mock_client.describe_user_pool_client.assert_called_once_with(
            UserPoolId="us-east-1_pnikLWYzO",
            ClientId="my-client-id",
        )

    @patch("cli.agentcore.token_refresher.boto3")
    def test_returns_none_on_error(self, mock_boto3):
        mock_boto3.client.side_effect = Exception("Access denied")

        result = _get_cognito_client_secret(
            "https://cognito-idp.us-east-1.amazonaws.com/pool/.well-known/openid-configuration",
            "client-id",
        )
        assert result is None


# ---------------------------------------------------------------------------
# _get_client_secret
# ---------------------------------------------------------------------------


class TestGetClientSecret:
    """Tests for client secret resolution per IdP vendor."""

    def test_per_client_env_var_takes_priority(self):
        env = {"OAUTH_CLIENT_SECRET_my-client-id": "from-env"}
        with patch.dict(os.environ, env):
            result = _get_client_secret(
                "cognito", "https://cognito-idp.example.com", "my-client-id"
            )
            assert result == "from-env"

    @patch("cli.agentcore.token_refresher._get_cognito_client_secret")
    def test_cognito_delegates_to_auto_retrieval(self, mock_cognito):
        mock_cognito.return_value = "cognito-secret"

        result = _get_client_secret("cognito", "https://cognito-idp.example.com", "client-id")

        assert result == "cognito-secret"
        mock_cognito.assert_called_once()

    def test_auth0_reads_from_env(self):
        with patch.dict(os.environ, {"AUTH0_CLIENT_SECRET": "auth0-secret"}):
            result = _get_client_secret("auth0", "https://myorg.auth0.com", "client-id")
            assert result == "auth0-secret"

    def test_okta_reads_from_env(self):
        with patch.dict(os.environ, {"OKTA_CLIENT_SECRET": "okta-secret"}):
            result = _get_client_secret("okta", "https://myorg.okta.com", "client-id")
            assert result == "okta-secret"

    def test_entra_reads_from_env(self):
        with patch.dict(os.environ, {"ENTRA_CLIENT_SECRET": "entra-secret"}):
            result = _get_client_secret("entra", "https://login.microsoftonline.com", "client-id")
            assert result == "entra-secret"

    def test_missing_env_returns_none(self):
        with patch.dict(os.environ, {}, clear=True):
            result = _get_client_secret("auth0", "https://myorg.auth0.com", "client-id")
            assert result is None

    def test_unknown_vendor_returns_none(self):
        result = _get_client_secret("unknown", "https://custom.example.com", "client-id")
        assert result is None


# ---------------------------------------------------------------------------
# _get_token_endpoint
# ---------------------------------------------------------------------------


class TestGetTokenEndpoint:
    """Tests for OIDC discovery token endpoint extraction."""

    @patch("cli.agentcore.token_refresher.requests.get")
    def test_extracts_token_endpoint(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "token_endpoint": "https://auth.example.com/oauth2/token",
            "issuer": "https://auth.example.com",
        }
        mock_get.return_value = mock_response

        result = _get_token_endpoint("https://auth.example.com/.well-known/openid-configuration")
        assert result == "https://auth.example.com/oauth2/token"

    @patch("cli.agentcore.token_refresher.requests.get")
    def test_returns_none_on_error(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")

        result = _get_token_endpoint("https://unreachable.example.com")
        assert result is None


# ---------------------------------------------------------------------------
# _request_token
# ---------------------------------------------------------------------------


class TestRequestToken:
    """Tests for OAuth2 client_credentials token request."""

    @patch("cli.agentcore.token_refresher.requests.post")
    def test_successful_token_request(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "eyJtoken123"}
        mock_post.return_value = mock_response

        result = _request_token(
            "https://auth.example.com/oauth2/token",
            "client-id",
            "client-secret",
        )

        assert result == "eyJtoken123"
        mock_post.assert_called_once()
        call_data = mock_post.call_args[1]["data"]
        assert call_data["grant_type"] == "client_credentials"
        assert call_data["client_id"] == "client-id"
        assert call_data["client_secret"] == "client-secret"

    @patch("cli.agentcore.token_refresher.requests.post")
    def test_returns_none_on_error(self, mock_post):
        mock_post.side_effect = Exception("401 Unauthorized")

        result = _request_token("https://auth.example.com/token", "id", "secret")
        assert result is None


# ---------------------------------------------------------------------------
# _update_registry_credential
# ---------------------------------------------------------------------------


class TestUpdateRegistryCredential:
    """Tests for PATCH auth_credential in registry."""

    @patch("cli.agentcore.token_refresher.requests.patch")
    def test_successful_update(self, mock_patch):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_patch.return_value = mock_response

        result = _update_registry_credential(
            "https://registry.example.com",
            "registry-token",
            "/my-server",
            "eyJnewtoken",
        )

        assert result is True
        mock_patch.assert_called_once()
        url = mock_patch.call_args[0][0]
        assert url == "https://registry.example.com/api/servers/my-server/auth-credential"

    @patch("cli.agentcore.token_refresher.requests.patch")
    def test_returns_false_on_error(self, mock_patch):
        mock_patch.side_effect = Exception("500 Server Error")

        result = _update_registry_credential(
            "https://registry.example.com",
            "token",
            "/server",
            "cred",
        )

        assert result is False


# ---------------------------------------------------------------------------
# _load_registry_token
# ---------------------------------------------------------------------------


class TestLoadRegistryToken:
    """Tests for loading registry auth token from file."""

    def test_loads_access_token(self, tmp_path):
        token_file = tmp_path / ".token"
        token_file.write_text(json.dumps({"access_token": "my-jwt-token"}))

        result = _load_registry_token(str(token_file))
        assert result == "my-jwt-token"

    def test_loads_token_field(self, tmp_path):
        token_file = tmp_path / ".token"
        token_file.write_text(json.dumps({"token": "alt-token"}))

        result = _load_registry_token(str(token_file))
        assert result == "alt-token"

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_registry_token(str(tmp_path / "missing.json"))

    def test_raises_on_missing_token_field(self, tmp_path):
        token_file = tmp_path / ".token"
        token_file.write_text(json.dumps({"other": "field"}))

        with pytest.raises(ValueError, match="No access_token or token"):
            _load_registry_token(str(token_file))


# ---------------------------------------------------------------------------
# refresh_all (end-to-end)
# ---------------------------------------------------------------------------


class TestTriggerSecurityScan:
    """Tests for triggering security rescan after credential update."""

    @patch("cli.agentcore.token_refresher.requests.post")
    def test_successful_scan(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "is_safe": True,
            "critical_issues": 0,
            "high_severity": 0,
        }
        mock_post.return_value = mock_response

        result = _trigger_security_scan(
            "https://registry.example.com",
            "registry-token",
            "/my-server",
        )

        assert result is True
        mock_post.assert_called_once()
        url = mock_post.call_args[0][0]
        assert url == "https://registry.example.com/api/servers/my-server/rescan"

    @patch("cli.agentcore.token_refresher.requests.post")
    def test_scan_with_findings(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "is_safe": False,
            "critical_issues": 1,
            "high_severity": 2,
        }
        mock_post.return_value = mock_response

        result = _trigger_security_scan(
            "https://registry.example.com",
            "token",
            "/my-server",
        )

        assert result is True

    @patch("cli.agentcore.token_refresher.requests.post")
    def test_scan_forbidden_returns_false(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 403
        http_error = requests.exceptions.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error
        mock_post.return_value = mock_response

        result = _trigger_security_scan(
            "https://registry.example.com",
            "non-admin-token",
            "/my-server",
        )

        assert result is False

    @patch("cli.agentcore.token_refresher.requests.post")
    def test_scan_error_returns_false(self, mock_post):
        mock_post.side_effect = Exception("Connection refused")

        result = _trigger_security_scan(
            "https://registry.example.com",
            "token",
            "/my-server",
        )

        assert result is False


# ---------------------------------------------------------------------------
# refresh_all (end-to-end)
# ---------------------------------------------------------------------------


class TestRefreshAll:
    """Tests for the end-to-end refresh_all flow."""

    def _write_manifest(self, tmp_path, entries):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(entries))
        return str(manifest)

    @patch("cli.agentcore.token_refresher._update_registry_credential")
    @patch("cli.agentcore.token_refresher._request_token")
    @patch("cli.agentcore.token_refresher._get_token_endpoint")
    @patch("cli.agentcore.token_refresher._get_client_secret")
    def test_refresh_cognito_success(
        self, mock_secret, mock_endpoint, mock_token, mock_update, tmp_path
    ):
        mock_secret.return_value = "cognito-secret"
        mock_endpoint.return_value = "https://cognito.example.com/oauth2/token"
        mock_token.return_value = "eyJnewtoken"
        mock_update.return_value = True

        manifest_path = self._write_manifest(tmp_path, [{
            "server_path": "/my-gw",
            "gateway_arn": "arn:aws:bedrock:us-east-1:123:gateway/gw-1",
            "discovery_url": "https://cognito-idp.us-east-1.amazonaws.com/pool/.well-known/openid-configuration",
            "allowed_clients": ["client-1"],
            "idp_vendor": "cognito",
        }])

        summary = refresh_all(
            manifest_path, "https://registry.example.com", "reg-token",
            run_scan=False,
        )

        assert summary["success"] == 1
        assert summary["failed"] == 0
        assert summary["skipped"] == 0
        mock_update.assert_called_once_with(
            "https://registry.example.com", "reg-token", "/my-gw", "eyJnewtoken"
        )

    @patch("cli.agentcore.token_refresher._update_registry_credential")
    @patch("cli.agentcore.token_refresher._request_token")
    @patch("cli.agentcore.token_refresher._get_token_endpoint")
    @patch("cli.agentcore.token_refresher._get_client_secret")
    def test_refresh_mixed_idps(
        self, mock_secret, mock_endpoint, mock_token, mock_update, tmp_path
    ):
        mock_secret.side_effect = ["cognito-secret", "auth0-secret", None]
        mock_endpoint.return_value = "https://example.com/token"
        mock_token.return_value = "eyJtoken"
        mock_update.return_value = True

        entries = [
            {
                "server_path": "/gw-cognito",
                "gateway_arn": "arn:1",
                "discovery_url": "https://cognito-idp.example.com",
                "allowed_clients": ["c1"],
                "idp_vendor": "cognito",
            },
            {
                "server_path": "/gw-auth0",
                "gateway_arn": "arn:2",
                "discovery_url": "https://myorg.auth0.com",
                "allowed_clients": ["c2"],
                "idp_vendor": "auth0",
            },
            {
                "server_path": "/gw-unknown",
                "gateway_arn": "arn:3",
                "discovery_url": "https://custom.example.com",
                "allowed_clients": ["c3"],
                "idp_vendor": "unknown",
            },
        ]
        manifest_path = self._write_manifest(tmp_path, entries)

        summary = refresh_all(
            manifest_path, "https://registry.example.com", "reg-token",
            run_scan=False,
        )

        assert summary["success"] == 2
        assert summary["skipped"] == 1
        assert summary["total"] == 3

    @patch("cli.agentcore.token_refresher._get_client_secret")
    def test_refresh_skips_no_allowed_clients(self, mock_secret, tmp_path):
        manifest_path = self._write_manifest(tmp_path, [{
            "server_path": "/no-clients",
            "gateway_arn": "arn:1",
            "discovery_url": "https://example.com",
            "allowed_clients": [],
            "idp_vendor": "cognito",
        }])

        summary = refresh_all(
            manifest_path, "https://registry.example.com", "token",
            run_scan=False,
        )

        assert summary["skipped"] == 1
        mock_secret.assert_not_called()

    @patch("cli.agentcore.token_refresher._update_registry_credential")
    @patch("cli.agentcore.token_refresher._request_token")
    @patch("cli.agentcore.token_refresher._get_token_endpoint")
    @patch("cli.agentcore.token_refresher._get_client_secret")
    def test_refresh_writes_timestamps(
        self, mock_secret, mock_endpoint, mock_token, mock_update, tmp_path
    ):
        mock_secret.return_value = "secret"
        mock_endpoint.return_value = "https://example.com/token"
        mock_token.return_value = "eyJtoken"
        mock_update.return_value = True

        manifest_path = self._write_manifest(tmp_path, [{
            "server_path": "/gw",
            "gateway_arn": "arn:1",
            "discovery_url": "https://cognito-idp.example.com",
            "allowed_clients": ["c1"],
            "idp_vendor": "cognito",
        }])

        refresh_all(
            manifest_path, "https://registry.example.com", "token",
            run_scan=False,
        )

        updated = json.loads((tmp_path / "manifest.json").read_text())
        assert "last_refreshed" in updated[0]

    @patch("cli.agentcore.token_refresher._trigger_security_scan")
    @patch("cli.agentcore.token_refresher._update_registry_credential")
    @patch("cli.agentcore.token_refresher._request_token")
    @patch("cli.agentcore.token_refresher._get_token_endpoint")
    @patch("cli.agentcore.token_refresher._get_client_secret")
    def test_refresh_triggers_scan_after_update(
        self, mock_secret, mock_endpoint, mock_token, mock_update, mock_scan, tmp_path
    ):
        mock_secret.return_value = "secret"
        mock_endpoint.return_value = "https://example.com/token"
        mock_token.return_value = "eyJtoken"
        mock_update.return_value = True
        mock_scan.return_value = True

        manifest_path = self._write_manifest(tmp_path, [{
            "server_path": "/gw",
            "gateway_arn": "arn:1",
            "discovery_url": "https://cognito-idp.example.com",
            "allowed_clients": ["c1"],
            "idp_vendor": "cognito",
        }])

        summary = refresh_all(
            manifest_path, "https://registry.example.com", "token",
            run_scan=True,
        )

        assert summary["success"] == 1
        assert summary["scans_triggered"] == 1
        assert summary["scans_failed"] == 0
        mock_scan.assert_called_once_with(
            "https://registry.example.com", "token", "/gw"
        )

    @patch("cli.agentcore.token_refresher._trigger_security_scan")
    @patch("cli.agentcore.token_refresher._update_registry_credential")
    @patch("cli.agentcore.token_refresher._request_token")
    @patch("cli.agentcore.token_refresher._get_token_endpoint")
    @patch("cli.agentcore.token_refresher._get_client_secret")
    def test_refresh_no_scan_when_disabled(
        self, mock_secret, mock_endpoint, mock_token, mock_update, mock_scan, tmp_path
    ):
        mock_secret.return_value = "secret"
        mock_endpoint.return_value = "https://example.com/token"
        mock_token.return_value = "eyJtoken"
        mock_update.return_value = True

        manifest_path = self._write_manifest(tmp_path, [{
            "server_path": "/gw",
            "gateway_arn": "arn:1",
            "discovery_url": "https://cognito-idp.example.com",
            "allowed_clients": ["c1"],
            "idp_vendor": "cognito",
        }])

        summary = refresh_all(
            manifest_path, "https://registry.example.com", "token",
            run_scan=False,
        )

        assert summary["success"] == 1
        assert "scans_triggered" not in summary
        mock_scan.assert_not_called()

    @patch("cli.agentcore.token_refresher._trigger_security_scan")
    @patch("cli.agentcore.token_refresher._update_registry_credential")
    @patch("cli.agentcore.token_refresher._request_token")
    @patch("cli.agentcore.token_refresher._get_token_endpoint")
    @patch("cli.agentcore.token_refresher._get_client_secret")
    def test_refresh_scan_failure_tracked(
        self, mock_secret, mock_endpoint, mock_token, mock_update, mock_scan, tmp_path
    ):
        mock_secret.return_value = "secret"
        mock_endpoint.return_value = "https://example.com/token"
        mock_token.return_value = "eyJtoken"
        mock_update.return_value = True
        mock_scan.return_value = False

        manifest_path = self._write_manifest(tmp_path, [{
            "server_path": "/gw",
            "gateway_arn": "arn:1",
            "discovery_url": "https://cognito-idp.example.com",
            "allowed_clients": ["c1"],
            "idp_vendor": "cognito",
        }])

        summary = refresh_all(
            manifest_path, "https://registry.example.com", "token",
            run_scan=True,
        )

        assert summary["success"] == 1
        assert summary["scans_triggered"] == 0
        assert summary["scans_failed"] == 1

    @patch("cli.agentcore.token_refresher._trigger_security_scan")
    @patch("cli.agentcore.token_refresher._update_registry_credential")
    @patch("cli.agentcore.token_refresher._request_token")
    @patch("cli.agentcore.token_refresher._get_token_endpoint")
    @patch("cli.agentcore.token_refresher._get_client_secret")
    def test_refresh_no_scan_on_failed_update(
        self, mock_secret, mock_endpoint, mock_token, mock_update, mock_scan, tmp_path
    ):
        mock_secret.return_value = "secret"
        mock_endpoint.return_value = "https://example.com/token"
        mock_token.return_value = "eyJtoken"
        mock_update.return_value = False

        manifest_path = self._write_manifest(tmp_path, [{
            "server_path": "/gw",
            "gateway_arn": "arn:1",
            "discovery_url": "https://cognito-idp.example.com",
            "allowed_clients": ["c1"],
            "idp_vendor": "cognito",
        }])

        summary = refresh_all(
            manifest_path, "https://registry.example.com", "token",
            run_scan=True,
        )

        assert summary["failed"] == 1
        assert summary["scans_triggered"] == 0
        mock_scan.assert_not_called()
