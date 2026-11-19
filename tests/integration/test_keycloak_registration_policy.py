"""Opt-in registration policy checks against a disposable Keycloak instance.

Set RUN_KEYCLOAK_REGISTRATION_INTEGRATION=1 to test the official Keycloak image.
The container binds only loopback and is removed even when a test fails.
"""

import base64
import hashlib
import json
import os
import secrets
import subprocess
import time
import uuid
from html.parser import HTMLParser
from http.cookiejar import CookieJar, DefaultCookiePolicy
from pathlib import Path
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, quote, urljoin, urlsplit

import httpx
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_KEYCLOAK_REGISTRATION_INTEGRATION") != "1",
    reason="requires Docker, the official Keycloak image and explicit opt-in",
)
ROOT = Path(__file__).resolve().parents[2]
IMAGE = "quay.io/keycloak/keycloak:26.8.2"
REALM = "mcp-gateway"
CALLBACK = "http://127.0.0.1:33418/"
VENDOR_CLIENT = "https://vscode.dev/oauth/client-metadata.json"


def _docker(
    *args: str,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        env=env,
        check=check,
        timeout=180,
    )


class _FormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.action: str | None = None
        self.fields: dict[str, str] = {}

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Read the login and consent fields without executing page content."""
        values = dict(attrs)
        if tag == "form" and values.get("action"):
            self.action = values["action"]
        if tag == "input" and values.get("name") and values.get("type") == "hidden":
            self.fields[values["name"]] = values.get("value") or ""


def _admin_token(client: httpx.Client, password: str) -> str:
    response = client.post(
        "/realms/master/protocol/openid-connect/token",
        data={
            "grant_type": "password",
            "client_id": "admin-cli",
            "username": "policy-admin",
            "password": password,
        },
    )
    response.raise_for_status()
    return response.json()["access_token"]


def _admin(
    client: httpx.Client,
    password: str,
    method: str,
    path: str,
    body: dict | list | None = None,
) -> httpx.Response:
    response = client.request(
        method,
        f"/admin/realms/{REALM}{path}",
        headers={"Authorization": f"Bearer {_admin_token(client, password)}"},
        json=body,
    )
    response.raise_for_status()
    return response


def _configure(
    client: httpx.Client,
    password: str,
    *,
    dcr: bool = False,
    cimd: bool = True,
    domains: list[str] | None = None,
    keep: str = "",
) -> None:
    env = {
        **os.environ,
        "KEYCLOAK_ADMIN_URL": str(client.base_url).rstrip("/"),
        "KEYCLOAK_ADMIN": "policy-admin",
        "KEYCLOAK_ADMIN_PASSWORD": password,
        "ENABLE_DCR_CONFIG": str(dcr).lower(),
        "ENABLE_CIMD_CONFIG": str(cimd).lower(),
        "DCR_SWEEP_ACKNOWLEDGED": "true",
        "DCR_KEEP_CLIENTS": keep,
        "REGISTRATION_TRUSTED_CLIENT_IDS": "[]",
        "CIMD_TRUSTED_DOMAINS": json.dumps(
            domains or ["vscode.dev", "code.visualstudio.com", "localhost", "127.0.0.1"]
        ),
    }
    result = subprocess.run(
        ["bash", str(ROOT / "keycloak/setup/upgrade-realm-for-dcr.sh")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _lifecycle_helper(
    client: httpx.Client,
    password: str,
    command: str,
    *,
    cimd: bool = False,
    domains: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "KEYCLOAK_URL": str(client.base_url).rstrip("/"),
        "REALM": REALM,
        "TOKEN": _admin_token(client, password),
        "ENABLE_CIMD_CONFIG": str(cimd).lower(),
        "CIMD_TRUSTED_DOMAINS": json.dumps(domains or ["vscode.dev"]),
        "DCR_SWEEP_ACKNOWLEDGED": "true",
    }
    return subprocess.run(
        [
            "bash",
            "-c",
            "source keycloak/setup/group-path-preflight.sh\n"
            'realm_curl() { credential_curl "$@"; }\n'
            '_admin_get() { realm_curl -fsS -H "Authorization: Bearer $1" "$KEYCLOAK_URL$2"; }\n'
            '_list_all_clients() { _admin_get "$1" "/admin/realms/$REALM/clients?max=10000"; }\n'
            + command,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.fixture(scope="module")
def keycloak():
    name = "kc-registration-test-" + uuid.uuid4().hex[:12]
    password = secrets.token_urlsafe(36)
    env = {
        **os.environ,
        "KC_BOOTSTRAP_ADMIN_USERNAME": "policy-admin",
        "KC_BOOTSTRAP_ADMIN_PASSWORD": password,
    }
    _docker(
        "run",
        "-d",
        "--name",
        name,
        "-p",
        "127.0.0.1::8080",
        "-e",
        "KC_BOOTSTRAP_ADMIN_USERNAME",
        "-e",
        "KC_BOOTSTRAP_ADMIN_PASSWORD",
        "-e",
        "KC_DB=dev-file",
        "-e",
        "KC_FEATURES=cimd",
        "-e",
        "KC_FEATURES_DISABLED=device-flow,ciba",
        "-e",
        "KC_SPI_CLIENT_REGISTRATION__DEFAULT__ENABLED=false",
        IMAGE,
        "start-dev",
        env=env,
    )
    try:
        binding = _docker("port", name, "8080/tcp").stdout.strip()
        with httpx.Client(base_url=f"http://{binding}", timeout=20, trust_env=False) as client:
            for _ in range(120):
                try:
                    if client.get("/realms/master").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(1)
            else:
                pytest.fail("Disposable Keycloak did not become ready within 120 seconds")
            response = client.post(
                "/admin/realms",
                headers={"Authorization": f"Bearer {_admin_token(client, password)}"},
                json={"realm": REALM, "enabled": True, "sslRequired": "none"},
            )
            response.raise_for_status()
            for client_id in ("mcp-gateway-web", "mcp-gateway-m2m"):
                _admin(
                    client,
                    password,
                    "POST",
                    "/clients",
                    {
                        "clientId": client_id,
                        "protocol": "openid-connect",
                        "enabled": True,
                        "publicClient": False,
                        "serviceAccountsEnabled": True,
                        "standardFlowEnabled": client_id.endswith("web"),
                        "directAccessGrantsEnabled": False,
                        "redirectUris": [CALLBACK] if client_id.endswith("web") else [],
                    },
                )
            _admin(
                client,
                password,
                "POST",
                "/client-scopes",
                {
                    "name": "registry-admins",
                    "protocol": "openid-connect",
                },
            )
            _admin(
                client,
                password,
                "POST",
                "/users",
                {
                    "username": "policy-user",
                    "enabled": True,
                    "firstName": "Policy",
                    "lastName": "Test",
                    "email": "policy-user@example.invalid",
                    "emailVerified": True,
                    "credentials": [{"type": "password", "value": password, "temporary": False}],
                },
            )
            _configure(client, password)
            yield client, password
    finally:
        _docker("rm", "-f", name, check=False)


def _code_token(
    client: httpx.Client,
    password: str,
    client_id: str,
    scope: str = "openid profile email",
    *,
    return_code: bool = False,
) -> dict:
    verifier = secrets.token_urlsafe(48)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    )
    # Browsers accept Secure cookies on loopback. Python's cookiejar requires
    # an explicit policy for this disposable HTTP-only loopback client.
    cookies = CookieJar(policy=DefaultCookiePolicy(secure_protocols=("https", "wss", "http")))

    def _send_loopback_cookies(request: httpx.Request) -> None:
        assert request.url.host == "127.0.0.1"
        request.headers.pop("cookie", None)
        httpx.Cookies(cookies).set_cookie_header(request)

    with httpx.Client(
        base_url=client.base_url,
        timeout=20,
        trust_env=False,
        cookies=cookies,
        event_hooks={"request": [_send_loopback_cookies]},
    ) as browser:
        response = browser.get(
            f"/realms/{REALM}/protocol/openid-connect/auth",
            params={
                "client_id": client_id,
                "redirect_uri": CALLBACK,
                "response_type": "code",
                "scope": scope,
                "state": "policy-test",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            },
        )
        for _ in range(5):
            if response.is_redirect:
                location = response.headers["location"]
                if location.startswith(CALLBACK):
                    query = parse_qs(urlsplit(location).query)
                    assert "code" in query, {
                        key: value for key, value in query.items() if key != "code"
                    }
                    if return_code:
                        return {"code": query["code"][0], "verifier": verifier}
                    token = browser.post(
                        f"/realms/{REALM}/protocol/openid-connect/token",
                        data={
                            "grant_type": "authorization_code",
                            "client_id": client_id,
                            "redirect_uri": CALLBACK,
                            "code": query["code"][0],
                            "code_verifier": verifier,
                        },
                    )
                    token.raise_for_status()
                    return token.json()
                destination = urljoin(str(response.url), location)
                assert urlsplit(destination).netloc == urlsplit(str(client.base_url)).netloc
                response = browser.get(destination)
                continue
            if response.status_code != 200:
                events = _admin(client, password, "GET", "/events?type=LOGIN_ERROR&max=1").json()
                errors = [
                    {
                        "error": event.get("error"),
                        "reason": event.get("details", {}).get("reason"),
                        "policy_error": event.get("details", {}).get("client_policy_error_detail"),
                    }
                    for event in events
                ]
                pytest.fail(f"Authorization failed: HTTP {response.status_code}; {errors}")
            form = _FormParser()
            form.feed(response.text)
            assert form.action, "Expected login or consent form"
            fields = {
                **form.fields,
                "username": "policy-user",
                "password": password,
                "accept": "Yes",
            }
            destination = urljoin(str(response.url), form.action)
            assert urlsplit(destination).netloc == urlsplit(str(client.base_url)).netloc
            response = browser.post(destination, data=fields)
    pytest.fail("Authorization did not produce a code after login and consent")


def _client_record(client: httpx.Client, password: str, client_id: str) -> dict:
    records = _admin(
        client, password, "GET", "/clients?clientId=" + quote(client_id, safe="")
    ).json()
    assert len(records) == 1
    return records[0]


@pytest.mark.parametrize("requested_scope", [None, "openid profile email offline_access"])
def test_dcr_omitted_scopes_are_constrained_and_tokens_refresh(keycloak, requested_scope):
    client, password = keycloak
    _configure(client, password, dcr=True, cimd=False)
    metadata = {
        "client_name": "policy-test-browser",
        "redirect_uris": [CALLBACK],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }
    if requested_scope is not None:
        metadata["scope"] = requested_scope
    response = client.post(
        f"/realms/{REALM}/clients-registrations/openid-connect",
        json=metadata,
    )
    assert response.status_code == 201, f"DCR failed: HTTP {response.status_code}"
    client_id = response.json()["client_id"]
    record = _client_record(client, password, client_id)
    for kind in ("default", "optional"):
        scopes = _admin(
            client, password, "GET", f"/clients/{record['id']}/{kind}-client-scopes"
        ).json()
        assert {item["name"] for item in scopes} <= {
            "basic",
            "profile",
            "email",
            "roles",
            "web-origins",
            "acr",
            "offline_access",
        }
    tokens = _code_token(client, password, client_id)
    claims = json.loads(base64.urlsafe_b64decode(tokens["access_token"].split(".")[1] + "=="))
    assert "mcp-gateway" in claims["aud"]
    assert "registry-admins" not in claims["scope"].split()
    response = client.post(
        f"/realms/{REALM}/protocol/openid-connect/token",
        data={
            "grant_type": "refresh_token",
            "client_id": client_id,
            "refresh_token": tokens["refresh_token"],
        },
    )
    assert response.status_code == 200


def test_dcr_explicit_privileged_scope_is_rejected(keycloak):
    client, _ = keycloak
    _configure(client, keycloak[1], dcr=True, cimd=False)
    response = client.post(
        f"/realms/{REALM}/clients-registrations/openid-connect",
        json={
            "redirect_uris": [CALLBACK],
            "token_endpoint_auth_method": "none",
            "scope": "openid registry-admins",
        },
    )
    assert response.status_code in (400, 403)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_uris", ["http://127.0.0.1/oidc.jwt"]),
        ("jwks_uri", "http://127.0.0.1/keys.json"),
        ("sector_identifier_uri", "http://127.0.0.1/sector.json"),
        ("backchannel_logout_uri", "http://127.0.0.1/logout"),
        ("backchannel_client_notification_endpoint", "http://127.0.0.1/ciba"),
    ],
)
def test_dcr_rejects_server_fetch_urls_on_creation_and_update(keycloak, field, value):
    client, password = keycloak
    _configure(client, password, dcr=True, cimd=False)
    metadata = {
        "redirect_uris": [CALLBACK],
        "token_endpoint_auth_method": "none",
        "grant_types": ["authorization_code", "refresh_token"],
    }
    if field == "sector_identifier_uri":
        metadata["subject_type"] = "pairwise"
    rejected = client.post(
        f"/realms/{REALM}/clients-registrations/openid-connect",
        json={**metadata, field: value},
    )
    assert rejected.status_code in (400, 403)
    assert rejected.json().get("error_description") == "invalid_client_metadata"

    created = client.post(f"/realms/{REALM}/clients-registrations/openid-connect", json=metadata)
    assert created.status_code == 201
    registration = created.json()
    headers = {"Authorization": f"Bearer {registration['registration_access_token']}"}
    try:
        updated = client.put(
            registration["registration_client_uri"],
            headers=headers,
            json={**metadata, "client_id": registration["client_id"], field: value},
        )
        assert updated.status_code in (400, 403)
        assert updated.json().get("error_description") == "invalid_client_metadata"
        unchanged = client.get(registration["registration_client_uri"], headers=headers)
        assert unchanged.status_code == 200
        assert not unchanged.json().get(field)
    finally:
        assert (
            client.delete(registration["registration_client_uri"], headers=headers).status_code
            == 204
        )


def test_switching_dcr_and_cimd_in_both_directions(keycloak):
    client, password = keycloak
    registrations = []
    try:
        _configure(client, password, dcr=True, cimd=False)
        registration = client.post(
            f"/realms/{REALM}/clients-registrations/openid-connect",
            json={
                "redirect_uris": [CALLBACK],
                "token_endpoint_auth_method": "none",
                "grant_types": ["authorization_code", "refresh_token"],
            },
        )
        assert registration.status_code == 201
        registrations.append(registration.json())
        assert "access_token" in _code_token(client, password, registrations[0]["client_id"])

        _configure(client, password, dcr=False, cimd=True)
        policies = _admin(client, password, "GET", "/client-policies/policies").json()["policies"]
        assert not any(p["name"] == "mcp-no-remote-request-objects" for p in policies)
        tokens = _code_token(client, password, VENDOR_CLIENT)
        refreshed = client.post(
            f"/realms/{REALM}/protocol/openid-connect/token",
            data={
                "client_id": VENDOR_CLIENT,
                "grant_type": "refresh_token",
                "refresh_token": tokens["refresh_token"],
            },
        )
        assert refreshed.status_code == 200

        _configure(client, password, dcr=True, cimd=False)
        policies = _admin(client, password, "GET", "/client-policies/policies").json()["policies"]
        assert any(p["name"] == "mcp-no-remote-request-objects" for p in policies)
        assert _client_record(client, password, VENDOR_CLIENT)["enabled"] is False
        registration = client.post(
            f"/realms/{REALM}/clients-registrations/openid-connect",
            json={"redirect_uris": [CALLBACK], "token_endpoint_auth_method": "none"},
        )
        assert registration.status_code == 201
        registrations.append(registration.json())
    finally:
        for registration in registrations:
            record = _admin(
                client,
                password,
                "GET",
                "/clients?clientId=" + quote(registration["client_id"], safe=""),
            ).json()
            for entry in record:
                _admin(client, password, "DELETE", f"/clients/{entry['id']}")
        _configure(client, password, dcr=False, cimd=True)


def test_cimd_metadata_refresh_after_guard_removal(keycloak):
    client, password = keycloak
    _configure(client, password, dcr=True, cimd=False)
    _configure(client, password, dcr=False, cimd=True)
    records = _admin(
        client, password, "GET", "/clients?clientId=" + quote(VENDOR_CLIENT, safe="")
    ).json()
    if records:
        # Setup deliberately keeps previously disabled clients disabled.
        _admin(client, password, "PUT", f"/clients/{records[0]['id']}", {"enabled": True})
    _code_token(client, password, VENDOR_CLIENT)
    record = _client_record(client, password, VENDOR_CLIENT)
    _admin(
        client,
        password,
        "PUT",
        f"/clients/{record['id']}",
        {"attributes": {**record["attributes"], "cimd.cache.expiry.time.in.sec": "1"}},
    )
    refreshed = _code_token(client, password, VENDOR_CLIENT)
    assert "access_token" in refreshed
    assert (
        int(
            _client_record(client, password, VENDOR_CLIENT)["attributes"][
                "cimd.cache.expiry.time.in.sec"
            ]
        )
        > 1
    )


@pytest.mark.parametrize("dcr", [False, True])
def test_cimd_disable_revokes_persisted_client_despite_keep_list(keycloak, dcr):
    client, password = keycloak
    client_id = "https://vscode.dev/policy-test-" + uuid.uuid4().hex + ".json"
    _admin(
        client,
        password,
        "POST",
        "/clients",
        {
            "clientId": client_id,
            "protocol": "openid-connect",
            "enabled": True,
            "publicClient": True,
            "standardFlowEnabled": True,
            "redirectUris": [CALLBACK],
            "attributes": {"cimd.cache.expiry.time.in.sec": str(int(time.time()) + 3600)},
        },
    )
    try:
        _configure(client, password, dcr=dcr, cimd=False, keep=client_id)
        record = _client_record(client, password, client_id)
        assert record["enabled"] is False
        assert record.get("notBefore", 0) > 0
        response = client.get(
            f"/realms/{REALM}/protocol/openid-connect/auth",
            params={
                "client_id": client_id,
                "redirect_uri": CALLBACK,
                "response_type": "code",
                "scope": "openid",
            },
        )
        assert response.status_code == 400
    finally:
        _configure(client, password)


def test_cimd_trust_withdrawal_disables_existing_client(keycloak):
    client, password = keycloak
    client_id = "https://vscode.dev/policy-test-" + uuid.uuid4().hex + ".json"
    _admin(
        client,
        password,
        "POST",
        "/clients",
        {
            "clientId": client_id,
            "protocol": "openid-connect",
            "enabled": True,
            "publicClient": True,
            "standardFlowEnabled": True,
            "redirectUris": [CALLBACK],
            "attributes": {"cimd.cache.expiry.time.in.sec": str(int(time.time()) + 3600)},
        },
    )
    try:
        _configure(client, password, domains=["claude.ai", "127.0.0.1"], keep=client_id)
        assert _client_record(client, password, client_id)["enabled"] is False
    finally:
        _configure(client, password)


def test_cimd_vendor_login_refresh_and_disable(keycloak):
    client, password = keycloak
    _configure(client, password, dcr=False, cimd=True)
    records = _admin(
        client, password, "GET", "/clients?clientId=" + quote(VENDOR_CLIENT, safe="")
    ).json()
    if records:
        _admin(client, password, "PUT", f"/clients/{records[0]['id']}", {"enabled": True})
    tokens = _code_token(client, password, VENDOR_CLIENT)
    claims = json.loads(base64.urlsafe_b64decode(tokens["access_token"].split(".")[1] + "=="))
    assert "mcp-gateway" in claims["aud"]
    assert "registry-admins" not in claims["scope"].split()
    response = client.post(
        f"/realms/{REALM}/protocol/openid-connect/token",
        data={
            "grant_type": "refresh_token",
            "client_id": VENDOR_CLIENT,
            "refresh_token": tokens["refresh_token"],
        },
    )
    assert response.status_code == 200
    refreshed = response.json()
    try:
        _configure(client, password, cimd=False, keep=VENDOR_CLIENT)
        response = client.post(
            f"/realms/{REALM}/protocol/openid-connect/token",
            data={
                "grant_type": "refresh_token",
                "client_id": VENDOR_CLIENT,
                "refresh_token": refreshed["refresh_token"],
            },
        )
        assert response.status_code in (400, 401)
    finally:
        _configure(client, password)
    assert _client_record(client, password, VENDOR_CLIENT)["enabled"] is False


def test_operator_machine_retains_privileged_scopes(keycloak):
    client, password = keycloak
    record = _client_record(client, password, "mcp-gateway-m2m")
    scopes = _admin(client, password, "GET", "/client-scopes").json()
    admin_scope = next(scope["id"] for scope in scopes if scope["name"] == "registry-admins")
    _admin(client, password, "PUT", f"/clients/{record['id']}/default-client-scopes/{admin_scope}")
    secret = _admin(client, password, "GET", f"/clients/{record['id']}/client-secret").json()[
        "value"
    ]
    _configure(client, password)
    response = client.post(
        f"/realms/{REALM}/protocol/openid-connect/token",
        data={
            "grant_type": "client_credentials",
            "client_id": "mcp-gateway-m2m",
            "client_secret": secret,
        },
    )
    assert response.status_code == 200
    assert "registry-admins" in response.json()["scope"].split()


def test_first_enable_refuses_unexpected_cimd_without_deleting(keycloak):
    client, password = keycloak
    policies = _admin(client, password, "GET", "/client-policies/policies").json()
    client_id = "https://vscode.dev/unexpected-" + uuid.uuid4().hex + ".json"
    _admin(
        client,
        password,
        "POST",
        "/clients",
        {
            "clientId": client_id,
            "protocol": "openid-connect",
            "enabled": False,
            "attributes": {"cimd.cache.expiry.time.in.sec": "2147483647"},
        },
    )
    record = _client_record(client, password, client_id)
    try:
        _admin(
            client,
            password,
            "PUT",
            "/client-policies/policies",
            {
                "policies": [
                    p for p in policies["policies"] if p["name"] != "mcp-cimd-runtime-guard-policy"
                ]
            },
        )
        result = _lifecycle_helper(client, password, '_check_cimd_first_enable "$TOKEN"')
        assert result.returncode != 0
        assert "Administrator review required" in result.stderr
        assert _client_record(client, password, client_id)["enabled"] is False
    finally:
        _admin(client, password, "PUT", "/client-policies/policies", policies)
        _admin(client, password, "DELETE", f"/clients/{record['id']}")


def test_native_cimd_privileged_assignment_blocks_login_and_refresh(keycloak):
    client, password = keycloak
    client_id = "https://vscode.dev/scope-test-" + uuid.uuid4().hex + ".json"
    _admin(
        client,
        password,
        "POST",
        "/clients",
        {
            "clientId": client_id,
            "protocol": "openid-connect",
            "enabled": True,
            "publicClient": True,
            "standardFlowEnabled": True,
            "redirectUris": [CALLBACK],
            "attributes": {"cimd.cache.expiry.time.in.sec": str(int(time.time()) + 3600)},
        },
    )
    tokens = _code_token(client, password, client_id)
    record = _client_record(client, password, client_id)
    scopes = _admin(client, password, "GET", "/client-scopes").json()
    scope_id = next(scope["id"] for scope in scopes if scope["name"] == "registry-admins")
    _admin(client, password, "PUT", f"/clients/{record['id']}/default-client-scopes/{scope_id}")
    try:
        response = client.post(
            f"/realms/{REALM}/protocol/openid-connect/token",
            data={
                "grant_type": "refresh_token",
                "client_id": client_id,
                "refresh_token": tokens["refresh_token"],
            },
        )
        assert response.status_code in (400, 401)
        response = client.get(
            f"/realms/{REALM}/protocol/openid-connect/auth",
            params={
                "client_id": client_id,
                "redirect_uri": CALLBACK,
                "response_type": "code",
                "scope": "openid registry-admins",
                "code_challenge": "a" * 43,
                "code_challenge_method": "S256",
            },
        )
        assert response.status_code == 400 or "error=" in response.headers.get("location", "")
    finally:
        _admin(client, password, "DELETE", f"/clients/{record['id']}")


def test_dcr_groups_authorize_gateway_and_refresh_removes_revoked_grants(keycloak, monkeypatch):
    client, password = keycloak
    _configure(client, password, dcr=True, cimd=False)
    response = client.post(
        f"/realms/{REALM}/clients-registrations/openid-connect",
        json={
            "client_name": "group-policy-browser",
            "redirect_uris": [CALLBACK],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
            "scope": "openid profile email",
        },
    )
    assert response.status_code == 201
    client_id = response.json()["client_id"]
    created = _admin(client, password, "POST", "/groups", {"name": "finance"})
    parent_id = created.headers["location"].rsplit("/", 1)[1]
    nested = _admin(client, password, "POST", f"/groups/{parent_id}/children", {"name": "readers"})
    group_id = nested.headers["location"].rsplit("/", 1)[1]
    user_id = _admin(client, password, "GET", "/users?username=policy-user").json()[0]["id"]
    membership = f"/users/{user_id}/groups/{group_id}"
    _admin(client, password, "PUT", membership)
    try:
        tokens = _code_token(client, password, client_id)
        jwks = client.get(f"/realms/{REALM}/protocol/openid-connect/certs").json()
        monkeypatch.syspath_prepend(str(ROOT / "auth_server"))
        from fastapi.testclient import TestClient
        from providers.keycloak import KeycloakProvider

        from auth_server import server

        provider = KeycloakProvider(
            keycloak_url=str(client.base_url),
            realm=REALM,
            client_id="mcp-gateway-web",
            client_secret="unused-in-bearer-validation",
            m2m_client_id="mcp-gateway-m2m",
        )
        monkeypatch.setattr(provider, "get_jwks", lambda: jwks)
        validated = provider.validate_token(tokens["access_token"])
        assert validated["groups"] == ["finance/readers"]
        repo = AsyncMock()
        repo.get_group_mappings_bulk.side_effect = lambda groups: (
            ["finance-read"] if "finance/readers" in groups else []
        )
        repo.get_server_scopes.side_effect = lambda scope: (
            [{"server": "finance", "methods": ["tools/call"], "tools": ["read_report"]}]
            if scope == "finance-read"
            else []
        )
        monkeypatch.setattr(server, "get_auth_provider", lambda: provider)
        monkeypatch.setattr(server, "get_scope_repository", lambda: repo)
        monkeypatch.setattr(server, "_get_blocked_tools", AsyncMock(return_value=set()))
        monkeypatch.setattr(
            server, "reload_scopes_config", AsyncMock(return_value={"group_mappings": {}})
        )
        monkeypatch.setattr(server, "_audit_legacy_scopes_on_startup", AsyncMock())
        monkeypatch.setattr(server, "_build_static_token_map", AsyncMock())
        import mongodb_groups_enrichment

        monkeypatch.setattr(mongodb_groups_enrichment, "should_enrich_groups", lambda result: False)
        monkeypatch.setattr(
            mongodb_groups_enrichment, "should_enrich_user_groups", lambda *args: False
        )

        def _invoke(gateway, token, tool):
            return gateway.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Original-URL": "https://gateway.example/finance/mcp",
                    "X-Original-Method": "POST",
                    "X-Body": json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/call",
                            "params": {"name": tool},
                        }
                    ),
                },
            )

        with TestClient(server.app) as gateway:
            allowed = _invoke(gateway, tokens["access_token"], "read_report")
            assert allowed.status_code == 200, allowed.text
            assert allowed.json()["scopes"] == ["finance-read"]
            assert _invoke(gateway, tokens["access_token"], "delete_report").status_code == 403
            _admin(client, password, "DELETE", membership)
            refreshed = client.post(
                f"/realms/{REALM}/protocol/openid-connect/token",
                data={
                    "grant_type": "refresh_token",
                    "client_id": client_id,
                    "refresh_token": tokens["refresh_token"],
                },
            )
            assert refreshed.status_code == 200
            fresh_token = refreshed.json()["access_token"]
            assert provider.validate_token(fresh_token)["groups"] == []
            assert _invoke(gateway, fresh_token, "read_report").status_code == 403
    finally:
        _admin(client, password, "DELETE", membership)
        _admin(client, password, "DELETE", f"/groups/{parent_id}")


def test_native_registration_cannot_forge_cimd_provenance(keycloak):
    client, password = keycloak
    client_id = "https://vscode.dev/forged-client.json"
    response = client.post(
        f"/realms/{REALM}/clients-registrations/default",
        json={
            "clientId": client_id,
            "protocol": "openid-connect",
            "publicClient": True,
            "standardFlowEnabled": True,
            "redirectUris": [CALLBACK],
            "attributes": {"cimd.cache.expiry.time.in.sec": "2147483647"},
        },
    )
    assert response.status_code == 404
    records = _admin(
        client, password, "GET", "/clients?clientId=" + quote(client_id, safe="")
    ).json()
    assert records == []


@pytest.mark.parametrize("mechanism", ["dcr", "cimd"])
@pytest.mark.parametrize("pkce", ["missing", "plain"])
def test_public_registration_rejects_missing_or_plain_pkce(keycloak, mechanism, pkce):
    client, password = keycloak
    _configure(client, password, dcr=mechanism == "dcr", cimd=mechanism == "cimd")
    if mechanism == "dcr":
        registered = client.post(
            f"/realms/{REALM}/clients-registrations/openid-connect",
            json={
                "redirect_uris": [CALLBACK],
                "token_endpoint_auth_method": "none",
                "grant_types": ["authorization_code", "refresh_token"],
            },
        )
        assert registered.status_code == 201
        client_id = registered.json()["client_id"]
    else:
        # Use the real vendor document; earlier lifecycle tests may leave it disabled.
        records = _admin(
            client, password, "GET", "/clients?clientId=" + quote(VENDOR_CLIENT, safe="")
        ).json()
        if records:
            _admin(client, password, "PUT", f"/clients/{records[0]['id']}", {"enabled": True})
        client_id = VENDOR_CLIENT
    params = {
        "client_id": client_id,
        "redirect_uri": CALLBACK,
        "response_type": "code",
        "scope": "openid profile email",
        "state": "pkce-test",
    }
    if pkce == "plain":
        params.update(code_challenge="x" * 43, code_challenge_method="plain")
    response = client.get(f"/realms/{REALM}/protocol/openid-connect/auth", params=params)
    assert response.status_code == 400 or "error=" in response.headers.get("location", "")


@pytest.mark.parametrize("supplied_verifier", [None, "incorrect-verifier-" * 4])
def test_public_code_exchange_requires_bound_verifier(keycloak, supplied_verifier):
    client, password = keycloak
    _configure(client, password, dcr=True, cimd=False)
    registered = client.post(
        f"/realms/{REALM}/clients-registrations/openid-connect",
        json={
            "redirect_uris": [CALLBACK],
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"],
        },
    )
    assert registered.status_code == 201
    client_id = registered.json()["client_id"]
    authorization = _code_token(client, password, client_id, return_code=True)
    data = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "redirect_uri": CALLBACK,
        "code": authorization["code"],
    }
    if supplied_verifier is not None:
        data["code_verifier"] = supplied_verifier
    response = client.post(f"/realms/{REALM}/protocol/openid-connect/token", data=data)
    assert response.status_code == 400
    assert "access_token" not in response.json()


@pytest.mark.parametrize("cutover", ["disabled", "withdrawn", "failed-sweep"])
def test_cimd_runtime_guard_denies_late_client_code_refresh_and_login(keycloak, cutover):
    client, password = keycloak
    _configure(client, password)
    existing = _admin(
        client, password, "GET", "/clients?clientId=" + quote(VENDOR_CLIENT, safe="")
    ).json()
    if existing:
        _admin(client, password, "PUT", f"/clients/{existing[0]['id']}", {"enabled": True})
    tokens = _code_token(client, password, VENDOR_CLIENT)
    code = _code_token(client, password, VENDOR_CLIENT, return_code=True)
    record = _client_record(client, password, VENDOR_CLIENT)
    # A client that commits after the inventory is still enabled. Keep it enabled
    # here to prove the native runtime rule, independently of the cleanup sweep.
    assert record["attributes"]["display.on.consent.screen"] == "true"
    assert record["attributes"]["consent.screen.text"] == "The client's hostname is vscode.dev"
    try:
        command = '_configure_cimd_runtime_guard "$TOKEN"'
        if cutover == "failed-sweep":
            command = 'reconcile_cimd_clients() { return 1; }\n_prepare_cimd_lifecycle "$TOKEN"'
        result = _lifecycle_helper(
            client,
            password,
            command,
            cimd=cutover == "withdrawn",
            domains=["claude.ai"],
        )
        assert (result.returncode != 0) == (cutover == "failed-sweep"), (
            result.stdout + result.stderr
        )
        # Admin views must still work under denial so cleanup can inspect and disable.
        current = _admin(client, password, "GET", f"/clients/{record['id']}").json()
        assert current["enabled"] is True
        for grant in (
            {
                "grant_type": "authorization_code",
                "code": code["code"],
                "code_verifier": code["verifier"],
                "redirect_uri": CALLBACK,
            },
            {"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"]},
        ):
            response = client.post(
                f"/realms/{REALM}/protocol/openid-connect/token",
                data={"client_id": VENDOR_CLIENT, **grant},
            )
            assert response.status_code == 400
            assert "access_token" not in response.json()
        response = client.get(
            f"/realms/{REALM}/protocol/openid-connect/auth",
            params={
                "client_id": VENDOR_CLIENT,
                "redirect_uri": CALLBACK,
                "response_type": "code",
                "scope": "openid profile email",
                "code_challenge": "a" * 43,
                "code_challenge_method": "S256",
            },
        )
        assert response.status_code == 400 or "error=" in response.headers.get("location", "")
    finally:
        _configure(client, password)


def test_cimd_late_persisted_client_stays_denied_after_disable(keycloak):
    client, password = keycloak
    _configure(client, password)
    existing = _client_record(client, password, VENDOR_CLIENT)
    _admin(client, password, "PUT", f"/clients/{existing['id']}", {"enabled": True})
    _code_token(client, password, VENDOR_CLIENT)
    original = _client_record(client, password, VENDOR_CLIENT)
    late_id = "https://vscode.dev/late-" + uuid.uuid4().hex + ".json"
    try:
        _configure(client, password, cimd=False)
        # Simulate persistence completing after the setup inventory. Use attributes
        # observed on a real CIMD client rather than a fabricated policy fixture.
        _admin(
            client,
            password,
            "POST",
            "/clients",
            {
                "clientId": late_id,
                "protocol": "openid-connect",
                "enabled": True,
                "publicClient": True,
                "standardFlowEnabled": True,
                "redirectUris": [CALLBACK],
                "attributes": original["attributes"],
            },
        )
        response = client.get(
            f"/realms/{REALM}/protocol/openid-connect/auth",
            params={
                "client_id": late_id,
                "redirect_uri": CALLBACK,
                "response_type": "code",
                "scope": "openid",
                "code_challenge": "a" * 43,
                "code_challenge_method": "S256",
            },
        )
        assert response.status_code == 400 or "error=" in response.headers.get("location", "")
        _configure(client, password, dcr=True, cimd=False)
        # Standard OIDC DCR still works while CIMD is disabled.
        registered = client.post(
            f"/realms/{REALM}/clients-registrations/openid-connect",
            json={
                "redirect_uris": [CALLBACK],
                "token_endpoint_auth_method": "none",
                "grant_types": ["authorization_code", "refresh_token"],
            },
        )
        assert registered.status_code == 201
        assert "access_token" in _code_token(client, password, registered.json()["client_id"])
    finally:
        records = _admin(
            client, password, "GET", "/clients?clientId=" + quote(late_id, safe="")
        ).json()
        for record in records:
            _admin(client, password, "DELETE", f"/clients/{record['id']}")
        _configure(client, password)
