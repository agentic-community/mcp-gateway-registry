"""End-to-end test for OpenAPI MCP servers on a running local stack (issue #2022).

Skipped unless OPENAPI_MCP_E2E=1. Needs:
- the Compose stack with OPENAPI_MCP_ENABLED=true and the openapi-mcp profile;
- the mock REST API (tests/integration/openapi_mock_api.py) running on the
  compose network as ``openapi-mock-api:8000``, with that host in
  SSRF_ALLOWED_HOSTS for the registry and the openapi-mcp container;
- an admin token in ``.token`` (``.tokens.access_token``) or ACCESS_TOKEN.

Run:
    OPENAPI_MCP_E2E=1 uv run pytest tests/integration/test_openapi_mcp_e2e.py -v --no-cov
"""

import json
import os
import subprocess  # nosec B404 - restarts a local test container
import time
from pathlib import Path

import httpx
import pytest

from tests.integration.openapi_mock_api import (
    BASIC_PASSWORD,
    BASIC_USER,
    BEARER_TOKEN,
    HEADER_KEY,
    QUERY_KEY,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("OPENAPI_MCP_E2E") != "1", reason="set OPENAPI_MCP_E2E=1 on a running stack"),
]

REGISTRY_URL: str = os.environ.get("REGISTRY_URL", "http://localhost").rstrip("/")
MOCK_URL: str = os.environ.get("OPENAPI_MOCK_URL", "http://openapi-mock-api:8000")
CONTAINER_NAME: str = os.environ.get("OPENAPI_MCP_CONTAINER", "mcp-gateway-registry-openapi-mcp-1")
SERVER_NAME: str = "E2E OpenAPI"
SLUG: str = "e2e-openapi"
RESTART_WAIT_SECONDS: int = 90
TIMEOUT: float = 60.0
MCP_HEADERS: dict = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
META_2026: dict = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientInfo": {"name": "e2e", "version": "1"},
    "io.modelcontextprotocol/clientCapabilities": {},
}
GATEWAY_ONLY_HEADERS: tuple[str, ...] = (
    "x-user", "x-username", "x-scopes", "x-auth-method", "x-client-id", "x-forwarded-for",
    "x-real-ip", "x-resolved-upstream", "x-validate-source-secret", "cookie",
)


def _token() -> str:
    token = os.environ.get("ACCESS_TOKEN")
    if token:
        return token
    data = json.loads((Path(__file__).resolve().parents[2] / ".token").read_text())
    return data["tokens"]["access_token"]


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=REGISTRY_URL, headers={"Authorization": f"Bearer {_token()}"}, timeout=TIMEOUT) as c:
        yield c


def _parse(
    response: httpx.Response,
) -> dict:
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        for line in response.text.splitlines():
            if line.startswith("data:"):
                return json.loads(line[len("data:") :])
    return response.json()


def _call_2026(
    client: httpx.Client,
    tool: str,
    arguments: dict,
) -> dict:
    """Session-less call (MCP 2026-07-28)."""
    headers = {**MCP_HEADERS, "MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "tools/call", "Mcp-Name": tool}
    body = {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": tool, "arguments": arguments, "_meta": META_2026}}
    response = client.post(f"/{SLUG}/mcp", json=body, headers=headers)
    assert response.status_code == 200, response.text
    return _parse(response)


def _session_2025(
    client: httpx.Client,
) -> dict:
    """Handshake (MCP 2025-11-25). Returns the headers to send on later calls."""
    body = {
        "jsonrpc": "2.0", "id": 0, "method": "initialize",
        "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "e2e", "version": "1"}},
    }
    response = client.post(f"/{SLUG}/mcp", json=body, headers=MCP_HEADERS)
    assert response.status_code == 200, response.text
    headers = {**MCP_HEADERS, "MCP-Protocol-Version": "2025-11-25"}
    if response.headers.get("mcp-session-id"):
        headers["Mcp-Session-Id"] = response.headers["mcp-session-id"]
    client.post(f"/{SLUG}/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=headers)
    return headers


def _tool_json(
    result: dict,
) -> dict:
    structured = result["result"].get("structuredContent")
    if structured is not None:
        return structured
    return json.loads(result["result"]["content"][0]["text"])


def _add_target(
    client: httpx.Client,
    name: str,
    **body,
) -> httpx.Response:
    payload = {"name": name, "spec_url": f"{MOCK_URL}/spec.json", "base_url": MOCK_URL, **body}
    return client.post(f"/api/openapi-servers/{SLUG}/targets", json=payload)


@pytest.fixture(scope="module")
def server(client):
    client.delete(f"/api/openapi-servers/{SLUG}")
    response = client.post("/api/openapi-servers", json={"name": SERVER_NAME, "description": "end-to-end test"})
    assert response.status_code == 201, response.text
    targets = [
        ("items", {"include_tags": ["items", "admin", "debug"]}),
        ("hdr", {"include_operations": ["secureHeader"], "auth": {"scheme": "api_key_header", "header_name": "X-Api-Key", "credential": HEADER_KEY}}),
        ("qry", {"include_operations": ["secureQuery"], "auth": {"scheme": "api_key_query", "query_param": "apikey", "credential": QUERY_KEY}}),
        ("brr", {"include_operations": ["secureBearer"], "auth": {"scheme": "bearer", "credential": BEARER_TOKEN}}),
        ("bas", {"include_operations": ["secureBasic"], "auth": {"scheme": "basic", "username": BASIC_USER, "credential": BASIC_PASSWORD}}),
    ]
    for name, body in targets:
        added = _add_target(client, name, **body)
        assert added.status_code == 201, added.text
        assert added.json()["sync_status"] == "applied", added.text
    yield response.json()
    client.delete(f"/api/openapi-servers/{SLUG}")


class TestRejections:
    # FastMCP truncates a generated name to 56 characters before the target
    # prefix, so the 64-character limit only trips with a long target name.
    @pytest.mark.parametrize(
        ("target", "spec_path", "code"),
        [
            ("swagger", "/swagger2.json", "spec_invalid"),
            ("alias", "/alias-spec.yml", "spec_invalid"),
            ("big", "/big-spec.json", "too_many_tools"),
            ("longnametarget", "/long-name-spec.json", "tool_name_invalid"),
        ],
    )
    def test_bad_specs_are_refused(self, client, server, target, spec_path, code):
        response = client.post(
            f"/api/openapi-servers/{SLUG}/targets",
            json={"name": target, "spec_url": f"{MOCK_URL}{spec_path}", "base_url": MOCK_URL},
        )

        assert (response.status_code, response.json()["error"]) == (422, code)

    def test_metadata_base_url_is_blocked(self, client, server):
        response = _add_target(client, "meta", base_url="http://169.254.169.254")

        assert response.json()["error"] == "url_blocked"

    def test_credential_never_returned(self, client, server):
        body = client.get(f"/api/openapi-servers/{SLUG}").text

        for secret in (HEADER_KEY, QUERY_KEY, BEARER_TOKEN, BASIC_PASSWORD):
            assert secret not in body


class TestToolCalls:
    @pytest.mark.parametrize(
        ("tool", "scheme"),
        [("hdr_secureHeader", "api_key_header"), ("qry_secureQuery", "api_key_query"), ("brr_secureBearer", "bearer"), ("bas_secureBasic", "basic")],
    )
    def test_each_auth_scheme_reaches_backend(self, client, server, tool, scheme):
        assert _tool_json(_call_2026(client, tool, {})) == {"ok": True, "scheme": scheme}

    def test_array_response_is_wrapped(self, client, server):
        result = _tool_json(_call_2026(client, "items_listItems", {}))

        assert result == {"result": [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}]}

    def test_gateway_headers_never_reach_backend(self, client, server):
        received = _tool_json(_call_2026(client, "items_echoHeaders", {}))["headers"]

        assert [h for h in GATEWAY_ONLY_HEADERS if h in received] == []
        assert "authorization" not in received

    def test_handshake_era_lists_tools_with_hints(self, client, server):
        headers = _session_2025(client)
        listed = _parse(client.post(f"/{SLUG}/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}, headers=headers))
        tools = {t["name"]: t for t in listed["result"]["tools"]}

        assert tools["items_getItem"]["annotations"]["readOnlyHint"] is True
        assert tools["items_deleteItem"]["annotations"]["destructiveHint"] is True
        assert tools["items_getItem"]["inputSchema"]["additionalProperties"] is False

    def test_registry_record_has_the_tools(self, client, server):
        info = client.get(f"/api/servers/{SLUG}").json()

        assert info.get("num_tools", 0) >= 8


UPLOAD_PADDING_BYTES: int = 1_500_000


def _uploaded_spec(
    padding: int = UPLOAD_PADDING_BYTES,
) -> str:
    """A one-operation spec, padded past nginx's 1m default to prove the larger limit holds."""
    spec = {
        "openapi": "3.0.3",
        "info": {"title": "Uploaded", "version": "1"},
        "x-padding": "p" * padding,
        "paths": {"/items": {"get": {"operationId": "listItems", "responses": {"200": {"description": "ok"}}}}},
    }
    return json.dumps(spec)


class TestSpecUpload:
    def test_uploaded_spec_over_one_mib_serves_tools(self, client, server):
        added = client.post(
            f"/api/openapi-servers/{SLUG}/targets",
            json={"name": "upl", "spec_content": _uploaded_spec(), "base_url": MOCK_URL},
        )
        assert added.status_code == 201, added.text[:500]
        target = client.get(f"/api/openapi-servers/{SLUG}/targets/upl").json()

        assert (target["spec_source"], target["spec_url"]) == ("upload", None)
        assert _tool_json(_call_2026(client, "upl_listItems", {}))["result"][0]["id"] == 1

    def test_refresh_of_uploaded_spec_is_conflict(self, client, server):
        response = client.patch(f"/api/openapi-servers/{SLUG}/targets/upl", json={"refresh_spec": True})

        assert (response.status_code, response.json()["error"]) == (409, "spec_url_missing")

    def test_upload_over_gateway_limit_is_413(self, client, server):
        response = client.post(
            f"/api/openapi-servers/{SLUG}/targets",
            json={"name": "huge", "spec_content": _uploaded_spec(9 * 1024 * 1024), "base_url": MOCK_URL},
        )

        assert response.status_code == 413

    def test_other_api_routes_keep_the_one_mib_limit(self, client, server):
        response = client.post("/api/servers/register", content=b"x" * 2 * 1024 * 1024)

        assert response.status_code == 413


class TestRecovery:
    def test_tools_survive_container_restart(self, client, server):
        subprocess.run(["docker", "restart", CONTAINER_NAME], check=True, capture_output=True, timeout=120)  # nosec B603 B607 - local test container
        deadline = time.time() + RESTART_WAIT_SECONDS
        while time.time() < deadline:
            response = client.post(
                f"/{SLUG}/mcp",
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "items_getItem", "arguments": {"item_id": 7}, "_meta": META_2026}},
                headers={**MCP_HEADERS, "MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "tools/call", "Mcp-Name": "items_getItem"},
            )
            if response.status_code == 200:
                assert _tool_json(_parse(response)) == {"id": 7, "name": "item-7"}
                return
            time.sleep(3)
        pytest.fail(f"tools did not come back within {RESTART_WAIT_SECONDS}s of a container restart")
