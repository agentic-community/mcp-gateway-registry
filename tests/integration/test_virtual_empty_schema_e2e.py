"""End-to-end check that empty inputSchema arrays survive as ``[]`` (issue #1532).

This is a live-stack test, not an in-process one. The empty-array serialization
bug lived in the nginx + lua virtual router, which the FastAPI-app integration
tests cannot exercise, and it only reproduced on the shipped runtime (Debian
nginx + Debian lua-cjson, where ``cjson.empty_array_mt`` is nil). So this test
drives a real gateway over HTTP.

It skips unless a gateway URL and bearer token are provided, so it is a no-op in
the standard unit/integration CI. Point it at a running stack to exercise it:

    export MCP_GATEWAY_E2E_URL=http://localhost
    export MCP_GATEWAY_E2E_TOKEN="$(python3 -c \
      "import json;print(json.load(open('.token'))['tokens']['access_token'])")"
    uv run pytest tests/integration/test_virtual_empty_schema_e2e.py -v

The durable, always-on guardrail for the same property is the build-time
assertion in ``docker/Dockerfile.registry`` (the image build fails if the shipped
cjson lacks ``empty_array_mt``); this test adds an HTTP-level confirmation when a
stack is available.
"""

import json
import os
import time
import uuid

import httpx
import pytest

# Enabling a virtual server makes the registry regenerate the nginx config, write
# the mapping file, and reload nginx asynchronously; the MCP endpoint is not
# routable until that finishes. Poll for readiness rather than racing the reload.
_READY_TIMEOUT_S = 30.0
_READY_POLL_S = 1.0

_GATEWAY_URL = os.environ.get("MCP_GATEWAY_E2E_URL")
_TOKEN = os.environ.get("MCP_GATEWAY_E2E_TOKEN")

pytestmark = pytest.mark.skipif(
    not (_GATEWAY_URL and _TOKEN),
    reason="Set MCP_GATEWAY_E2E_URL and MCP_GATEWAY_E2E_TOKEN to run the live E2E.",
)

_MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_TOKEN}"}


def _parse_mcp_body(text: str) -> dict:
    """Return the JSON object from a plain or SSE-framed MCP response body."""
    data_lines = [
        line[len("data:") :].strip() for line in text.splitlines() if line.startswith("data:")
    ]
    return json.loads(data_lines[-1] if data_lines else text)


_INIT_BODY = json.dumps(
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "e2e-1532", "version": "1.0"},
        },
    }
)


def _initialize_with_retry(client: httpx.Client, mcp_url: str) -> str:
    """Wait for the enabled virtual server to be routable and return a session id.

    Retries ``initialize`` until it yields an ``Mcp-Session-Id`` (or the timeout
    elapses), so the test does not race the post-enable nginx reload / mapping
    write. Raises with the last response on timeout.
    """
    deadline = time.monotonic() + _READY_TIMEOUT_S
    last = "no attempt made"
    while time.monotonic() < deadline:
        resp = client.post(mcp_url, headers={**_auth_headers(), **_MCP_HEADERS}, content=_INIT_BODY)
        session_id = resp.headers.get("Mcp-Session-Id")
        if session_id:
            return session_id
        last = f"HTTP {resp.status_code}: {resp.text[:200]}"
        time.sleep(_READY_POLL_S)
    raise AssertionError(f"virtual server not ready within {_READY_TIMEOUT_S}s (last: {last})")


def _tools_list_with_retry(
    client: httpx.Client, mcp_url: str, session_id: str
) -> tuple[str, list[dict]]:
    """Retry ``tools/list`` until the backend enrichment yields tools.

    A just-enabled virtual server may return an empty aggregation on the first
    call while backend discovery completes; poll until tools are present.
    """
    deadline = time.monotonic() + _READY_TIMEOUT_S
    body_text = ""
    while time.monotonic() < deadline:
        resp = client.post(
            mcp_url,
            headers={**_auth_headers(), **_MCP_HEADERS, "Mcp-Session-Id": session_id},
            content=json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
        )
        body_text = resp.text
        tools = _parse_mcp_body(body_text).get("result", {}).get("tools", [])
        if tools:
            return body_text, tools
        time.sleep(_READY_POLL_S)
    raise AssertionError(
        f"no tools from virtual server within {_READY_TIMEOUT_S}s (last: {body_text[:200]})"
    )


def _find_backend_with_empty_required(client: httpx.Client) -> tuple[str, str] | None:
    """Find a (server_path, tool_name) whose tool advertises ``required: []``."""
    resp = client.get("/api/servers", params={"limit": 200}, headers=_auth_headers())
    resp.raise_for_status()
    for server in resp.json().get("servers", []):
        for tool in server.get("tool_list", []):
            schema = tool.get("schema") or tool.get("inputSchema") or {}
            if isinstance(schema.get("required"), list) and not schema["required"]:
                return server["path"], tool["name"]
    return None


def test_empty_required_serializes_as_array_through_virtual_server() -> None:
    base = _GATEWAY_URL.rstrip("/")
    vs_path = f"/virtual/e2e-empty-required-{uuid.uuid4().hex[:8]}"

    with httpx.Client(base_url=base, timeout=30.0) as client:
        backend = _find_backend_with_empty_required(client)
        if backend is None:
            pytest.skip("No registered backend advertises a tool with required: [].")
        backend_path, tool_name = backend

        create_body = {
            "path": vs_path,
            "server_name": "E2E empty-required (issue 1532)",
            "description": "Transient VS asserting empty required stays [].",
            "tool_mappings": [{"tool_name": tool_name, "backend_server_path": backend_path}],
            "required_scopes": [],
            "tool_scope_overrides": [],
            "tags": ["e2e", "issue-1532"],
            "supported_transports": ["streamable-http"],
            "is_enabled": True,
        }

        try:
            created = client.post("/api/virtual-servers", json=create_body, headers=_auth_headers())
            assert created.status_code == 201, created.text

            toggled = client.post(
                f"/api/virtual-servers{vs_path}/toggle",
                json={"enabled": True},
                headers=_auth_headers(),
            )
            assert toggled.status_code == 200, toggled.text

            mcp_url = f"{base}{vs_path}/mcp"
            # Wait for the enable -> nginx reload to settle before driving MCP,
            # then wait for backend enrichment to produce tools.
            session_id = _initialize_with_retry(client, mcp_url)
            body_text, tools = _tools_list_with_retry(client, mcp_url, session_id)

            target = next((t for t in tools if t.get("name") == tool_name), tools[0])
            required = target.get("inputSchema", {}).get("required")

            # The fix: empty required must be a JSON array, not an object.
            assert required == [], f"expected required == [], got {required!r} in {body_text}"
            assert '"required":[]' in body_text.replace(" ", "")
            assert '"required":{}' not in body_text.replace(" ", "")
        finally:
            client.delete(f"/api/virtual-servers{vs_path}", headers=_auth_headers())
