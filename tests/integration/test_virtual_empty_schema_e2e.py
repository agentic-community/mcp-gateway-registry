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
import uuid

import httpx
import pytest

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
            init = client.post(
                mcp_url,
                headers={**_auth_headers(), **_MCP_HEADERS},
                content=json.dumps(
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
                ),
            )
            session_id = init.headers.get("Mcp-Session-Id")
            assert session_id, f"no Mcp-Session-Id from initialize: {init.text}"

            listed = client.post(
                mcp_url,
                headers={**_auth_headers(), **_MCP_HEADERS, "Mcp-Session-Id": session_id},
                content=json.dumps(
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
                ),
            )
            body_text = listed.text
            body = _parse_mcp_body(body_text)

            tools = body.get("result", {}).get("tools", [])
            assert tools, f"no tools returned: {body_text}"

            target = next((t for t in tools if t.get("name") == tool_name), tools[0])
            required = target.get("inputSchema", {}).get("required")

            # The fix: empty required must be a JSON array, not an object.
            assert required == [], f"expected required == [], got {required!r} in {body_text}"
            assert '"required":[]' in body_text.replace(" ", "")
            assert '"required":{}' not in body_text.replace(" ", "")
        finally:
            client.delete(f"/api/virtual-servers{vs_path}", headers=_auth_headers())
