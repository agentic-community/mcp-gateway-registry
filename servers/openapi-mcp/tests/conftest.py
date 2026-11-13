"""Shared fixtures for the openapi-mcp container tests."""

import os

os.environ.setdefault("OPENAPI_MCP_SYNC_SECRET", "test-sync-secret-0123456789abcdef-0123456789")

import copy  # noqa: E402
import json  # noqa: E402
from collections.abc import Callable  # noqa: E402

import guarded_transport  # noqa: E402
import httpx  # noqa: E402
import hubs  # noqa: E402
import pytest  # noqa: E402
from settings import GuardSettings  # noqa: E402

from registry.utils import url_guard  # noqa: E402

SYNC_SECRET: str = os.environ["OPENAPI_MCP_SYNC_SECRET"]

ITEMS_SPEC: dict = {
    "openapi": "3.1.0",
    "info": {"title": "Items API", "version": "1.0.0"},
    "paths": {
        "/items": {
            "get": {
                "operationId": "listItems",
                "tags": ["items"],
                "summary": "List items",
                "responses": {"200": {"description": "ok"}},
            },
            "post": {
                "operationId": "createItem",
                "tags": ["items"],
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {"type": "object", "properties": {"name": {"type": "string"}}}
                        }
                    }
                },
                "responses": {"201": {"description": "created"}},
            },
        },
        "/items/{item_id}": {
            "get": {
                "operationId": "getItem",
                "tags": ["items"],
                "parameters": [
                    {
                        "name": "item_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                    }
                ],
                "responses": {
                    "200": {
                        "description": "ok",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["id", "photoUrls"],
                                    "properties": {
                                        "id": {"type": "integer"},
                                        "photoUrls": {"type": "array", "items": {"type": "string"}},
                                    },
                                }
                            }
                        },
                    }
                },
            },
            "delete": {
                "operationId": "deleteItem",
                "tags": ["admin"],
                "parameters": [
                    {
                        "name": "item_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                    }
                ],
                "responses": {"204": {"description": "deleted"}},
            },
        },
        "/echo": {
            "get": {
                "summary": "Echo Things",
                "responses": {"200": {"description": "ok"}},
            }
        },
    },
}


@pytest.fixture
def items_spec() -> dict:
    return copy.deepcopy(ITEMS_SPEC)


@pytest.fixture(autouse=True)
def guard_settings():
    """Run url_guard on the container's guard-only settings."""
    previous = url_guard.settings
    url_guard.settings = GuardSettings()
    yield
    url_guard.settings = previous


@pytest.fixture(autouse=True)
async def reset_hubs():
    yield
    await hubs.close_all()


class RecordingBackend:
    """A mock REST backend that records each request it receives."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path.removeprefix("/v1")
        if path.startswith("/items/"):
            return httpx.Response(200, json={"id": 7})
        if path == "/items":
            return httpx.Response(200, json=[{"id": 1}, {"id": 2}])
        return httpx.Response(200, text="plain text", headers={"content-type": "application/json"})


@pytest.fixture
def backend(monkeypatch) -> RecordingBackend:
    """Replace the guard's inner transport with a recording mock backend."""
    recorder = RecordingBackend()

    def _factory(
        **kwargs,
    ) -> httpx.MockTransport:
        return httpx.MockTransport(recorder.handler)

    monkeypatch.setattr(guarded_transport.url_guard, "GuardedAsyncTransport", _factory)
    return recorder


def mcp_post_body(
    method: str,
    params: dict,
    request_id: int = 1,
) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})


def parse_mcp_response(
    response: httpx.Response,
) -> dict:
    """Read a JSON-RPC message from a JSON or SSE response."""
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        for line in response.text.splitlines():
            if line.startswith("data:"):
                return json.loads(line[len("data:") :])
        raise AssertionError(f"no data line in SSE response: {response.text[:200]}")
    return response.json()


ResponseParser = Callable[[httpx.Response], dict]
