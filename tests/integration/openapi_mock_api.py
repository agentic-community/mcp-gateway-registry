"""Mock REST API for OpenAPI MCP server tests (issue #2022).

Serves its own OpenAPI 3.1 spec plus specs the registry must refuse, and checks
that each auth scheme delivers its credential. /echo-headers returns the
headers it received, so a test can prove gateway headers never reach a
third-party API.

Run (inside the compose network, so the registry and openapi-mcp can reach it):
    uv run uvicorn tests.integration.openapi_mock_api:app --host 0.0.0.0 --port 8000
"""

import base64
import json
import logging
from pathlib import Path

from fastapi import (
    FastAPI,
    Request,
)
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
)

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


# Test-only credentials the mock accepts. Not secrets.
HEADER_KEY: str = "mock-header-key"  # pragma: allowlist secret
QUERY_KEY: str = "mock-query-key"  # pragma: allowlist secret
BEARER_TOKEN: str = "mock-bearer-token"  # pragma: allowlist secret
BASIC_USER: str = "mock-user"
BASIC_PASSWORD: str = "mock-password"  # pragma: allowlist secret
TOO_MANY_OPERATIONS: int = 150
HUGE_SPEC_OPERATIONS: int = 2500
HUGE_SPEC_DESCRIPTION: str = "Padding so the spec approaches the 5 MiB cap. " * 35
LONG_OPERATION_ID: str = "get" + "VeryLongOperationName" * 4

ITEMS: list[dict] = [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}]


def _get(
    operation_id: str,
    tag: str,
    params: list[dict] | None = None,
    summary: str = "",
) -> dict:
    return {
        "get": {
            "operationId": operation_id,
            "tags": [tag],
            "summary": summary or operation_id,
            "parameters": params or [],
            "responses": {"200": {"description": "ok"}},
        }
    }


def _spec() -> dict:
    item_id = [{"name": "item_id", "in": "path", "required": True, "schema": {"type": "integer"}}]
    paths = {
        "/items": _get("listItems", "items", summary="List items"),
        "/items/{item_id}": {
            **_get("getItem", "items", item_id, "Get one item"),
            "delete": {
                "operationId": "deleteItem",
                "tags": ["admin"],
                "parameters": item_id,
                "responses": {"204": {"description": "deleted"}},
            },
        },
        "/secure/header": _get("secureHeader", "secure"),
        "/secure/query": _get("secureQuery", "secure"),
        "/secure/bearer": _get("secureBearer", "secure"),
        "/secure/basic": _get("secureBasic", "secure"),
        "/echo-headers": _get("echoHeaders", "debug"),
    }
    return {
        "openapi": "3.1.0",
        "info": {"title": "Mock REST API", "version": "1.0.0"},
        "paths": paths,
    }


# A hand-written OpenAPI 3.0 spec for three NASA endpoints (api.nasa.gov does not
# publish one). The one copy lives in docs/examples/openapi/nasa.json, usable as an
# upload or, through /nasa-spec.json here, as a spec URL. It declares no api_key
# parameter: per-user targets send each user's key in the X-Api-Key header, which
# api.nasa.gov accepts.
NASA_SPEC_PATH: Path = Path(__file__).resolve().parents[2] / "docs" / "examples" / "openapi" / "nasa.json"
NASA_SPEC: dict = json.loads(NASA_SPEC_PATH.read_text())


app = FastAPI(title="openapi-mcp mock REST API", openapi_url=None, docs_url=None, redoc_url=None)


@app.get("/spec.json")
async def spec() -> dict:
    return _spec()


@app.get("/nasa-spec.json")
async def nasa_spec() -> dict:
    return NASA_SPEC


@app.get("/swagger2.json")
async def swagger2() -> dict:
    return {"swagger": "2.0", "info": {"title": "old", "version": "1"}, "paths": {"/a": {}}}


@app.get("/alias-spec.yml")
async def alias_spec() -> PlainTextResponse:
    body = "openapi: 3.0.0\ninfo: {title: alias, version: '1'}\npaths: {/a: {}}\nx-a: &a [1, 2]\nx-b: *a\n"
    return PlainTextResponse(body, media_type="application/yaml")


@app.get("/big-spec.json")
async def big_spec() -> dict:
    paths = {
        f"/op{i}": _get(f"op{i}", "bulk" if i % 2 else "other") for i in range(TOO_MANY_OPERATIONS)
    }
    return {"openapi": "3.1.0", "info": {"title": "big", "version": "1"}, "paths": paths}


@app.get("/huge-spec.json")
async def huge_spec() -> dict:
    """About 4.5 MB: for memory tests near OPENAPI_MCP_SPEC_MAX_BYTES (5 MiB)."""
    paths = {}
    for i in range(HUGE_SPEC_OPERATIONS):
        operation = _get(f"huge{i}", "huge")
        operation["get"]["description"] = HUGE_SPEC_DESCRIPTION
        paths[f"/huge{i}"] = operation
    return {"openapi": "3.1.0", "info": {"title": "huge", "version": "1"}, "paths": paths}


@app.get("/long-name-spec.json")
async def long_name_spec() -> dict:
    return {
        "openapi": "3.1.0",
        "info": {"title": "long", "version": "1"},
        "paths": {"/long": _get(LONG_OPERATION_ID, "x")},
    }


@app.get("/items")
async def list_items() -> list[dict]:
    return ITEMS


@app.get("/items/{item_id}")
async def get_item(
    item_id: int,
) -> dict:
    return {"id": item_id, "name": f"item-{item_id}"}


@app.delete("/items/{item_id}", status_code=204)
async def delete_item(
    item_id: int,
) -> None:
    return None


def _denied() -> JSONResponse:
    return JSONResponse({"ok": False}, status_code=401)


@app.get("/secure/header")
async def secure_header(
    request: Request,
):
    return (
        {"ok": True, "scheme": "api_key_header"}
        if request.headers.get("x-api-key") == HEADER_KEY
        else _denied()
    )


@app.get("/secure/query")
async def secure_query(
    request: Request,
):
    return (
        {"ok": True, "scheme": "api_key_query"}
        if request.query_params.get("apikey") == QUERY_KEY
        else _denied()
    )


@app.get("/secure/bearer")
async def secure_bearer(
    request: Request,
):
    expected = f"Bearer {BEARER_TOKEN}"
    return (
        {"ok": True, "scheme": "bearer"}
        if request.headers.get("authorization") == expected
        else _denied()
    )


@app.get("/secure/basic")
async def secure_basic(
    request: Request,
):
    expected = "Basic " + base64.b64encode(f"{BASIC_USER}:{BASIC_PASSWORD}".encode()).decode()
    return (
        {"ok": True, "scheme": "basic"}
        if request.headers.get("authorization") == expected
        else _denied()
    )


@app.get("/echo-headers")
async def echo_headers(
    request: Request,
) -> dict:
    headers = {k.lower(): v for k, v in request.headers.items()}
    logger.info(f"echo-headers received:\n{json.dumps(sorted(headers), indent=2)}")
    return {"headers": headers}
