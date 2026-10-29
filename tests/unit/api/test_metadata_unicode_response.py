"""Request validation must not turn malformed metadata Unicode into HTTP 500."""

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel, field_validator

from registry.api.ard_routes import ard_validation_exception_handler
from registry.utils.metadata_limits import validate_metadata_limits


class _MetadataRequest(BaseModel):
    metadata: dict[str, Any]

    @field_validator("metadata")
    @classmethod
    def _check_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        validate_metadata_limits(value)
        return value


@pytest.fixture
def app():
    app = FastAPI()
    app.add_exception_handler(RequestValidationError, ard_validation_exception_handler)

    @app.post("/api/metadata-validation")
    async def _validate_metadata(request: _MetadataRequest):
        return {"ok": True}

    return app


@pytest.mark.parametrize(
    "raw",
    [
        r'{"metadata":{"key":"\ud800"}}',
        r'{"metadata":{"\udfff":"value"}}',
    ],
    ids=["surrogate-value", "surrogate-key"],
)
@pytest.mark.asyncio
async def test_unpaired_surrogate_is_a_renderable_validation_error(app, raw):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/metadata-validation",
            content=raw,
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "not valid Unicode" in detail[0]["msg"]
    assert "input" not in detail[0]


@pytest.mark.asyncio
async def test_ordinary_validation_error_preserves_default_response(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/metadata-validation", json={"unexpected": 1})

    assert response.status_code == 422
    assert "input" in response.json()["detail"][0]
