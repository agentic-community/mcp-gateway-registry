"""Route-level tests for the ARD Registry adapter (issue #1295, Phase 2).

Uses a minimal FastAPI app with just the ARD router + exception handlers and a
dependency-overridden auth, so we exercise the HTTP contract (ARD error
envelope, 401 reshape, 404 toggle, pagination, federation) without standing up
the whole registry app. The service layer is mocked; service logic is covered
by tests/unit/services/test_ard_search_service.py.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from registry.api import ard_routes
from registry.api.ard_routes import (
    ard_http_exception_handler,
    ard_validation_exception_handler,
)
from registry.auth.dependencies import nginx_proxied_auth
from registry.schemas.ard_models import ArdCatalogEntry, ArdSearchResult


def _make_app(user_context: dict | None = None, auth_raises: bool = False) -> FastAPI:
    app = FastAPI()
    app.include_router(ard_routes.router, prefix="/api/ard")
    app.add_exception_handler(StarletteHTTPException, ard_http_exception_handler)
    app.add_exception_handler(RequestValidationError, ard_validation_exception_handler)

    def _override():
        if auth_raises:
            raise HTTPException(status_code=401, detail="auth required")
        return user_context or {"username": "admin", "is_admin": True}

    app.dependency_overrides[nginx_proxied_auth] = _override
    return app


def _result(ident: str, score: int) -> ArdSearchResult:
    return ArdSearchResult(
        identifier=ident, display_name=ident.split(":")[-1],
        type="application/mcp-server-card+json", url="http://x", score=score, source="http://s",
    )


@pytest.fixture(autouse=True)
def _enable_registry():
    with patch.object(ard_routes.settings, "ard_registry_enabled", True):
        yield


class TestSearch:
    def test_happy_path(self):
        app = _make_app()
        with patch.object(
            ard_routes.ard_search_service, "search_and_scope",
            AsyncMock(return_value=([_result("urn:air:x:server:a", 90)], 2)),
        ):
            r = TestClient(app).post("/api/ard/search", json={"query": {"text": "hi"}})
        assert r.status_code == 200
        body = r.json()
        assert body["results"][0]["score"] == 90
        assert body["results"][0]["source"] == "http://s"
        assert body["referrals"] == []
        assert "pageToken" in body

    def test_pagination_next_token(self):
        app = _make_app()
        results = [_result(f"urn:air:x:server:{i}", 100 - i) for i in range(5)]
        with patch.object(
            ard_routes.ard_search_service, "search_and_scope",
            AsyncMock(return_value=(results, 0)),
        ):
            r = TestClient(app).post("/api/ard/search", json={"query": {"text": "q"}, "pageSize": 2})
        body = r.json()
        assert len(body["results"]) == 2
        assert body["pageToken"] is not None  # more pages remain

    @pytest.mark.parametrize("federation", ["none", "auto", "referrals"])
    def test_federation_modes_accepted(self, federation):
        app = _make_app()
        with patch.object(
            ard_routes.ard_search_service, "search_and_scope",
            AsyncMock(return_value=([], 0)),
        ):
            r = TestClient(app).post(
                "/api/ard/search", json={"query": {"text": "q"}, "federation": federation}
            )
        assert r.status_code == 200

    def test_extra_property_is_ard_400(self):
        app = _make_app()
        r = TestClient(app).post("/api/ard/search", json={"query": {"text": "q"}, "surprise": 1})
        assert r.status_code == 400
        assert r.json() == {"errorCode": "INVALID_REQUEST", "message": "The request was invalid."}

    def test_bad_page_token_is_ard_400(self):
        app = _make_app()
        r = TestClient(app).post(
            "/api/ard/search", json={"query": {"text": "q"}, "pageToken": "!!bad!!"}
        )
        assert r.status_code == 400
        assert r.json()["errorCode"] == "INVALID_REQUEST"

    def test_unknown_filter_key_is_ard_400(self):
        app = _make_app()
        r = TestClient(app).post(
            "/api/ard/search", json={"query": {"text": "q", "filter": {"bogus": "y"}}}
        )
        assert r.status_code == 400
        assert r.json()["errorCode"] == "INVALID_REQUEST"

    def test_unauthenticated_is_clean_ard_401(self):
        app = _make_app(auth_raises=True)
        r = TestClient(app).post("/api/ard/search", json={"query": {"text": "q"}})
        assert r.status_code == 401
        # Clean ARD envelope, not a redirect or {"detail": ...}
        assert r.json() == {"errorCode": "UNAUTHENTICATED", "message": "Authentication required."}

    def test_toggle_off_is_ard_404(self):
        app = _make_app()
        with patch.object(ard_routes.settings, "ard_registry_enabled", False):
            r = TestClient(app).post("/api/ard/search", json={"query": {"text": "q"}})
        assert r.status_code == 404
        assert r.json()["errorCode"] == "NOT_FOUND"


class TestBrowse:
    def test_happy_path(self):
        app = _make_app()
        items = [
            ArdCatalogEntry(
                identifier="urn:air:x:agent:a", display_name="A",
                type="application/a2a-agent-card+json", url="http://x",
            )
        ]
        with patch.object(
            ard_routes.ard_search_service, "browse", AsyncMock(return_value=(items, 1)),
        ):
            r = TestClient(app).get("/api/ard/agents?pageSize=5")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["items"][0]["identifier"] == "urn:air:x:agent:a"

    def test_toggle_off_is_ard_404(self):
        app = _make_app()
        with patch.object(ard_routes.settings, "ard_registry_enabled", False):
            r = TestClient(app).get("/api/ard/agents")
        assert r.status_code == 404
        assert r.json()["errorCode"] == "NOT_FOUND"
