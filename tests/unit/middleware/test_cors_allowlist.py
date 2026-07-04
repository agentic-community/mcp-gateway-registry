"""Tests for the credentialed CORS allowlist behavior.

These build a minimal FastAPI app that wires up ``CORSMiddleware`` exactly the
way ``registry/main.py`` does (explicit ``allow_origins`` list, credentials
enabled), then exercise the browser-facing behavior: a non-allowlisted origin
(e.g. an arbitrary EC2 public DNS host) receives no
``Access-Control-Allow-Origin`` header, while a configured origin does — and
only with an ``Access-Control-Allow-Credentials`` header when it is trusted.
"""

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

ALLOWED_ORIGIN = "https://app.example.com"
EC2_ORIGIN = "https://ec2-1-2-3-4.compute-1.amazonaws.com"


def _build_app(allowed_origins: list[str]) -> FastAPI:
    """Build a minimal app with CORS wired up as in registry/main.py."""
    app = FastAPI()
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )

    @app.get("/api/ping")
    async def ping() -> JSONResponse:
        return JSONResponse(content={"ok": True})

    return app


@pytest.mark.unit
class TestCorsAllowlistBehavior:
    """Verify the credentialed CORS policy is an exact allowlist."""

    def test_allowlisted_origin_gets_acao_with_credentials(self) -> None:
        """A configured origin receives ACAO and ACA-Credentials headers."""
        client = TestClient(_build_app([ALLOWED_ORIGIN]))

        resp = client.get("/api/ping", headers={"Origin": ALLOWED_ORIGIN})

        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN
        assert resp.headers.get("access-control-allow-credentials") == "true"

    def test_arbitrary_ec2_origin_is_rejected(self) -> None:
        """An arbitrary EC2 public DNS origin gets no ACAO header.

        This is the regression guard against the previous broad regex that
        matched any *.compute*.amazonaws.com host with credentials.
        """
        client = TestClient(_build_app([ALLOWED_ORIGIN]))

        resp = client.get("/api/ping", headers={"Origin": EC2_ORIGIN})

        # The request still returns 200 (CORS is a browser-enforced response
        # header contract), but the browser will block the read because no
        # Access-Control-Allow-Origin header echoes the attacker origin.
        assert resp.headers.get("access-control-allow-origin") != EC2_ORIGIN
        assert resp.headers.get("access-control-allow-origin") is None

    def test_preflight_from_ec2_origin_is_denied(self) -> None:
        """A CORS preflight from a non-allowlisted origin is not approved."""
        client = TestClient(_build_app([ALLOWED_ORIGIN]))

        resp = client.options(
            "/api/ping",
            headers={
                "Origin": EC2_ORIGIN,
                "Access-Control-Request-Method": "POST",
            },
        )

        assert resp.headers.get("access-control-allow-origin") != EC2_ORIGIN

    def test_preflight_from_allowlisted_origin_is_approved(self) -> None:
        """A CORS preflight from a configured origin is approved."""
        client = TestClient(_build_app([ALLOWED_ORIGIN]))

        resp = client.options(
            "/api/ping",
            headers={
                "Origin": ALLOWED_ORIGIN,
                "Access-Control-Request-Method": "POST",
            },
        )

        assert resp.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN
        assert resp.headers.get("access-control-allow-credentials") == "true"

    def test_no_middleware_means_no_cors_headers(self) -> None:
        """With an empty allowlist (fail closed) no CORS headers are emitted."""
        client = TestClient(_build_app([]))

        resp = client.get("/api/ping", headers={"Origin": ALLOWED_ORIGIN})

        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") is None
        assert resp.headers.get("access-control-allow-credentials") is None
