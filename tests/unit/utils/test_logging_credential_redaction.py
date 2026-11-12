"""Tests that request URLs carrying credentials stay out of the logs (#2027).

The registry's logout calls the auth server with the user's id_token_hint in the
query string. httpx logged that URL at INFO in the registry, and uvicorn's access
log recorded it in the auth server. Both lines still log, without the query string.
"""

import asyncio
import logging
import socket
from unittest.mock import patch

import httpx
import pytest
import uvicorn

from registry.utils.logging_setup import (
    ACCESS_LOG_PATH_ARG_INDEX,
    ACCESS_LOGGER_NAME,
    HTTPX_LOGGER_NAME,
    _QueryStringFilter,
    _strip_query_strings_from_request_logs,
    setup_logging,
)

FAKE_ID_TOKEN: str = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ0ZXN0LXVzZXIifQ.c2lnbmF0dXJl"
LOGOUT_PATH: str = "/oauth2/logout/cognito"
SERVER_START_TIMEOUT_SECONDS: float = 5.0


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _ok_app(scope, receive, send) -> None:
    """A minimal ASGI app that answers every HTTP request with 200."""
    if scope["type"] != "http":
        return
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


async def _request_through_real_uvicorn(
    path_with_query: str,
) -> None:
    """Serve ``_ok_app`` with uvicorn on loopback, send one request, then stop."""
    port = _free_loopback_port()
    config = uvicorn.Config(_ok_app, host="127.0.0.1", port=port, log_config=None)
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())
    try:
        waited = 0.0
        while not server.started:
            await asyncio.sleep(0.05)
            waited += 0.05
            if waited > SERVER_START_TIMEOUT_SECONDS:
                raise RuntimeError("uvicorn did not start in time")
        async with httpx.AsyncClient() as client:
            await client.get(f"http://127.0.0.1:{port}{path_with_query}")
    finally:
        server.should_exit = True
        await serve_task


@pytest.fixture
def restore_logger_state():
    """Put the httpx and access-log levels and filters back after each test."""
    names = (HTTPX_LOGGER_NAME, ACCESS_LOGGER_NAME)
    saved = {
        name: (logging.getLogger(name).level, list(logging.getLogger(name).filters))
        for name in names
    }
    yield
    for name, (level, filters) in saved.items():
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.filters = filters


class TestHttpxRequestLine:
    """httpx still logs each request, without the query string."""

    @pytest.mark.asyncio
    async def test_logout_request_logs_url_without_id_token(self, restore_logger_state, caplog):
        _strip_query_strings_from_request_logs()
        transport = httpx.MockTransport(lambda request: httpx.Response(302))

        with caplog.at_level(logging.INFO, logger=HTTPX_LOGGER_NAME):
            async with httpx.AsyncClient(transport=transport) as client:
                await client.get(
                    f"http://auth-server:8888{LOGOUT_PATH}",
                    params={
                        "redirect_uri": "https://example.com/logout",
                        "id_token_hint": FAKE_ID_TOKEN,
                    },
                )

        assert f"HTTP Request: GET http://auth-server:8888{LOGOUT_PATH} " in caplog.text
        assert FAKE_ID_TOKEN not in caplog.text

    def test_httpx_stays_at_its_configured_level(self, restore_logger_state):
        logging.getLogger(HTTPX_LOGGER_NAME).setLevel(logging.INFO)

        _strip_query_strings_from_request_logs()

        assert logging.getLogger(HTTPX_LOGGER_NAME).level == logging.INFO


class TestAccessLogQueryFilter:
    """uvicorn access-log lines keep the path and drop the query string."""

    def test_removes_query_string_from_access_record(self):
        record = logging.LogRecord(
            name=ACCESS_LOGGER_NAME,
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg='%s - "%s %s HTTP/%s" %d',
            args=(
                "127.0.0.1:5000",
                "GET",
                f"{LOGOUT_PATH}?id_token_hint={FAKE_ID_TOKEN}",
                "1.1",
                302,
            ),
            exc_info=None,
        )

        _QueryStringFilter(ACCESS_LOG_PATH_ARG_INDEX).filter(record)

        assert record.getMessage() == f'127.0.0.1:5000 - "GET {LOGOUT_PATH} HTTP/1.1" 302'

    def test_leaves_records_of_another_shape_unchanged(self):
        record = logging.LogRecord(
            name=ACCESS_LOGGER_NAME,
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="plain message %s",
            args=("value",),
            exc_info=None,
        )

        assert _QueryStringFilter(ACCESS_LOG_PATH_ARG_INDEX).filter(record) is True
        assert record.getMessage() == "plain message value"

    def test_filter_is_added_only_once(self, restore_logger_state):
        _strip_query_strings_from_request_logs()
        _strip_query_strings_from_request_logs()

        for name in (HTTPX_LOGGER_NAME, ACCESS_LOGGER_NAME):
            filters = logging.getLogger(name).filters
            assert sum(isinstance(f, _QueryStringFilter) for f in filters) == 1

    @pytest.mark.asyncio
    async def test_real_uvicorn_access_log_omits_id_token(self, restore_logger_state, caplog):
        _strip_query_strings_from_request_logs()

        with caplog.at_level(logging.INFO, logger=ACCESS_LOGGER_NAME):
            await _request_through_real_uvicorn(f"{LOGOUT_PATH}?id_token_hint={FAKE_ID_TOKEN}")

        access_lines = [r.getMessage() for r in caplog.records if r.name == ACCESS_LOGGER_NAME]
        assert any(f"GET {LOGOUT_PATH} HTTP/1.1" in line for line in access_lines)
        assert FAKE_ID_TOKEN not in caplog.text


class TestSetupLoggingWiring:
    """setup_logging, which both services call at import, adds the filters."""

    def test_setup_logging_adds_both_filters(self, tmp_path, restore_logger_state):
        for name in (HTTPX_LOGGER_NAME, ACCESS_LOGGER_NAME):
            logging.getLogger(name).filters = []

        with patch("registry.core.config.settings") as mock_settings:
            mock_settings.app_log_level = "INFO"
            mock_settings.app_log_max_bytes = 50 * 1024 * 1024
            mock_settings.app_log_backup_count = 5
            mock_settings.app_log_centralized_enabled = False
            mock_settings.log_dir = tmp_path
            setup_logging(service_name="test-service", log_file=tmp_path / "test.log")

        for name in (HTTPX_LOGGER_NAME, ACCESS_LOGGER_NAME):
            filters = logging.getLogger(name).filters
            assert any(isinstance(f, _QueryStringFilter) for f in filters)
