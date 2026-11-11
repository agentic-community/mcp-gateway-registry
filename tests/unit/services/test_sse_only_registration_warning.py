"""Registration warns when a server offers only the deprecated HTTP+SSE transport."""

import logging

import pytest

from registry.services.server_service import (
    SSE_TRANSPORT_REMOVAL_EARLIEST,
    _warn_if_sse_only,
)


def test_sse_only_server_logs_a_deprecation_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="registry.services.server_service"):
        _warn_if_sse_only({"path": "/old", "supported_transports": ["sse"]})

    assert "deprecated HTTP+SSE" in caplog.text
    assert SSE_TRANSPORT_REMOVAL_EARLIEST in caplog.text


@pytest.mark.parametrize(
    "transports", [["streamable-http"], ["streamable-http", "sse"], [], None, ["stdio"]]
)
def test_other_transports_log_nothing(caplog, transports):
    with caplog.at_level(logging.WARNING, logger="registry.services.server_service"):
        _warn_if_sse_only({"path": "/s", "supported_transports": transports})

    assert caplog.text == ""
