"""The session cookie isolation middleware runs before every other registry middleware.

``SessionCookieIsolationMiddleware`` caches the session cookie read and removes
the session cookie from the ``Cookie`` fields it passes on. A registry
middleware that ran before it, such as the audit middleware, would see the raw
header and could pick one of two copies the shared reader refuses. Starlette
runs ``app.user_middleware`` in list order, first entry outermost, so the
isolation middleware must be the first entry. OpenTelemetry's ASGI middleware
wraps the whole stack outside this list and is out of scope here.

Each case imports ``registry.main`` in a fresh interpreter, because the
middleware list is fixed when the module is imported.
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - runs the registry import in a fresh interpreter
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.auth]

_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
_MARKER: str = "USER_MIDDLEWARE="
_LIST_MIDDLEWARE: str = (
    "import json, registry.main as main; "
    f"print({_MARKER!r} + json.dumps([m.cls.__name__ for m in main.app.user_middleware]))"
)


def _user_middleware(
    audit_enabled: bool,
) -> list[str]:
    """Return the class names of ``registry.main.app.user_middleware``, outermost first."""
    env = {
        **os.environ,
        "PYTHONPATH": str(_REPO_ROOT),
        "AUDIT_LOG_ENABLED": "true" if audit_enabled else "false",
    }
    result = subprocess.run(  # nosec B603 - fixed interpreter and fixed import statement
        [sys.executable, "-c", _LIST_MIDDLEWARE],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    lines = [line for line in result.stdout.splitlines() if line.startswith(_MARKER)]
    assert lines, result.stdout[-2000:]
    return json.loads(lines[-1].removeprefix(_MARKER))


@pytest.mark.parametrize("audit_enabled", [True, False], ids=["audit-on", "audit-off"])
def test_isolation_middleware_is_the_first_user_middleware(audit_enabled):
    """The isolation middleware is outermost of the registry's own middleware, audit on or off."""
    middleware = _user_middleware(audit_enabled)

    assert ("AuditMiddleware" in middleware) is audit_enabled, middleware
    assert middleware[0] == "SessionCookieIsolationMiddleware", middleware
    assert middleware.count("SessionCookieIsolationMiddleware") == 1, middleware
