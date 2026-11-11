"""The consent cookie is Secure on HTTPS when ``SESSION_COOKIE_SECURE`` is true.

``set_consent_cookie`` and ``clear_consent_cookie`` share one rule: Secure on
HTTPS when ``SESSION_COOKIE_SECURE`` is true, never on plain HTTP, where a
browser drops a Secure cookie. A set and a clear that disagree leave the rule
in two places.
"""

from __future__ import annotations

from http.cookies import SimpleCookie

import pytest
from starlette.requests import Request
from starlette.responses import Response

from registry.core.config import settings
from registry.egress_auth.browser_binding import (
    clear_consent_cookie,
    consent_cookie_name,
    new_browser_binding,
    set_consent_cookie,
)

pytestmark = [pytest.mark.unit, pytest.mark.auth]


def _request(
    https: bool,
) -> Request:
    """Return a browser request that reached the gateway over HTTPS or plain HTTP."""
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/oauth2/egress/callback",
            "scheme": "https" if https else "http",
            "server": ("gw.example.com", 443 if https else 80),
            "headers": [],
        }
    )


def _secure_flags(
    response: Response,
    name: str,
) -> list[bool]:
    """Return the Secure flag of every ``Set-Cookie`` the response sends for ``name``."""
    flags = []
    for header in response.headers.getlist("set-cookie"):
        parsed = SimpleCookie()
        parsed.load(header)
        flags.extend(bool(morsel["secure"]) for key, morsel in parsed.items() if key == name)
    return flags


@pytest.mark.parametrize("https", [True, False], ids=["https", "http"])
@pytest.mark.parametrize("setting_secure", [True, False], ids=["secure-on", "secure-off"])
def test_consent_cookie_set_and_clear_are_secure_only_on_https_with_the_setting_on(
    monkeypatch, setting_secure, https
):
    """The consent cookie's set and clear carry Secure exactly on HTTPS with the setting on."""
    monkeypatch.setattr(settings, "session_cookie_secure", setting_secure)
    monkeypatch.setattr(settings, "session_cookie_domain", None)
    request = _request(https)
    binding = new_browser_binding()
    consent_name = consent_cookie_name(binding.digest)

    set_response = Response()
    set_consent_cookie(set_response, request, binding)
    clear_response = Response()
    clear_consent_cookie(clear_response, request, binding.digest)

    expected = setting_secure and https
    assert _secure_flags(set_response, consent_name) == [expected]
    assert _secure_flags(clear_response, consent_name) == [expected]
