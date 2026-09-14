"""Regression tests for issue #1740: the pinned-IP rewrite vs the redirect check.

The guarded transport pins every request to a validated IP by rewriting the URL
host (``url_guard._rewrite_to_pinned_ip``), keeping identity in the ``Host``
header and TLS SNI. That makes ``response.url`` differ from the requested URL on
**every** fetch, and read as an IP literal.

``skill_service`` used to treat that difference as evidence of a redirect, so a
SKILL.md on a forge whose hostname is allowlisted via ``github_extra_hosts`` but
resolves to a private address was rejected with "Redirect to unsafe URL blocked"
even though the origin returned 200 and no ``Location`` header existed. The check
shipped in 1.23.0 and stayed harmless until the pinning rewrite landed in 1.29.0.

These tests use a real local origin so the transport genuinely pins and rewrites.
A mocked client would not reproduce the bug, because the rewrite happens inside
the transport.
"""

import asyncio
import http.server
import socket
import socketserver
import threading
from collections.abc import Iterator
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

SKILL_MD = b"---\nname: pinned-repro\ndescription: regression fixture\n---\n\n# Body\n"
METADATA_URL = "http://169.254.169.254/latest/meta-data/"


class _Origin(http.server.BaseHTTPRequestHandler):
    """Serves SKILL.md at /, and two redirect shapes for the negative cases."""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/to-metadata":
            self.send_response(302)
            self.send_header("Location", METADATA_URL)
            self.end_headers()
        elif self.path == "/to-other-private":
            # 127.0.0.2 is private and NOT on the allowlist, unlike this origin's
            # own hostname, so a redirect there must still be refused.
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.2:9/SKILL.md")
            self.end_headers()
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(SKILL_MD)))
            self.end_headers()
            self.wfile.write(SKILL_MD)

    def do_HEAD(self) -> None:  # noqa: N802 - used by the health check path
        self.send_response(200)
        self.send_header("Content-Length", str(len(SKILL_MD)))
        self.end_headers()

    def log_message(self, *args: object) -> None:
        """Silence the default stderr access log."""


@pytest.fixture(scope="module")
def origin() -> Iterator[int]:
    """A loopback origin bound where the guard will actually dial.

    ``url_guard`` resolves the hostname itself and pins the first address
    ``getaddrinfo`` returns, so the fixture must bind that same address. Hosts
    resolving ``localhost`` to ``::1`` first (GitHub runners) would otherwise get
    a connection failure that looks like an SSRF refusal. Loopback is a private
    address, which is what makes it a valid stand-in for an internal forge.
    """
    family, _, _, _, sockaddr = socket.getaddrinfo("localhost", 0, proto=socket.IPPROTO_TCP)[0]

    class _Server(socketserver.TCPServer):
        address_family = family
        allow_reuse_address = True

    server = _Server((sockaddr[0], 0), _Origin)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # Fail loudly here rather than let an unreachable origin surface later as
        # a "redirect blocked" error, which is the very confusion under test.
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.settimeout(5)
            probe.connect((sockaddr[0], port))
        yield port
    finally:
        server.shutdown()
        server.server_close()


class _Settings:
    """Settings stub: an internal forge host on the operator bypass allowlist."""

    github_extra_hosts = "localhost"
    ssrf_allowed_hosts = ""
    ssrf_allowed_cidrs = ""
    gateway_proxy_allow_private_targets = False

    def __getattr__(self, name: str) -> None:
        return None


def _allowlisted_settings():
    """Patch the guard's settings and clear the cached skill allowlist."""
    import registry.utils.url_guard as url_guard

    url_guard._skill_allowlist.cache_clear()
    return patch.object(url_guard, "_get_settings", return_value=_Settings())


def _clear_allowlist_cache() -> None:
    import registry.utils.url_guard as url_guard

    url_guard._skill_allowlist.cache_clear()


class TestPinnedIpIsNotARedirect:
    """A 200 with no Location must never be reported as an unsafe redirect."""

    def test_validate_accepts_allowlisted_private_host(self, origin: int) -> None:
        """The 1.29.0 regression: registration rejected a fetch that succeeded.

        The origin answers 200 and sends no Location, so any "redirect blocked"
        error here comes from our own pinning rewrite rather than the server.
        """
        from registry.services.skill_service import _validate_skill_md_url

        with _allowlisted_settings():
            result = asyncio.run(_validate_skill_md_url(f"http://localhost:{origin}/SKILL.md"))
        _clear_allowlist_cache()

        assert result["valid"] is True
        assert result["content_version"]

    def test_parse_accepts_allowlisted_private_host(self, origin: int) -> None:
        """Same fetch through the parse path, which is what the UI dialog calls."""
        from registry.services.skill_service import _parse_skill_md_content

        with _allowlisted_settings():
            result = asyncio.run(_parse_skill_md_content(f"http://localhost:{origin}/SKILL.md"))
        _clear_allowlist_cache()

        assert result["name"] == "pinned-repro"
        assert result["description"] == "regression fixture"

    def test_health_check_reports_healthy(self, origin: int) -> None:
        """Existing skills on an internal forge must not flip to unhealthy."""
        from registry.services.skill_service import _check_skill_health

        with _allowlisted_settings():
            result = asyncio.run(_check_skill_health(f"http://localhost:{origin}/SKILL.md"))
        _clear_allowlist_cache()

        assert result["healthy"] is True, result
        assert result["error"] is None

    def test_content_fetch_succeeds_for_allowlisted_private_host(self, origin: int) -> None:
        """The fourth call site, behind ``GET /api/skills/{path}/content``.

        Serving stored SKILL.md content used the same ``response.url`` inference,
        so the content endpoint returned an SSRF error for an internal forge.
        """
        from registry.schemas.skill_models import SkillCard
        from registry.services.skill_service import _fetch_authenticated_content

        url = f"http://localhost:{origin}/SKILL.md"
        skill = SkillCard(
            path="/skills/internal-forge-probe",
            name="internal-forge-probe",
            description="fixture",
            skill_md_url=url,
            auth_scheme="none",
        )

        with _allowlisted_settings():
            response = asyncio.run(_fetch_authenticated_content(url, skill))
        _clear_allowlist_cache()

        assert response.status_code == 200
        assert b"pinned-repro" in response.content


class TestRealRedirectsStillRefused:
    """The CWE-918 protection the check was added for must survive the fix."""

    def test_redirect_to_metadata_ip_is_refused(self, origin: int) -> None:
        from registry.services.skill_service import (
            SkillUrlValidationError,
            _validate_skill_md_url,
        )

        with _allowlisted_settings(), pytest.raises(SkillUrlValidationError) as exc:
            asyncio.run(_validate_skill_md_url(f"http://localhost:{origin}/to-metadata"))
        _clear_allowlist_cache()

        # Assert the reason, not just the type. A transport error raises the same
        # class, which would let an unreachable origin pose as a security refusal.
        # These hops are refused by the guarded transport's own per-hop validation
        # before any connect, so the message names the failure, not the target.
        message = str(exc.value)
        assert "All connection attempts failed" not in message
        assert "SSRF validation" in message or "Redirect to unsafe URL blocked" in message

    def test_redirect_to_non_allowlisted_private_ip_is_refused(self, origin: int) -> None:
        """Allowlisting one internal host must not admit every internal host.

        The origin's own hostname is on the allowlist; the redirect target is a
        different private address that is not.
        """
        from registry.services.skill_service import (
            SkillUrlValidationError,
            _validate_skill_md_url,
        )

        with _allowlisted_settings(), pytest.raises(SkillUrlValidationError) as exc:
            asyncio.run(_validate_skill_md_url(f"http://localhost:{origin}/to-other-private"))
        _clear_allowlist_cache()

        message = str(exc.value)
        assert "All connection attempts failed" not in message
        assert "SSRF validation" in message or "Redirect to unsafe URL blocked" in message


class TestRedirectDetectionHelper:
    """The helper reads redirect history, never the transport-rewritten URL."""

    def test_no_history_means_no_redirect(self) -> None:
        """Guards the exact inference that broke: url difference is not a redirect."""
        import httpx

        from registry.services.skill_service import _unsafe_redirect_target

        request = httpx.Request("GET", "http://10.0.0.5/SKILL.md")  # pinned-IP form
        response = httpx.Response(200, request=request)

        assert _unsafe_redirect_target(response) is None

    def test_hop_is_validated_under_its_hostname_not_the_pinned_ip(self) -> None:
        """A hop rewritten to a private IP is judged by the name it was asked for."""
        import httpx

        from registry.services.skill_service import _unsafe_redirect_target

        # What the transport produces: URL holds the pinned IP, identity is kept
        # in the Host header and the sni_hostname extension.
        pinned = httpx.Request(
            "GET",
            "http://127.0.0.1:8080/SKILL.md",
            headers={"Host": "localhost:8080"},
            extensions={"sni_hostname": "localhost"},
        )
        first = httpx.Response(302, request=httpx.Request("GET", "http://localhost:8080/x"))
        final = httpx.Response(200, request=pinned, history=[first])

        with _allowlisted_settings():
            assert _unsafe_redirect_target(final) is None
        _clear_allowlist_cache()
