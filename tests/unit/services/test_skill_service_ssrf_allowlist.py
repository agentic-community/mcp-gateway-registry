"""
Unit tests for SKILL.md SSRF handling via the hardened URL guard.

skill_service._is_safe_url now delegates to registry.utils.url_guard. The only
hosts that bypass the private-IP block are those explicitly configured in
settings.github_extra_hosts (e.g. GitHub Enterprise Server on a private
network, issue #938). Built-in public forge domains (github.com, gitlab.com,
...) are NOT blindly trusted -- they are IP-validated like any other host, which
closes the trusted-domain SSRF bypass.
"""

import logging
from unittest.mock import patch

logger = logging.getLogger(__name__)


def _clear_bypass_cache() -> None:
    """Reset the lru_cache so each test sees the patched settings value."""
    from registry.utils.url_guard import _skill_allowlist

    _skill_allowlist.cache_clear()


def _bypass_hosts() -> frozenset[str]:
    """Return the skill-fetch bypass host set derived from current settings."""
    from registry.utils.url_guard import _skill_allowlist

    return _skill_allowlist().hosts


class TestBypassAllowlistDerivation:
    """The operator bypass allowlist is derived solely from github_extra_hosts."""

    @patch("registry.utils.url_guard.settings")
    def test_empty_when_unconfigured(
        self,
        mock_settings,
    ) -> None:
        """No configured hosts means an empty bypass set (public domains not in it)."""
        mock_settings.github_extra_hosts = ""
        _clear_bypass_cache()

        assert _bypass_hosts() == frozenset()

    @patch("registry.utils.url_guard.settings")
    def test_public_forge_domains_not_in_bypass(
        self,
        mock_settings,
    ) -> None:
        """Built-in public forge domains are never auto-bypassed."""
        mock_settings.github_extra_hosts = ""
        _clear_bypass_cache()

        assert "github.com" not in _bypass_hosts()
        assert "gitlab.com" not in _bypass_hosts()

    @patch("registry.utils.url_guard.settings")
    def test_single_ghes_host_added(
        self,
        mock_settings,
    ) -> None:
        """One configured GHES host is the only member of the bypass set."""
        mock_settings.github_extra_hosts = "github.mycompany.com"
        _clear_bypass_cache()

        assert _bypass_hosts() == frozenset({"github.mycompany.com"})

    @patch("registry.utils.url_guard.settings")
    def test_whitespace_and_case_normalised(
        self,
        mock_settings,
    ) -> None:
        """Whitespace is stripped and hostnames are lowercased."""
        mock_settings.github_extra_hosts = "  GitHub.MyCompany.com  ,  RAW.github.mycompany.com  "
        _clear_bypass_cache()

        assert _bypass_hosts() == frozenset({"github.mycompany.com", "raw.github.mycompany.com"})


class TestSafeUrlForGHES:
    """GHES URLs bypass the private-IP check only once explicitly configured."""

    @patch("registry.utils.url_guard.settings")
    def test_ghes_url_blocked_when_not_configured(
        self,
        mock_settings,
    ) -> None:
        """Without github_extra_hosts, GHES on a private IP fails SSRF."""
        mock_settings.github_extra_hosts = ""
        _clear_bypass_cache()

        with patch("registry.utils.url_guard.socket.getaddrinfo") as mock_resolve:
            mock_resolve.return_value = [
                (None, None, None, None, ("10.0.0.5", 443)),
            ]

            from registry.services.skill_service import _is_safe_url

            url = "https://github.mycompany.com/org/repo/blob/main/SKILL.md"
            assert _is_safe_url(url) is False

    @patch("registry.utils.url_guard.settings")
    def test_ghes_url_allowed_when_configured(
        self,
        mock_settings,
    ) -> None:
        """With github_extra_hosts set, GHES on a private IP passes SSRF.

        DNS resolution should be skipped entirely for bypass hosts -- patch
        getaddrinfo to fail loudly if it gets called.
        """
        mock_settings.github_extra_hosts = "github.mycompany.com"
        _clear_bypass_cache()

        with patch("registry.utils.url_guard.socket.getaddrinfo") as mock_resolve:
            mock_resolve.side_effect = AssertionError(
                "getaddrinfo should not be called for bypass hosts"
            )

            from registry.services.skill_service import _is_safe_url

            url = "https://github.mycompany.com/org/repo/blob/main/SKILL.md"
            assert _is_safe_url(url) is True

    @patch("registry.utils.url_guard.settings")
    def test_public_github_on_private_ip_is_blocked(
        self,
        mock_settings,
    ) -> None:
        """A public forge domain resolving to a private IP is now BLOCKED.

        This is the core hardening: github.com is no longer a blind-trust
        bypass, so an internal host masquerading as github.com (or DNS
        poisoning) cannot reach private ranges.
        """
        mock_settings.github_extra_hosts = ""
        _clear_bypass_cache()

        with patch("registry.utils.url_guard.socket.getaddrinfo") as mock_resolve:
            mock_resolve.return_value = [
                (None, None, None, None, ("10.0.0.5", 443)),
            ]

            from registry.services.skill_service import _is_safe_url

            assert _is_safe_url("https://github.com/org/repo/blob/main/SKILL.md") is False

    @patch("registry.utils.url_guard.settings")
    def test_public_github_on_public_ip_is_allowed(
        self,
        mock_settings,
    ) -> None:
        """A public forge domain resolving to a public IP is allowed."""
        mock_settings.github_extra_hosts = ""
        _clear_bypass_cache()

        with patch("registry.utils.url_guard.socket.getaddrinfo") as mock_resolve:
            mock_resolve.return_value = [
                (None, None, None, None, ("140.82.112.3", 443)),
            ]

            from registry.services.skill_service import _is_safe_url

            assert _is_safe_url("https://github.com/org/repo/blob/main/SKILL.md") is True

    @patch("registry.utils.url_guard.settings")
    def test_unconfigured_internal_host_still_blocked(
        self,
        mock_settings,
    ) -> None:
        """A non-GitHub internal host on a private IP is still blocked."""
        mock_settings.github_extra_hosts = "github.mycompany.com"
        _clear_bypass_cache()

        with patch("registry.utils.url_guard.socket.getaddrinfo") as mock_resolve:
            mock_resolve.return_value = [
                (None, None, None, None, ("10.0.0.5", 443)),
            ]

            from registry.services.skill_service import _is_safe_url

            assert _is_safe_url("https://internal.example.com/foo") is False
