"""Tests for registry/common/request_origin.py."""

import pytest

from registry.common.request_origin import (
    ORIGIN_MALFORMED,
    ORIGIN_MISSING,
    ORIGIN_NULL,
    ORIGIN_UNTRUSTED,
    normalize_origin,
    origin_of_url,
    origin_refusal,
)

TRUSTED: frozenset[str] = frozenset({"https://gw.example.com", "http://localhost"})


class TestNormalizeOrigin:
    """normalize_origin accepts exactly scheme://host[:port]."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("https://GW.Example.com", "https://gw.example.com"),
            ("http://localhost:8080", "http://localhost:8080"),
            ("http://[::1]:8080", "http://[::1]:8080"),
            ("https://gw.example.com.", "https://gw.example.com"),
            ("https://gw.example.com/", "https://gw.example.com"),
        ],
    )
    def test_accepts_an_exact_origin(
        self,
        value,
        expected,
    ):
        assert normalize_origin(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "*",
            "https://*.example.com",
            "ftp://gw.example.com",
            "https://gw.example.com/path",
            "https://gw.example.com?q=1",
            "https://user:pw@gw.example.com",
            "https://gw.example.com:99999",
            "gw.example.com",
        ],
    )
    def test_rejects_anything_else(
        self,
        value,
    ):
        assert normalize_origin(value) is None


class TestOriginOfUrl:
    """origin_of_url keeps scheme, host and port of a configured URL."""

    def test_drops_path_and_default_port(self):
        assert origin_of_url("https://gw.example.com:443/registry") == "https://gw.example.com"

    def test_keeps_a_nondefault_port(self):
        assert origin_of_url("http://localhost:8888") == "http://localhost:8888"

    @pytest.mark.parametrize("value", ["", "   ", "localhost", "/relative/path"])
    def test_returns_none_for_a_url_without_scheme_and_host(
        self,
        value,
    ):
        assert origin_of_url(value) is None


class TestOriginRefusal:
    """origin_refusal fails closed for anything but a trusted origin."""

    def test_allows_a_trusted_origin(self):
        assert origin_refusal("https://gw.example.com", "POST", TRUSTED) is None

    def test_allows_a_trusted_origin_with_explicit_default_port(self):
        assert origin_refusal("http://localhost:80", "POST", TRUSTED) is None

    @pytest.mark.parametrize("verb", ["GET", "head", "OPTIONS"])
    def test_allows_a_safe_verb_without_origin(
        self,
        verb,
    ):
        assert origin_refusal(None, verb, TRUSTED) is None

    @pytest.mark.parametrize("verb", ["POST", "DELETE", "PUT", ""])
    def test_refuses_an_unsafe_or_unknown_verb_without_origin(
        self,
        verb,
    ):
        assert origin_refusal(None, verb, TRUSTED) == ORIGIN_MISSING

    def test_refuses_null(self):
        assert origin_refusal("null", "POST", TRUSTED) == ORIGIN_NULL

    def test_refuses_a_malformed_origin(self):
        assert origin_refusal("https://gw.example.com/x", "POST", TRUSTED) == ORIGIN_MALFORMED

    def test_refuses_another_origin_even_on_a_get(self):
        assert origin_refusal("https://evil.example", "GET", TRUSTED) == ORIGIN_UNTRUSTED

    def test_refuses_a_lookalike_subdomain(self):
        assert origin_refusal("https://gw.example.com.evil.example", "POST", TRUSTED) == (
            ORIGIN_UNTRUSTED
        )

    def test_refuses_everything_when_nothing_is_trusted(self):
        assert origin_refusal("https://gw.example.com", "POST", frozenset()) == ORIGIN_UNTRUSTED
