"""Unit tests for _build_provider_entry in server_routes.py.

Provider organization and URL are only meaningful as a pair. Before issue #1651
the two form handlers guarded on ``or``, so a half-filled pair entered the branch
and constructed ``AgentProvider`` with one field set to None. ``AgentProvider``
requires both, so Pydantic raised a ValidationError that no handler caught, and
the caller got a 500 with nothing indicating which field was at fault.

These tests pin the three outcomes: both absent is valid and yields no provider,
exactly one is a 400 naming the missing field, and both present builds the entry.
"""

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from registry.api.server_routes import _build_provider_entry


class TestBuildProviderEntry:
    """Tests for the shared provider-pair helper."""

    # -- Neither field supplied: the pair is optional as a whole ------------

    def test_both_none_returns_none(self):
        """Omitting the provider entirely stays valid."""
        assert _build_provider_entry(None, None) is None

    def test_both_empty_strings_return_none(self):
        """An HTML form submits "" for a field the user left alone."""
        assert _build_provider_entry("", "") is None

    def test_both_whitespace_return_none(self):
        """Whitespace-only is the same as blank, not a value."""
        assert _build_provider_entry("   ", "\t\n ") is None

    # -- Exactly one field supplied: 400, never a 500 -----------------------

    def test_organization_without_url_is_400(self):
        with pytest.raises(HTTPException) as exc_info:
            _build_provider_entry("Example Org", None)
        assert exc_info.value.status_code == 400
        assert "provider_url" in exc_info.value.detail

    def test_url_without_organization_is_400(self):
        with pytest.raises(HTTPException) as exc_info:
            _build_provider_entry(None, "https://example.com")
        assert exc_info.value.status_code == 400
        assert "provider_organization" in exc_info.value.detail

    def test_organization_with_blank_url_is_400(self):
        """The form case: the field is submitted but left empty."""
        with pytest.raises(HTTPException) as exc_info:
            _build_provider_entry("Example Org", "  ")
        assert exc_info.value.status_code == 400
        assert "provider_url" in exc_info.value.detail

    def test_blank_organization_with_url_is_400(self):
        with pytest.raises(HTTPException) as exc_info:
            _build_provider_entry("", "https://example.com")
        assert exc_info.value.status_code == 400
        assert "provider_organization" in exc_info.value.detail

    @pytest.mark.parametrize(
        ("organization", "url"),
        [
            ("Example Org", None),
            (None, "https://example.com"),
            ("Example Org", ""),
            ("", "https://example.com"),
        ],
    )
    def test_half_filled_pair_never_raises_validation_error(self, organization, url):
        """The regression: a half-filled pair must not reach Pydantic.

        An uncaught ValidationError is what produced the unhandled 500. Asserting
        the exception type (not just the status code) pins the fix at the layer
        it belongs to, so a future refactor cannot reintroduce the 500 while
        still returning 400 on some other path.
        """
        with pytest.raises(HTTPException):
            _build_provider_entry(organization, url)

        with pytest.raises(HTTPException):  # i.e. not ValidationError
            try:
                _build_provider_entry(organization, url)
            except ValidationError as exc:  # pragma: no cover - fails the test
                pytest.fail(f"half-filled pair reached Pydantic: {exc}")

    def test_error_message_names_both_fields(self):
        """The message must say what to supply, not just that something is wrong."""
        with pytest.raises(HTTPException) as exc_info:
            _build_provider_entry("Example Org", None)
        detail = exc_info.value.detail
        assert "provider_url" in detail
        assert "provider_organization" in detail

    # -- Both fields supplied: build the entry ------------------------------

    def test_both_supplied_builds_entry(self):
        result = _build_provider_entry("Example Org", "https://example.com")
        assert result == {"organization": "Example Org", "url": "https://example.com"}

    def test_values_are_stripped(self):
        """Leading/trailing whitespace from a form field is not part of the value."""
        result = _build_provider_entry("  Example Org  ", "  https://example.com  ")
        assert result == {"organization": "Example Org", "url": "https://example.com"}

    def test_result_is_a_plain_dict(self):
        """Callers assign the result straight into the server document."""
        result = _build_provider_entry("Example Org", "https://example.com")
        assert isinstance(result, dict)
