"""
Unit tests for _parse_and_validate_custom_headers in server_routes.py.
"""

import json

import pytest
from fastapi import HTTPException

from registry.api.server_routes import _parse_and_validate_custom_headers


class TestParseAndValidateCustomHeaders:
    """Tests for the shared validation helper."""

    def test_none_input_returns_none(self):
        assert _parse_and_validate_custom_headers(None) is None

    def test_empty_list_returns_empty(self):
        result = _parse_and_validate_custom_headers("[]")
        assert result == []

    def test_valid_headers(self):
        raw = json.dumps([
            {"name": "X-Tenant-Id", "value": "42"},
            {"name": "X-Route", "value": "prod"},
        ])
        result = _parse_and_validate_custom_headers(raw)
        assert len(result) == 2
        assert result[0] == {"name": "X-Tenant-Id", "value": "42"}

    def test_invalid_json(self):
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_custom_headers("not-json{")
        assert exc_info.value.status_code == 400
        assert "Invalid JSON" in exc_info.value.detail

    def test_not_a_list(self):
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_custom_headers('{"name": "X-Foo"}')
        assert exc_info.value.status_code == 400
        assert "JSON array" in exc_info.value.detail

    def test_too_many_headers(self):
        headers = [{"name": f"X-H-{i}", "value": "v"} for i in range(11)]
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_custom_headers(json.dumps(headers))
        assert exc_info.value.status_code == 400
        assert "Too many" in exc_info.value.detail

    def test_reserved_name_rejected(self):
        raw = json.dumps([{"name": "Authorization", "value": "Bearer x"}])
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_custom_headers(raw)
        assert exc_info.value.status_code == 400
        assert "managed by the gateway" in exc_info.value.detail

    def test_reserved_name_case_insensitive(self):
        raw = json.dumps([{"name": "CONTENT-TYPE", "value": "text/plain"}])
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_custom_headers(raw)
        assert exc_info.value.status_code == 400

    def test_duplicate_names_rejected(self):
        raw = json.dumps([
            {"name": "X-Foo", "value": "a"},
            {"name": "x-foo", "value": "b"},
        ])
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_custom_headers(raw)
        assert exc_info.value.status_code == 400
        assert "Duplicate" in exc_info.value.detail

    def test_invalid_header_name_characters(self):
        raw = json.dumps([{"name": "X Header", "value": "v"}])
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_custom_headers(raw)
        assert exc_info.value.status_code == 400
        assert "Invalid custom header" in exc_info.value.detail

    def test_cr_lf_in_value_rejected(self):
        raw = json.dumps([{"name": "X-Foo", "value": "val\r\nInjection: bad"}])
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_custom_headers(raw)
        assert exc_info.value.status_code == 400

    def test_empty_value_rejected_by_default(self):
        raw = json.dumps([{"name": "X-Foo", "value": ""}])
        with pytest.raises(HTTPException) as exc_info:
            _parse_and_validate_custom_headers(raw)
        assert exc_info.value.status_code == 400
        assert "empty value" in exc_info.value.detail

    def test_empty_value_allowed_in_edit_mode(self):
        raw = json.dumps([{"name": "X-Foo", "value": ""}])
        result = _parse_and_validate_custom_headers(raw, allow_empty_values=True)
        assert result == [{"name": "X-Foo", "value": ""}]
