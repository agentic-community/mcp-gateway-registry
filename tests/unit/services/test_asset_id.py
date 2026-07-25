"""Unit tests for resolve_asset_id (registry/services/_asset_id.py)."""

import uuid

import pytest

from registry.services._asset_id import (
    MAX_ID_LENGTH,
    CallerSuppliedIdDisabledError,
    InvalidAssetIdError,
    check_caller_supplied_id_allowed,
    resolve_asset_id,
    validate_asset_id,
)


def test_none_generates_a_uuid4_string():
    result = resolve_asset_id(None)
    assert isinstance(result, str)  # a string, not a UUID object
    assert uuid.UUID(result).version == 4  # and a real v4 UUID


def test_supplied_arn_returned_verbatim():
    arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/my-runtime"
    assert resolve_asset_id(arn) == arn


def test_supplied_uuid_string_is_kept_not_regenerated():
    fixed = "11111111-1111-4111-8111-111111111111"
    assert resolve_asset_id(fixed) == fixed


def test_surrounding_whitespace_is_stripped():
    assert resolve_asset_id("  urn:custom:thing  ") == "urn:custom:thing"


@pytest.mark.parametrize("blank", ["", "   ", "\t", "\n", " \t\n "])
def test_blank_or_whitespace_only_raises(blank):
    with pytest.raises(InvalidAssetIdError):
        resolve_asset_id(blank)


def test_exactly_max_length_is_allowed():
    ok = "x" * MAX_ID_LENGTH
    assert resolve_asset_id(ok) == ok


def test_over_max_length_raises():
    with pytest.raises(InvalidAssetIdError):
        resolve_asset_id("x" * (MAX_ID_LENGTH + 1))


@pytest.mark.parametrize("bad", ["line\none", "tab\there", "bell\x07", "del\x7f"])
def test_control_characters_raise(bad):
    with pytest.raises(InvalidAssetIdError):
        resolve_asset_id(bad)


def test_error_is_a_valueerror_subclass():
    # Pydantic + the route both rely on this to produce a 422.
    assert issubclass(InvalidAssetIdError, ValueError)


# --- Safe-charset floor (#1276) ------------------------------------------


@pytest.mark.parametrize(
    "good",
    [
        "11111111-1111-4111-8111-111111111111",  # UUID
        "arn:aws:iam::123456789012:role/my-role",  # ARN
        "urn:example:agent:1",  # URN
        "peer#42",  # peer id with '#'
        "a.b_c-d:e/f@g#h=i+j",  # every allowed punctuation char
        "ABCabc0189",  # plain alphanumerics
    ],
)
def test_safe_charset_accepts_expected_id_shapes(good):
    assert validate_asset_id(good) == good


@pytest.mark.parametrize(
    "bad",
    [
        "has space",
        "semi;colon",
        "pipe|char",
        "dollar$sign",
        "brace{x}",
        "angle<x>",
        "quote'x",
        'doublequote"x',
        "back\\slash",
        "tick`x",
        "star*x",
        "paren(x)",
        "amp&x",
        "percent%x",
    ],
)
def test_safe_charset_rejects_dangerous_characters(bad):
    with pytest.raises(InvalidAssetIdError):
        validate_asset_id(bad)


# --- Feature-flag gate (#1276) -------------------------------------------


def test_gate_allows_omitted_id_regardless_of_flag():
    # Omitting the id is the default for every existing caller; never gated.
    assert check_caller_supplied_id_allowed(None, feature_enabled=False) is None
    assert check_caller_supplied_id_allowed(None, feature_enabled=True) is None


def test_gate_rejects_supplied_id_when_disabled():
    with pytest.raises(CallerSuppliedIdDisabledError):
        check_caller_supplied_id_allowed("my-id", feature_enabled=False)


def test_gate_allows_supplied_id_when_enabled():
    assert check_caller_supplied_id_allowed("my-id", feature_enabled=True) is None


def test_disabled_error_is_invalid_asset_id_subclass():
    # Routes map InvalidAssetIdError -> 422; the disabled case must inherit that.
    assert issubclass(CallerSuppliedIdDisabledError, InvalidAssetIdError)
