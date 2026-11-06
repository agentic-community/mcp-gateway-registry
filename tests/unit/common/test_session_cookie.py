"""Unit tests for registry.common.session_cookie.

A configured ``SESSION_COOKIE_NAME`` must be a name a browser stores and sends
back unchanged, and must not carry a cookie prefix the browser gives special
rules to. Both services check it with ``check_session_cookie_name`` and name
the cookie with ``effective_session_cookie_name``.

Both services also read the session cookie from the raw ``Cookie`` header
through ``single_session_cookie``, which refuses a header that carries the name
more than once. One example table and one property pin the parser; the
isolation middleware's property checks it against Starlette's parser over
several ``Cookie`` fields.
"""

import logging

import pytest
from hypothesis import (
    given,
    settings,
)
from hypothesis import strategies as st

from registry.common.session_cookie import (
    check_session_cookie_name,
    effective_session_cookie_name,
    single_session_cookie,
)

_DEFAULT_NAME: str = "mcp_gateway_session"
_HOST_NAME: str = "__Host-" + _DEFAULT_NAME
# Names that resemble the session cookie name without being it.
_OTHER_NAMES: tuple[str, ...] = (
    "theme",
    "old_" + _DEFAULT_NAME,
    _DEFAULT_NAME + "_old",
    _DEFAULT_NAME.upper(),
    _DEFAULT_NAME,
    _HOST_NAME,
)
# A cookie value as a browser sends it: printable ASCII without ";" (which ends
# the pair) and without surrounding spaces, so "=" and quotes can appear.
_COOKIE_VALUES = st.text(
    alphabet=st.characters(min_codepoint=0x21, max_codepoint=0x7E, exclude_characters=";"),
    max_size=12,
)
_PAIR_SEPARATORS = st.sampled_from([";", "; ", " ;  "])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("configured", "host_prefix", "expected"),
    [
        pytest.param("custom_session", False, "custom_session", id="plain"),
        pytest.param(" custom_session ", False, "custom_session", id="surrounding-whitespace"),
        pytest.param("Session-Name.v2_x", False, "Session-Name.v2_x", id="token-punctuation"),
        pytest.param("a!#$%&'*+^`|~z", False, "a!#$%&'*+^`|~z", id="token-symbols"),
        pytest.param("my__Host-session", False, "my__Host-session", id="prefix-not-leading"),
        pytest.param("custom_session", True, "__Host-custom_session", id="host-prefix-on"),
    ],
)
def test_accepts_a_cookie_token_name(configured, host_prefix, expected):
    """A valid RFC 6265 token name is kept, with __Host- in front when the prefix is on."""
    name = check_session_cookie_name(configured)

    assert effective_session_cookie_name(name, host_prefix) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("configured", "reason"),
    [
        pytest.param("", "empty", id="empty"),
        pytest.param("   ", "empty", id="whitespace-only"),
        pytest.param("__Host-session", "prefix", id="host-prefix"),
        pytest.param("__host-session", "prefix", id="host-prefix-lowercase"),
        pytest.param("__SECURE-session", "prefix", id="secure-prefix-uppercase"),
        pytest.param("__Secure-session", "prefix", id="secure-prefix"),
        pytest.param("my session", "character", id="inner-space"),
        pytest.param("my\tsession", "character", id="tab"),
        pytest.param("my\x01session", "character", id="control-character"),
        pytest.param("my\x7fsession", "character", id="delete-character"),
        pytest.param("sessión", "character", id="non-ascii"),
        *(
            pytest.param(f"my{separator}session", "character", id=f"separator-{separator}")
            for separator in '()<>@,;:\\"/[]?={}'
        ),
    ],
)
def test_refuses_a_name_a_browser_would_not_keep(configured, reason):
    """An empty, prefixed or non-token name raises a ValueError naming the variable."""
    with pytest.raises(ValueError, match="SESSION_COOKIE_NAME") as raised:
        check_session_cookie_name(configured)

    assert reason in str(raised.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("session_name", "cookie_header", "expected"),
    [
        pytest.param(_DEFAULT_NAME, None, None, id="no-header"),
        pytest.param(_DEFAULT_NAME, "theme=dark; lang=en", None, id="absent"),
        pytest.param(_DEFAULT_NAME, f"{_DEFAULT_NAME}=", "", id="one-empty-value"),
        pytest.param(
            _DEFAULT_NAME,
            f"theme=dark; old_{_DEFAULT_NAME}=x; {_DEFAULT_NAME}_old=y; "
            f"MCP_GATEWAY_SESSION=z; {_DEFAULT_NAME}=abc; {_HOST_NAME}=w",
            "abc",
            id="look-alike-names-around-it",
        ),
        pytest.param(_DEFAULT_NAME, f"{_DEFAULT_NAME}; theme=dark", None, id="pair-without-equals"),
        pytest.param(
            _DEFAULT_NAME, f"  theme=dark ;  {_DEFAULT_NAME} =  a=b==  ", "a=b==", id="whitespace"
        ),
        pytest.param(_DEFAULT_NAME, f'{_DEFAULT_NAME}="abc"', '"abc"', id="quoted-value-verbatim"),
        pytest.param(
            _DEFAULT_NAME,
            f"{_DEFAULT_NAME}=abc;{_DEFAULT_NAME} = abc ",
            None,
            id="two-equal-copies",
        ),
        pytest.param(
            _HOST_NAME, f"{_DEFAULT_NAME}=tossed; {_HOST_NAME}=real", "real", id="host-prefixed"
        ),
        pytest.param(
            _HOST_NAME, f"{_HOST_NAME}=real; {_HOST_NAME}=tossed", None, id="host-prefixed-twice"
        ),
    ],
)
def test_single_session_cookie_returns_the_value_only_when_sent_once(
    session_name, cookie_header, expected
):
    """The session cookie is its value when the exact name appears once, and None otherwise.

    Values come back verbatim (quotes and inner ``=`` kept) without the
    surrounding whitespace.
    """
    assert single_session_cookie(cookie_header, session_name) == expected


@pytest.mark.unit
@pytest.mark.parametrize("session_name", [_DEFAULT_NAME, _HOST_NAME], ids=["plain", "host-prefix"])
@settings(max_examples=200, derandomize=True, database=None, report_multiple_bugs=False)
@given(
    pairs=st.lists(st.tuples(st.sampled_from(_OTHER_NAMES), _COOKIE_VALUES), max_size=6),
    separator=_PAIR_SEPARATORS,
)
def test_single_session_cookie_is_the_value_iff_the_name_occurs_once(
    session_name, pairs, separator
):
    """Property: for any header of pairs, the result is the value iff the name occurs once."""
    cookie_header = separator.join(f"{name}={value}" for name, value in pairs)
    matches = [value for name, value in pairs if name == session_name]

    expected = matches[0] if len(matches) == 1 else None
    assert single_session_cookie(cookie_header, session_name) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cookie_header", "warnings"),
    [
        pytest.param(f"{_DEFAULT_NAME}=secret-a", [], id="one"),
        pytest.param("theme=dark", [], id="absent"),
        pytest.param(f"{_DEFAULT_NAME}=secret-a; {_DEFAULT_NAME}=secret-b", ["2"], id="two"),
        pytest.param(
            f"{_DEFAULT_NAME}=secret-a; {_DEFAULT_NAME}=secret-b; {_DEFAULT_NAME}=secret-c",
            ["3"],
            id="three",
        ),
    ],
)
def test_refusal_logs_one_warning_with_the_count_and_no_value(caplog, cookie_header, warnings):
    """A refused header logs one WARNING naming how many there were, and no cookie value."""
    with caplog.at_level(logging.WARNING, logger="registry.common.session_cookie"):
        single_session_cookie(cookie_header, _DEFAULT_NAME)

    records = [r for r in caplog.records if r.name == "registry.common.session_cookie"]
    assert [r.levelno for r in records] == [logging.WARNING] * len(warnings)
    for record, count in zip(records, warnings, strict=True):
        assert count in record.getMessage()
        assert "secret-" not in record.getMessage()
