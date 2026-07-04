"""Regression guard against realistic-looking example credentials in source.

The M2M credential-provider CLIs render example client IDs and client secrets
in their argparse help text. Those examples must be unmistakably fake
placeholders so that neither an automated secret scanner nor a human reader
mistakes them for live credentials.

This test pins the placeholders in place and asserts the previously-committed
example values (which looked like real Okta/Auth0 credentials) are gone.
"""

from pathlib import Path

import pytest

# Repo root: tests/unit/utils/ -> parents[3].
_REPO_ROOT = Path(__file__).resolve().parents[3]

_OKTA_TOKEN_CLI = _REPO_ROOT / "credentials-provider" / "okta" / "get_m2m_token.py"
_AUTH0_TOKEN_CLI = _REPO_ROOT / "credentials-provider" / "auth0" / "get_m2m_token.py"

# Values that previously appeared in the help text and looked like real secrets.
_RETIRED_EXAMPLE_VALUES = (
    "0oa1100req1AzfKaY698",
    "EiZC6S2dyaWJ_qKmuToJ1KuZooVwOpGH4qF3N4Eao6YTFueAShId595ot9AyYCC6",
    "KhZMijfKUcl2TEJqZzrzVJb8rmwk6Qcd",
    "lbjH6Z81GkovgAHwXRV-qiKV9f6sUVzsnheJoX7KJcu2ojGXMTjJ4i0Zn49kKfVm",
    "integrator-9917255.okta.com",
)


@pytest.mark.parametrize("cli_path", [_OKTA_TOKEN_CLI, _AUTH0_TOKEN_CLI])
def test_no_retired_example_values_in_help_text(cli_path):
    """Neither credential CLI still ships the realistic-looking examples."""
    source = cli_path.read_text(encoding="utf-8")
    for value in _RETIRED_EXAMPLE_VALUES:
        assert value not in source, f"{cli_path.name} still contains '{value}'"


def test_okta_cli_uses_placeholder_values():
    """The Okta CLI advertises obvious placeholders in its help text."""
    source = _OKTA_TOKEN_CLI.read_text(encoding="utf-8")
    assert "YOUR_OKTA_CLIENT_ID" in source
    assert "YOUR_OKTA_CLIENT_SECRET" in source


def test_auth0_cli_uses_placeholder_values():
    """The Auth0 CLI advertises obvious placeholders in its help text."""
    source = _AUTH0_TOKEN_CLI.read_text(encoding="utf-8")
    assert "YOUR_AUTH0_CLIENT_ID" in source
    assert "YOUR_AUTH0_CLIENT_SECRET" in source
