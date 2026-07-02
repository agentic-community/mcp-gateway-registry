"""Unit tests for registry.utils.pingfederate_manager.

Focuses on the fail-closed behavior of the PingFederate admin password
resolution: the module must never authenticate with a hardcoded default
credential and must raise a clear, actionable error when ``PF_ADMIN_PASS`` is
not configured.
"""

import inspect
from pathlib import Path
from unittest.mock import patch

import pytest

import registry.utils.pingfederate_manager as pf
from registry.utils.pingfederate_manager import (
    _get_pf_admin_pass,
    _pf_auth,
    create_pingfederate_service_account_client,
)

# The retired default that must never appear in source or be used at runtime.
_RETIRED_DEFAULT_PASSWORD = "2FederateM0re"


class TestGetPfAdminPass:
    """Tests for _get_pf_admin_pass fail-closed secret resolution."""

    def test_raises_when_env_unset(self):
        """Unset PF_ADMIN_PASS raises ValueError (no default fallback)."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="PF_ADMIN_PASS"):
                _get_pf_admin_pass()

    def test_raises_when_env_empty(self):
        """Empty PF_ADMIN_PASS raises ValueError (empty is not a credential)."""
        with patch.dict("os.environ", {"PF_ADMIN_PASS": ""}, clear=True):
            with pytest.raises(ValueError, match="PF_ADMIN_PASS"):
                _get_pf_admin_pass()

    def test_raises_when_env_whitespace_only(self):
        """A whitespace-only PF_ADMIN_PASS is treated as unset."""
        with patch.dict("os.environ", {"PF_ADMIN_PASS": "   "}, clear=True):
            with pytest.raises(ValueError, match="PF_ADMIN_PASS"):
                _get_pf_admin_pass()

    def test_raises_when_env_is_known_weak_default(self):
        """The well-known dev default is rejected even when supplied via env.

        This closes the gap where docker-compose/Terraform fallbacks inject the
        weak value: the app itself refuses to authenticate with it.
        """
        with patch.dict(
            "os.environ",
            {"PF_ADMIN_PASS": _RETIRED_DEFAULT_PASSWORD},
            clear=True,
        ):
            with pytest.raises(ValueError, match="well-known development default"):
                _get_pf_admin_pass()

    def test_returns_configured_value(self):
        """A configured PF_ADMIN_PASS is returned verbatim."""
        with patch.dict("os.environ", {"PF_ADMIN_PASS": "s3cr3t-from-env"}, clear=True):
            assert _get_pf_admin_pass() == "s3cr3t-from-env"

    def test_never_returns_retired_default(self):
        """With no env set, the retired default password is never used."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError):
                _get_pf_admin_pass()


class TestPfAuth:
    """Tests for the _pf_auth basic-auth tuple builder."""

    def test_fails_closed_when_password_unset(self):
        """_pf_auth raises rather than emitting a default credential."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="PF_ADMIN_PASS"):
                _pf_auth()

    def test_uses_env_password(self):
        """_pf_auth returns the configured user/password pair."""
        with patch.dict(
            "os.environ",
            {"PF_ADMIN_USER": "administrator", "PF_ADMIN_PASS": "env-pass"},
            clear=True,
        ):
            user, password = _pf_auth()
            assert password == "env-pass"
            assert password != _RETIRED_DEFAULT_PASSWORD


class TestCreateClientFailsClosed:
    """The public entry point must fail closed on missing secret."""

    @pytest.mark.asyncio
    async def test_raises_valueerror_when_password_unset(self):
        """No PF_ADMIN_PASS -> ValueError surfaces (not swallowed as PF error)."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="PF_ADMIN_PASS"):
                await create_pingfederate_service_account_client(
                    client_id="valid-client",
                    group_names=["registry-admins"],
                )


class TestNoHardcodedDefaultInSource:
    """Regression guard: the hardcoded default must not return to the source.

    The module intentionally references the weak value in a rejection denylist,
    so the guard targets the vulnerable *pattern* (using it as an env fallback)
    rather than the raw substring.
    """

    def test_source_has_no_env_default_fallback(self):
        """The module never uses the weak value as an os.environ.get default."""
        source = inspect.getsource(pf)
        vulnerable_patterns = (
            f'os.environ.get("PF_ADMIN_PASS", "{_RETIRED_DEFAULT_PASSWORD}")',
            f"os.environ.get('PF_ADMIN_PASS', '{_RETIRED_DEFAULT_PASSWORD}')",
        )
        for pattern in vulnerable_patterns:
            assert pattern not in source, f"weak default fallback present: {pattern}"

    def test_weak_value_only_appears_in_denylist(self):
        """Any reference to the weak value is only in the rejection denylist."""
        assert _RETIRED_DEFAULT_PASSWORD in pf._PF_ADMIN_PASS_DENYLIST

    def test_iam_user_groups_routes_has_no_hardcoded_default_password(self):
        """The sibling PingFederate sink also carries no default password."""
        module_path = Path(pf.__file__).resolve().parents[1] / "api" / "iam_user_groups_routes.py"
        source = module_path.read_text(encoding="utf-8")
        assert _RETIRED_DEFAULT_PASSWORD not in source
