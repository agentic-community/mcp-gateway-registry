"""Unit tests for the shared SECRET_KEY validator.

These tests pin the fail-closed behaviour of ``validate_secret_key`` (used by
both the auth server and the registry startup guards) and assert that the
provider modules no longer ship a hardcoded default key.
"""

from pathlib import Path

import pytest

from registry.common.secret_key import (
    MIN_SECRET_KEY_LENGTH,
    validate_secret_key,
)

# Provider modules that historically carried the "development-secret-key"
# default. They must now read SECRET_KEY without a fallback.
_PROVIDER_MODULES: tuple[str, ...] = (
    "auth_server/providers/keycloak.py",
    "auth_server/providers/okta.py",
    "auth_server/providers/auth0.py",
    "auth_server/providers/entra.py",
    "auth_server/providers/pingfederate.py",
)

_REPO_ROOT: Path = Path(__file__).resolve().parents[3]


@pytest.mark.unit
@pytest.mark.core
class TestValidateSecretKey:
    """Behavioural tests for validate_secret_key."""

    def test_none_rejected(self) -> None:
        """An unset (None) key is rejected."""
        with pytest.raises(RuntimeError, match="required"):
            validate_secret_key(None)

    def test_empty_string_rejected(self) -> None:
        """An empty string is rejected."""
        with pytest.raises(RuntimeError, match="required"):
            validate_secret_key("")

    def test_short_key_rejected(self) -> None:
        """A key shorter than the minimum length is rejected."""
        short_key = "x" * (MIN_SECRET_KEY_LENGTH - 1)
        with pytest.raises(RuntimeError, match="at least 32"):
            validate_secret_key(short_key)

    def test_weak_development_default_rejected(self) -> None:
        """The historical 'development-secret-key' literal is rejected."""
        with pytest.raises(RuntimeError, match="well-known placeholder"):
            validate_secret_key("development-secret-key")

    def test_weak_default_rejected_case_insensitively(self) -> None:
        """Weak literals are matched regardless of case or surrounding space."""
        with pytest.raises(RuntimeError, match="well-known placeholder"):
            validate_secret_key("  Development-Secret-Key  ")

    def test_env_example_default_rejected(self) -> None:
        """The exact value shipped in .env.example is rejected.

        This value is 60+ characters, so it passes the length check; it must be
        caught by the weak-value guard instead.
        """
        env_example_value = "CHANGE-THIS-IMMEDIATELY-use-a-strong-random-key-in-production"
        assert len(env_example_value) >= MIN_SECRET_KEY_LENGTH
        with pytest.raises(RuntimeError, match="well-known placeholder"):
            validate_secret_key(env_example_value)

    def test_placeholder_marker_substring_rejected(self) -> None:
        """A long key that merely contains a placeholder marker is rejected."""
        with pytest.raises(RuntimeError, match="well-known placeholder"):
            validate_secret_key("prefix-CHANGE-THIS-IMMEDIATELY-suffix-padding-1234")

    def test_valid_long_key_accepted_and_returned(self) -> None:
        """A random 32+ character key passes and is returned unchanged."""
        valid = "a-sufficiently-long-random-secret-key-value"
        assert validate_secret_key(valid) == valid

    def test_exactly_minimum_length_accepted(self) -> None:
        """A key of exactly the minimum length is accepted."""
        valid = "a" * MIN_SECRET_KEY_LENGTH
        assert validate_secret_key(valid) == valid

    def test_valid_key_is_returned_whitespace_stripped(self) -> None:
        """A valid key is returned with surrounding whitespace stripped.

        Stripping is applied consistently so replicas whose SECRET_KEY differs
        only by accidental whitespace still derive the same signing key.
        """
        core = "a-sufficiently-long-random-secret-key-value"
        assert validate_secret_key(f"  {core}\n") == core

    def test_whitespace_only_rejected(self) -> None:
        """A whitespace-only key is treated as unset and rejected."""
        with pytest.raises(RuntimeError, match="required"):
            validate_secret_key("   \t\n  ")

    def test_whitespace_padded_short_key_rejected(self) -> None:
        """Padding a short key with whitespace does not satisfy the length check."""
        padded = "   short-key   "  # 9 real characters, well under the minimum
        with pytest.raises(RuntimeError, match="too short"):
            validate_secret_key(padded)


@pytest.mark.unit
@pytest.mark.core
class TestProviderModulesHaveNoDefaultSecret:
    """Guard against the hardcoded default key reappearing in providers."""

    @pytest.mark.parametrize("module_rel_path", _PROVIDER_MODULES)
    def test_no_hardcoded_default_secret_key(self, module_rel_path: str) -> None:
        """No provider module may embed the 'development-secret-key' default."""
        source = (_REPO_ROOT / module_rel_path).read_text(encoding="utf-8")
        assert "development-secret-key" not in source, (
            f"{module_rel_path} still contains the hardcoded default SECRET_KEY"
        )
        assert 'os.environ.get("SECRET_KEY", ' not in source, (
            f"{module_rel_path} still provides a fallback default for SECRET_KEY"
        )
