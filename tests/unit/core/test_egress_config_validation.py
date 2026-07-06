"""Validation tests for the egress credential vault config.

Covers _validate_secret_store_backend (@field_validator) and the cross-field
_validate_egress_auth_config checks run in Settings.__init__:
- SECRET_STORE_BACKEND allowlist + normalization.
- egress_auth_enabled=true requires a Mongo-family storage_backend.
- callback base URL required when the feature is enabled.
"""

import pytest

from registry.core.config import ALLOWED_SECRET_STORES, Settings


def _valid_egress_kwargs(**overrides):
    base = {
        "egress_auth_enabled": True,
        "storage_backend": "mongodb-ce",
        "egress_oauth_callback_base_url": "https://gw.example",
        "secret_store_backend": "openbao",
        "auth_server_nginx_marker_secret": "a" * 32,
    }
    base.update(overrides)
    return base


@pytest.mark.unit
@pytest.mark.core
class TestSecretStoreBackendAllowlist:
    @pytest.mark.parametrize("value", sorted(ALLOWED_SECRET_STORES))
    def test_every_allowlist_value_accepted(self, monkeypatch, value):
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        s = Settings(secret_store_backend=value)  # egress off -> no cross-field checks
        assert s.secret_store_backend == value

    @pytest.mark.parametrize("bad", ["vault", "aws", "fernet", "secretsmanager", "openbao "])
    def test_unknown_value_rejected(self, bad):
        # note: "openbao " (trailing space) normalizes to "openbao" and is OK;
        # exclude it from the bad set
        if bad.strip().lower() in ALLOWED_SECRET_STORES:
            s = Settings(secret_store_backend=bad)
            assert s.secret_store_backend == bad.strip().lower()
            return
        with pytest.raises(Exception) as exc:
            Settings(secret_store_backend=bad)
        assert "SECRET_STORE_BACKEND" in str(exc.value)

    def test_empty_coerces_to_openbao(self):
        assert Settings(secret_store_backend="").secret_store_backend == "openbao"


@pytest.mark.unit
@pytest.mark.core
class TestEgressCrossFieldValidation:
    def test_disabled_skips_all_checks(self, monkeypatch):
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        # egress off + no callback url: cross-field checks are skipped entirely
        s = Settings(egress_auth_enabled=False, egress_oauth_callback_base_url="")
        assert s.egress_auth_enabled is False

    def test_non_mongo_backend_rejected_when_enabled(self, monkeypatch):
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        # "documentdb" is Mongo-family and accepted; an unknown backend is
        # rejected by the storage_backend field validator before the egress
        # cross-field check runs. ("file" was removed in v1.24.8.)
        with pytest.raises(ValueError, match="Invalid STORAGE_BACKEND"):
            Settings(**_valid_egress_kwargs(storage_backend="sqlite"))

    @pytest.mark.parametrize("backend", ["documentdb", "mongodb-ce", "mongodb", "mongodb-atlas"])
    def test_mongo_family_backends_accepted_when_enabled(self, monkeypatch, backend):
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        s = Settings(**_valid_egress_kwargs(storage_backend=backend))
        assert s.storage_backend == backend

    def test_callback_url_derives_from_registry_url_when_unset(self, monkeypatch):
        """An empty EGRESS_OAUTH_CALLBACK_BASE_URL is allowed when registry_url is
        set: the callback base derives from registry_url. It only fails when BOTH
        are empty (see test_callback_url_required_when_both_empty)."""
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        s = Settings(
            **_valid_egress_kwargs(
                egress_oauth_callback_base_url="",
                registry_url="https://gw.example",
            )
        )
        assert s.egress_oauth_callback_base == "https://gw.example"

    def test_callback_url_required_when_both_empty(self, monkeypatch):
        """Fail loudly only when neither the explicit callback base nor
        registry_url is available to derive it from."""
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        with pytest.raises(ValueError, match="EGRESS_OAUTH_CALLBACK_BASE_URL"):
            Settings(
                **_valid_egress_kwargs(
                    egress_oauth_callback_base_url="",
                    registry_url="",
                )
            )

    def test_secrets_manager_allowed_in_production(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        s = Settings(**_valid_egress_kwargs(secret_store_backend="secrets-manager"))
        assert s.secret_store_backend == "secrets-manager"

    def test_openbao_allowed_in_production(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        s = Settings(**_valid_egress_kwargs(secret_store_backend="openbao"))
        assert s.secret_store_backend == "openbao"
