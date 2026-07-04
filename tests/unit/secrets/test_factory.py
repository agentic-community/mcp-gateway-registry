"""SecretStore factory tests: backend selection + singleton + reset."""

import pytest

from registry.secrets import factory
from registry.secrets.interfaces import SecretStoreBase


class _FakeStore(SecretStoreBase):
    """Minimal SecretStoreBase stand-in for selection/singleton tests."""

    async def put_token(self, auth_method, user_id, provider, server_path, token):  # noqa: D102
        raise NotImplementedError

    async def get_token(self, auth_method, user_id, provider, server_path):  # noqa: D102
        raise NotImplementedError

    async def delete_token(self, auth_method, user_id, provider, server_path):  # noqa: D102
        raise NotImplementedError

    async def list_for_user(self, auth_method, user_id):  # noqa: D102
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _reset_singleton():
    factory.reset_secret_store()
    yield
    factory.reset_secret_store()


@pytest.fixture
def _openbao_backend(monkeypatch):
    """Select the openbao backend with a stubbed builder (no hvac/login)."""
    monkeypatch.setattr(factory.settings, "secret_store_backend", "openbao")
    monkeypatch.setattr(factory, "_build_openbao", lambda: _FakeStore())


@pytest.mark.unit
class TestSecretStoreFactory:
    def test_builds_configured_backend(self, _openbao_backend):
        store = factory.get_secret_store()
        assert isinstance(store, SecretStoreBase)

    def test_singleton_returns_same_instance(self, _openbao_backend):
        assert factory.get_secret_store() is factory.get_secret_store()

    def test_reset_clears_singleton(self, _openbao_backend):
        first = factory.get_secret_store()
        factory.reset_secret_store()
        assert factory.get_secret_store() is not first

    def test_secrets_manager_requires_region(self, monkeypatch):
        """With no explicit region AND no AWS_REGION/AWS_DEFAULT_REGION env, fail."""
        monkeypatch.setattr(factory.settings, "secret_store_backend", "secrets-manager")
        monkeypatch.setattr(factory.settings, "aws_secrets_region", "")
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        with pytest.raises(ValueError, match="requires a region"):
            factory.get_secret_store()

    def test_secrets_manager_falls_back_to_aws_region_env(self, monkeypatch):
        """When AWS_SECRETS_REGION is unset, AWS_REGION is used (no error)."""
        monkeypatch.setattr(factory.settings, "secret_store_backend", "secrets-manager")
        monkeypatch.setattr(factory.settings, "aws_secrets_region", "")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        store = factory.get_secret_store()
        assert isinstance(store, SecretStoreBase)

    def test_secrets_manager_falls_back_to_aws_default_region_env(self, monkeypatch):
        """AWS_DEFAULT_REGION is also honored when AWS_REGION is absent."""
        monkeypatch.setattr(factory.settings, "secret_store_backend", "secrets-manager")
        monkeypatch.setattr(factory.settings, "aws_secrets_region", "")
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
        store = factory.get_secret_store()
        assert isinstance(store, SecretStoreBase)

    def test_secrets_manager_explicit_region_takes_precedence(self, monkeypatch):
        """An explicit AWS_SECRETS_REGION wins over the env fallback."""
        monkeypatch.setattr(factory.settings, "secret_store_backend", "secrets-manager")
        monkeypatch.setattr(factory.settings, "aws_secrets_region", "ap-south-1")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        store = factory.get_secret_store()
        assert isinstance(store, SecretStoreBase)

    def test_openbao_requires_addr(self, monkeypatch):
        monkeypatch.setattr(factory.settings, "secret_store_backend", "openbao")
        monkeypatch.setattr(factory.settings, "openbao_addr", "")
        with pytest.raises(ValueError, match="OPENBAO_ADDR"):
            factory.get_secret_store()
