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
        monkeypatch.setattr(factory.settings, "secret_store_backend", "secrets-manager")
        monkeypatch.setattr(factory.settings, "aws_secrets_region", "")
        with pytest.raises(ValueError, match="AWS_SECRETS_REGION"):
            factory.get_secret_store()

    def test_openbao_requires_addr(self, monkeypatch):
        monkeypatch.setattr(factory.settings, "secret_store_backend", "openbao")
        monkeypatch.setattr(factory.settings, "openbao_addr", "")
        with pytest.raises(ValueError, match="OPENBAO_ADDR"):
            factory.get_secret_store()
