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


class _FakeAuthMethod:
    """Records the keyword arguments of each login call."""

    def __init__(
        self,
        name: str,
        calls: list[tuple[str, dict]],
    ) -> None:
        self._name = name
        self._calls = calls

    def login(self, **kwargs) -> None:  # noqa: D102
        self._calls.append((self._name, kwargs))


class _FakeHvacClient:
    """hvac.Client stand-in exposing only the auth methods the factory uses."""

    instances: list["_FakeHvacClient"] = []

    def __init__(self, **kwargs) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.token = None
        self.auth = type(
            "_Auth",
            (),
            {
                "kubernetes": _FakeAuthMethod("kubernetes", self.calls),
                "approle": _FakeAuthMethod("approle", self.calls),
            },
        )()
        _FakeHvacClient.instances.append(self)


@pytest.fixture
def _openbao_login(monkeypatch, tmp_path):
    """Build the real openbao backend against a fake hvac client and store."""
    import sys
    import types

    from registry.secrets.openbao import store as openbao_store

    _FakeHvacClient.instances = []
    monkeypatch.setitem(sys.modules, "hvac", types.SimpleNamespace(Client=_FakeHvacClient))
    monkeypatch.setattr(openbao_store, "OpenBaoStore", lambda **kwargs: _FakeStore())
    monkeypatch.setattr(factory, "_build_credential_codec", lambda: None)
    monkeypatch.setattr(factory.settings, "openbao_addr", "https://vault.example.com:8200")
    monkeypatch.setattr(factory.settings, "openbao_namespace", "")
    monkeypatch.setattr(factory.settings, "openbao_role", "registry")
    monkeypatch.setenv("OPENBAO_ROLE_ID", "role-id")
    monkeypatch.setenv("OPENBAO_SECRET_ID", "secret-id")

    sa_token = tmp_path / "token"
    sa_token.write_text("sa-jwt")
    real_open = open

    def _open(path, *args, **kwargs):
        if path == "/var/run/secrets/kubernetes.io/serviceaccount/token":
            return real_open(sa_token, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _open)

    def _login_calls(
        method: str,
        auth_mount: str,
    ) -> list[tuple[str, dict]]:
        monkeypatch.setattr(factory.settings, "openbao_auth_method", method)
        monkeypatch.setattr(factory.settings, "openbao_auth_mount", auth_mount)
        factory._build_openbao()
        return _FakeHvacClient.instances[-1].calls

    return _login_calls


@pytest.mark.unit
class TestOpenBaoAuthMount:
    def test_kubernetes_default_mount_unchanged(self, _openbao_login):
        """No OPENBAO_AUTH_MOUNT: hvac keeps its default auth/kubernetes path."""
        calls = _openbao_login("kubernetes", "")
        assert calls == [("kubernetes", {"role": "registry", "jwt": "sa-jwt"})]

    def test_kubernetes_custom_mount(self, _openbao_login):
        calls = _openbao_login("kubernetes", "k8s-prod")
        assert calls == [
            ("kubernetes", {"role": "registry", "jwt": "sa-jwt", "mount_point": "k8s-prod"})
        ]

    def test_approle_custom_mount(self, _openbao_login):
        calls = _openbao_login("approle", "approle-gateway")
        assert calls == [
            (
                "approle",
                {"role_id": "role-id", "secret_id": "secret-id", "mount_point": "approle-gateway"},
            )
        ]

    def test_mount_slashes_stripped(self, _openbao_login):
        calls = _openbao_login("kubernetes", " /k8s-prod/ ")
        assert calls[0][1]["mount_point"] == "k8s-prod"
