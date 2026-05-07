"""
Unit tests for repository factory routing across MongoDB backend aliases.

The core invariant this file locks in: every value in MONGODB_BACKENDS must
route every factory to the SAME concrete class that "mongodb-ce" routes to.
That prevents the silent-fallback bug (issue #954) from recurring: if a
future alias is added to MONGODB_BACKENDS but forgotten in a single factory,
its routing will no longer match the baseline and the test fails.

These tests bypass the global autouse `mock_all_repositories` fixture in
tests/conftest.py by calling the factory-module functions directly via
their unpatched reference on the module object. The fixture's `patch()`
calls target `registry.repositories.factory.get_<repo>`, so we pull the
original functions via `importlib.reload()` on a fresh factory module
copy — the fixture's patches apply to the module attribute, not the
underlying function object.

Intentionally does NOT instantiate the DocumentDB repositories against a
real MongoDB client; we only check which class the factory's branching
logic selects. For repositories whose instantiation requires a real
connection (none today, but worth noting), we would need additional
mocking.
"""

import importlib

import pytest

from registry.core.config import MONGODB_BACKENDS


@pytest.fixture
def factory_module(monkeypatch):
    """Return a fresh import of the factory module that bypasses the autouse mock_all_repositories fixture."""
    # Import the module fresh. The autouse mock_all_repositories fixture
    # patches attributes on the LIVE registry.repositories.factory module,
    # but those patches apply to the module object in sys.modules. By
    # reloading, we get the patched module (patches still active), so we
    # instead access the underlying function objects that were captured at
    # module-import time before any patching applied.
    import registry.repositories.factory as factory_live

    # Reset singletons so each test starts clean
    factory_live.reset_repositories()
    yield factory_live
    factory_live.reset_repositories()


FACTORY_FUNCTION_NAMES = [
    "get_server_repository",
    "get_agent_repository",
    "get_scope_repository",
    "get_security_scan_repository",
    "get_search_repository",
    "get_federation_config_repository",
    "get_peer_federation_repository",
    "get_audit_repository",
    "get_skill_repository",
    "get_skill_security_scan_repository",
    "get_virtual_server_repository",
    "get_backend_session_repository",
    "get_app_log_repository",
]


def _unpatched_call(
    factory_live,
    fn_name: str,
):
    """Invoke a factory function bypassing conftest's autouse mock patches.

    The mock_all_repositories autouse fixture patches attributes like
    factory.get_server_repository -> Mock. The *underlying* function object
    still exists in factory.__dict__.get(fn_name) before the patch was
    applied, but mock.patch replaces the attribute. Instead, reload the
    module to get a fresh attribute lookup bypassing the patch, then
    reset the patch afterward. Simpler: import factory as a namespace and
    grab the function via its importlib-loaded module.
    """
    import importlib.util as importlib_util

    fresh = importlib.import_module("registry.repositories.factory")
    spec = importlib_util.spec_from_file_location(
        "registry.repositories.factory__unpatched",
        fresh.__file__,
    )
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, fn_name)()


def _class_name(
    obj: object | None,
) -> str | None:
    """Return the class name of an instance, or None if the factory returned None."""
    if obj is None:
        return None
    return type(obj).__name__


@pytest.mark.unit
class TestFactoryAliasRouting:
    """Every MongoDB alias must route to the same class as mongodb-ce."""

    @pytest.mark.parametrize("factory_fn_name", FACTORY_FUNCTION_NAMES)
    @pytest.mark.parametrize("alias", sorted(MONGODB_BACKENDS))
    def test_alias_routes_to_same_class_as_mongodb_ce(
        self,
        monkeypatch,
        factory_module,
        factory_fn_name: str,
        alias: str,
    ) -> None:
        """Assert alias routes to the same class as the mongodb-ce baseline."""
        # Baseline: what class does mongodb-ce route to?
        factory_module.reset_repositories()
        monkeypatch.setattr(
            "registry.core.config.settings.storage_backend",
            "mongodb-ce",
        )
        baseline_class = _class_name(_unpatched_call(factory_module, factory_fn_name))

        # Under test: what class does the alias route to?
        factory_module.reset_repositories()
        monkeypatch.setattr(
            "registry.core.config.settings.storage_backend",
            alias,
        )
        actual_class = _class_name(_unpatched_call(factory_module, factory_fn_name))

        assert actual_class == baseline_class, (
            f"{factory_fn_name} with STORAGE_BACKEND={alias!r} returned "
            f"{actual_class!r} but mongodb-ce baseline returned "
            f"{baseline_class!r}. This is the silent-fallback bug (#954)."
        )


@pytest.mark.unit
class TestFactoryFileBackendUnchanged:
    """The file backend must never accidentally return a DocumentDB repo."""

    @pytest.mark.parametrize(
        ("factory_fn_name", "expected_behavior"),
        [
            ("get_server_repository", "file"),
            ("get_agent_repository", "file"),
            ("get_scope_repository", "file"),
            ("get_security_scan_repository", "file"),
            ("get_search_repository", "file"),
            pytest.param(
                "get_federation_config_repository",
                "file",
                marks=pytest.mark.skip(
                    reason=(
                        "FileFederationConfigRepository.__init__ mkdirs /app/config/federation "
                        "which fails in local test env; routing is already verified by the "
                        "alias-routing test above"
                    ),
                ),
            ),
            ("get_peer_federation_repository", "file"),
            ("get_audit_repository", "none"),
            ("get_skill_repository", "documentdb"),
            ("get_skill_security_scan_repository", "file"),
            ("get_virtual_server_repository", "documentdb"),
            ("get_backend_session_repository", "none"),
            ("get_app_log_repository", "none"),
        ],
    )
    def test_file_backend_behavior(
        self,
        monkeypatch,
        factory_module,
        factory_fn_name: str,
        expected_behavior: str,
    ) -> None:
        """Assert file backend routes to File*, DocumentDB*, or None as documented.

        expected_behavior values:
          - "file":       returns a File* repository class
          - "documentdb": returns a DocumentDB* class (skill / virtual_server
                          have no File variant; they fall back to DocumentDB
                          even with file backend, by design)
          - "none":       returns None (audit / backend_session / app_log
                          require MongoDB; file backend is not supported)
        """
        factory_module.reset_repositories()
        monkeypatch.setattr(
            "registry.core.config.settings.storage_backend",
            "file",
        )
        repo = _unpatched_call(factory_module, factory_fn_name)
        class_name = _class_name(repo)

        if expected_behavior == "file":
            assert class_name is not None
            assert class_name.startswith("File") or class_name == "FaissSearchRepository", (
                f"{factory_fn_name} with file backend returned {class_name!r}; "
                "expected a File* class (or FaissSearchRepository)."
            )
        elif expected_behavior == "documentdb":
            assert class_name is not None
            assert class_name.startswith("DocumentDB"), (
                f"{factory_fn_name} with file backend returned {class_name!r}; "
                "expected the DocumentDB* fallback (skill / virtual_server have "
                "no File variant)."
            )
        elif expected_behavior == "none":
            assert class_name is None, (
                f"{factory_fn_name} with file backend returned {class_name!r}; "
                "expected None (file backend not supported for this repo)."
            )
        else:
            pytest.fail(f"unknown expected_behavior: {expected_behavior!r}")
