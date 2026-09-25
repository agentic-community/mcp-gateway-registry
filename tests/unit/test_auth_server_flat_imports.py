"""The auth-server container flattens ``auth_server/`` into ``/app``.

``docker/Dockerfile.auth`` copies the package contents, not the package::

    COPY auth_server/ /app/            # flattened: no auth_server package
    COPY registry/    /app/registry/   # registry stays importable as a package

So inside that container ``from observability.meters import x`` resolves and
``from auth_server.observability.meters import x`` raises ModuleNotFoundError.
Every module under ``auth_server/`` therefore has to reach its siblings by the
flat name, or try the flat name first and fall back to the packaged one.

Nothing else catches this. Run from the repo root, which is how pytest and CI
run, ``auth_server`` IS an importable package, so the packaged form resolves and
all of the suite passes. The failure appears only in the built container, at
request time, on whichever line happens to run first.

That is not hypothetical. #1718 added two such imports on live request paths and
every one of the 8815 tests still passed. ``_vend_egress_token`` returned HTTP
500 for every egress vend through mcp-proxy, and the OBO egress path did the
same, until the release smoke test caught the first one by hand.
"""

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AUTH_SERVER = REPO_ROOT / "auth_server"

# Handlers that make a packaged import a legitimate fallback rather than a bug.
_FALLBACK_EXCEPTIONS = {"ImportError", "ModuleNotFoundError"}


def _is_vendored(path: pathlib.Path) -> bool:
    """Whether a path sits inside vendored dependencies rather than our source."""
    return (
        ".venv" in path.parts
        or "site-packages" in path.parts
        or "node_modules" in path.parts
        or "__pycache__" in path.parts
    )


def _handler_catches_import_error(handler: ast.ExceptHandler) -> bool:
    """Whether an ``except`` clause catches ImportError or ModuleNotFoundError."""
    node = handler.type
    if node is None:
        # A bare ``except:`` swallows everything, ImportError included.
        return True
    candidates = node.elts if isinstance(node, ast.Tuple) else [node]
    for candidate in candidates:
        if isinstance(candidate, ast.Name) and candidate.id in _FALLBACK_EXCEPTIONS:
            return True
        if isinstance(candidate, ast.Attribute) and candidate.attr in _FALLBACK_EXCEPTIONS:
            return True
    return False


def _imports_auth_server_package(node: ast.AST) -> bool:
    """Whether this import statement reaches for the ``auth_server`` package."""
    if isinstance(node, ast.ImportFrom):
        # ``from . import x`` has module None; relative imports are not affected.
        module = node.module or ""
        return module == "auth_server" or module.startswith("auth_server.")
    if isinstance(node, ast.Import):
        return any(
            alias.name == "auth_server" or alias.name.startswith("auth_server.")
            for alias in node.names
        )
    return False


def _find_unguarded_imports(tree: ast.AST) -> list[int]:
    """Line numbers of packaged auth_server imports with no ImportError fallback.

    Walks with explicit ancestry rather than ``ast.walk`` because whether an
    import is a legitimate fallback depends entirely on what encloses it.
    """
    offenders: list[int] = []

    def visit(node: ast.AST, in_fallback: bool) -> None:
        if _imports_auth_server_package(node) and not in_fallback:
            offenders.append(node.lineno)  # type: ignore[attr-defined]
            return
        for child in ast.iter_child_nodes(node):
            # Only the handler body is a fallback context. A ``try:`` guarding a
            # network call is not: #1718's two bugs both sat in the try block of
            # a try/except that caught httpx errors, not ImportError.
            child_in_fallback = in_fallback
            if isinstance(node, ast.ExceptHandler):
                child_in_fallback = in_fallback or _handler_catches_import_error(node)
            visit(child, child_in_fallback)

    visit(tree, False)
    return offenders


def _auth_server_modules() -> list[pathlib.Path]:
    return sorted(p for p in AUTH_SERVER.rglob("*.py") if p.is_file() and not _is_vendored(p))


@pytest.mark.unit
class TestAuthServerImportsSurviveTheFlatLayout:
    """A packaged auth_server import with no flat fallback is a 500 in production."""

    def test_the_tree_is_actually_being_scanned(self):
        """Guard the guard: an empty or mis-rooted walk would pass everything."""
        modules = _auth_server_modules()
        assert len(modules) > 10, f"expected the auth_server tree, walked {len(modules)} files"
        assert (AUTH_SERVER / "server.py") in modules

    def test_no_unguarded_packaged_imports(self):
        """Every ``from auth_server.x import y`` needs a flat-first fallback.

        If this fails, the named line raises ModuleNotFoundError in the
        auth-server container while passing every test here. Use the pattern the
        codebase already applies in server.py, egress_obo.py, auth_path_stats.py
        and metrics_middleware.py::

            try:
                from observability.meters import thing
            except ImportError:
                from auth_server.observability.meters import thing
        """
        offenders = []
        for path in _auth_server_modules():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:  # pragma: no cover - a broken file fails elsewhere
                continue
            for lineno in _find_unguarded_imports(tree):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}")

        assert not offenders, (
            "These import the auth_server package with no flat fallback, so they raise "
            "ModuleNotFoundError in the container (Dockerfile.auth copies "
            "auth_server/ to /app/, so the package does not exist there):\n  "
            + "\n  ".join(offenders)
        )


@pytest.mark.unit
class TestTheDetectorItself:
    """The detector is the only thing standing between us and a repeat of #1718."""

    @staticmethod
    def _offenders(source: str) -> list[int]:
        return _find_unguarded_imports(ast.parse(source))

    def test_flags_a_bare_module_level_import(self):
        assert self._offenders("from auth_server.observability.meters import x") == [1]

    def test_flags_a_bare_function_level_import(self):
        source = "def f():\n    from auth_server.observability.meters import x\n"
        assert self._offenders(source) == [2]

    def test_flags_import_auth_server(self):
        assert self._offenders("import auth_server.egress_obo") == [1]

    def test_allows_the_flat_first_fallback(self):
        source = (
            "try:\n"
            "    from observability.meters import x\n"
            "except ImportError:\n"
            "    from auth_server.observability.meters import x\n"
        )
        assert self._offenders(source) == []

    def test_allows_a_tuple_handler(self):
        source = (
            "try:\n"
            "    from observability.meters import x\n"
            "except (ImportError, AttributeError):\n"
            "    from auth_server.observability.meters import x\n"
        )
        assert self._offenders(source) == []

    def test_flags_an_import_in_a_try_that_catches_something_else(self):
        """#1718's exact shape: inside a try, but the handler catches httpx errors.

        This is the case a simpler "is there a try nearby" check would miss, and
        it is the one that actually shipped.
        """
        source = (
            "try:\n"
            "    from auth_server.observability.meters import x\n"
            "    post()\n"
            "except httpx.RequestError:\n"
            "    pass\n"
        )
        assert self._offenders(source) == [2]

    def test_ignores_relative_imports(self):
        assert self._offenders("from .observability.meters import x") == []

    def test_ignores_registry_imports(self):
        """``registry`` stays a package in the container, so those are fine."""
        assert self._offenders("from registry.utils.url_guard import guarded_client") == []
