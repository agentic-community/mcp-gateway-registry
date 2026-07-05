"""
Regression test for SA-13: python-jose (and its transitive ecdsa) must not
return as a dependency.

python-jose < 3.4.0 is affected by CVE-2024-33663 (ECDSA algorithm confusion)
and CVE-2024-33664 (JWE DoS), and it drags in ecdsa, affected by the unfixed
CVE-2024-23342 (Minerva). The codebase uses pyjwt exclusively, so python-jose
was removed. These checks fail if it (or ecdsa) is reintroduced.
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def repo_root() -> Path:
    """Get repository root directory."""
    return Path(__file__).parent.parent.parent


# Every manifest that could reintroduce python-jose. Reintroduction via the root
# project (or any sub-project) must be caught, not just auth_server.
_MANIFESTS = (
    "auth_server/pyproject.toml",
    "pyproject.toml",
    "agents/a2a/pyproject.toml",
)

# Every lockfile that would resolve python-jose / its transitive ecdsa.
_LOCKFILES = (
    "auth_server/uv.lock",
    "uv.lock",
    "agents/a2a/uv.lock",
    "metrics-service/uv.lock",
    "servers/currenttime/uv.lock",
    "servers/mcpgw/uv.lock",
    "servers/realserverfaketools/uv.lock",
)


@pytest.mark.parametrize("manifest", _MANIFESTS)
def test_python_jose_not_declared(repo_root: Path, manifest: str):
    """No manifest may declare python-jose as a dependency."""
    path = repo_root / manifest
    if not path.exists():
        pytest.skip(f"{manifest} not present")
    content = path.read_text()
    assert "python-jose" not in content, (
        f"{manifest} declares python-jose; use pyjwt instead (SA-13)."
    )


@pytest.mark.parametrize("lockfile", _LOCKFILES)
def test_python_jose_absent_from_lock(repo_root: Path, lockfile: str):
    """No resolved lockfile may contain python-jose or ecdsa."""
    path = repo_root / lockfile
    if not path.exists():
        pytest.skip(f"{lockfile} not present")
    lock = path.read_text()
    assert 'name = "python-jose"' not in lock, (
        f"{lockfile} resolves python-jose; re-run `uv lock` after removing it (SA-13)."
    )
    assert 'name = "ecdsa"' not in lock, (
        f"{lockfile} resolves ecdsa (pulled by python-jose); it has an unfixed CVE (SA-13)."
    )


def test_no_jose_imports_in_source(repo_root: Path):
    """No source file should import python-jose (the `jose` package)."""
    offenders: list[str] = []
    for base in ("auth_server", "registry", "cli", "servers", "agents", "metrics-service", "scripts"):
        base_dir = repo_root / base
        if not base_dir.is_dir():
            continue
        for py in base_dir.rglob("*.py"):
            text = py.read_text(errors="ignore")
            if "from jose" in text or "import jose" in text:
                offenders.append(str(py.relative_to(repo_root)))
    assert not offenders, f"python-jose (`jose`) imported in: {offenders} (SA-13)"
