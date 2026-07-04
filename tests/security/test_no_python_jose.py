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


def test_python_jose_not_declared_in_auth_server(repo_root: Path):
    """auth_server must not declare python-jose as a dependency."""
    content = (repo_root / "auth_server" / "pyproject.toml").read_text()
    assert "python-jose" not in content, (
        "auth_server/pyproject.toml declares python-jose; use pyjwt instead (SA-13)."
    )


def test_python_jose_absent_from_auth_server_lock(repo_root: Path):
    """The resolved auth_server lock must not contain python-jose or ecdsa."""
    lock = (repo_root / "auth_server" / "uv.lock").read_text()
    assert 'name = "python-jose"' not in lock, (
        "auth_server/uv.lock resolves python-jose; re-run `uv lock` after removing it (SA-13)."
    )
    assert 'name = "ecdsa"' not in lock, (
        "auth_server/uv.lock resolves ecdsa (pulled by python-jose); it has an unfixed CVE (SA-13)."
    )


def test_no_jose_imports_in_source(repo_root: Path):
    """No source file should import python-jose (the `jose` package)."""
    offenders: list[str] = []
    for base in ("auth_server", "registry", "cli"):
        base_dir = repo_root / base
        if not base_dir.is_dir():
            continue
        for py in base_dir.rglob("*.py"):
            text = py.read_text(errors="ignore")
            if "from jose" in text or "import jose" in text:
                offenders.append(str(py.relative_to(repo_root)))
    assert not offenders, f"python-jose (`jose`) imported in: {offenders} (SA-13)"
