"""The auth server imports search_repository without numpy installed.

auth_server/pyproject.toml has no numpy, but the auth server imports
registry.repositories.factory, which imports the DocumentDB search repository.
A module-level import of the registry.embeddings package (which imports numpy)
made the auth server crash at startup, so this test imports the repository in a
fresh interpreter where numpy cannot be imported.
"""

import subprocess
import sys

import pytest

IMPORT_TIMEOUT_SECONDS: int = 60

BLOCK_NUMPY_AND_IMPORT: str = """
import sys
sys.modules["numpy"] = None
import registry.repositories.documentdb.search_repository
print("imported")
"""


@pytest.mark.unit
def test_search_repository_imports_without_numpy():
    result = subprocess.run(  # nosec B603 - fixed interpreter and test-authored code
        [sys.executable, "-c", BLOCK_NUMPY_AND_IMPORT],
        capture_output=True,
        text=True,
        timeout=IMPORT_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "imported" in result.stdout
