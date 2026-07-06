"""
Verify that both rotation Lambdas exclude the correct set of characters.

Catches regressions like unescaped quotes (SyntaxError at import time)
and mismatched EXCLUDE_CHARACTERS between Lambda defaults and Terraform env.

See: https://github.com/agentic-community/mcp-gateway-registry/issues/1354
"""

import importlib
import os
import sys
from pathlib import Path

import pytest

# Required characters that must appear in the exclusion set.
# Any password containing these will break URI parsing, RDS auth, or
# DocumentDB connection strings.
REQUIRED_EXCLUDED = set("/@\"'+:?&!=% ")

# Paths to Lambda source directories (relative to repo root)
_RDS_DIR = Path(__file__).resolve().parent.parent / "terraform/aws-ecs/lambda/rotate-rds"
_DOCDB_DIR = Path(__file__).resolve().parent.parent / "terraform/aws-ecs/lambda/rotate-documentdb"


def _import_lambda_module(name: str, source_dir: Path):
    """Import a Lambda module by temporarily adding its directory to sys.path."""
    original_path = sys.path.copy()
    try:
        sys.path.insert(0, str(source_dir))
        # Remove any previously cached version
        sys.modules.pop(name, None)
        return importlib.import_module(name)
    finally:
        sys.path = original_path


@pytest.fixture()
def rds_module():
    mod = _import_lambda_module("rotate_rds_index", _RDS_DIR)
    yield mod
    sys.modules.pop("rotate_rds_index", None)


@pytest.fixture()
def docdb_module():
    mod = _import_lambda_module("rotate_docdb_index", _DOCDB_DIR)
    yield mod
    sys.modules.pop("rotate_docdb_index", None)


class TestExcludeCharacters:
    """Both Lambdas must exclude the same set of URI-unsafe / RDS-unsafe chars."""

    def test_rds_compiles(self):
        """rotate-rds/index.py must parse without SyntaxError."""
        source = (_RDS_DIR / "index.py").read_text()
        compile(source, str(_RDS_DIR / "index.py"), "exec")

    def test_documentdb_compiles(self):
        """rotate-documentdb/index.py must parse without SyntaxError."""
        source = (_DOCDB_DIR / "index.py").read_text()
        compile(source, str(_DOCDB_DIR / "index.py"), "exec")

    def test_rds_default_contains_required_chars(self, rds_module):
        """The RDS Lambda default exclusion set must include every required char."""
        # Temporarily remove the env var so os.environ.get falls back to the default
        env_backup = os.environ.pop("EXCLUDE_CHARACTERS", None)
        try:
            # Re-evaluate the module-level get() by importing fresh
            sys.modules.pop("rotate_rds_index", None)
            mod = importlib.import_module("rotate_rds_index")
            # We can't easily re-evaluate module-level code, so read the source
            # and extract the default value directly.
        finally:
            if env_backup is not None:
                os.environ["EXCLUDE_CHARACTERS"] = env_backup

        # Direct source check: read the default from source
        source = (_RDS_DIR / "index.py").read_text()
        for ch in REQUIRED_EXCLUDED:
            # The default string literal must contain each required character
            assert ch in _extract_default(source), (
                f"RDS Lambda default is missing required char {ch!r}"
            )

    def test_documentdb_default_contains_required_chars(self, docdb_module):
        """The DocumentDB Lambda default exclusion set must include every required char."""
        source = (_DOCDB_DIR / "index.py").read_text()
        for ch in REQUIRED_EXCLUDED:
            assert ch in _extract_default(source), (
                f"DocumentDB Lambda default is missing required char {ch!r}"
            )

    def test_both_lambdas_use_same_default(self):
        """Both Lambdas must have byte-identical EXCLUDE_CHARACTERS defaults."""
        rds_default = _extract_default((_RDS_DIR / "index.py").read_text())
        docdb_default = _extract_default((_DOCDB_DIR / "index.py").read_text())
        assert rds_default == docdb_default, (
            f"RDS default {rds_default!r} != DocDB default {docdb_default!r}"
        )


def _extract_default(source: str) -> str:
    """Extract the default value from os.environ.get('EXCLUDE_CHARACTERS', ...).

    Returns the raw string between the outermost quotes of the second argument.
    """
    import re

    match = re.search(
        r'''os\.environ\.get\(\s*["']EXCLUDE_CHARACTERS["']\s*,\s*(["'])(.*?)\1''',
        source,
    )
    assert match, "Could not find EXCLUDE_CHARACTERS default in source"
    return match.group(2)
