"""
Verify that both rotation Lambdas exclude the correct set of characters.

Catches regressions like unescaped quotes (SyntaxError at import time)
and mismatched EXCLUDE_CHARACTERS between Lambda defaults and Terraform env.

See: https://github.com/agentic-community/mcp-gateway-registry/issues/1354
"""

from pathlib import Path

# Required characters that must appear in the exclusion set.
# Any password containing these will break URI parsing, RDS auth, or
# DocumentDB connection strings.
REQUIRED_EXCLUDED = set("/@\"'+:?#&!=% ")

# Paths to Lambda source directories (relative to repo root)
_RDS_DIR = Path(__file__).resolve().parent.parent / "terraform/aws-ecs/lambda/rotate-rds"
_DOCDB_DIR = Path(__file__).resolve().parent.parent / "terraform/aws-ecs/lambda/rotate-documentdb"

# Terraform file that sets EXCLUDE_CHARACTERS in each Lambda's env block. The
# env value is what actually runs in AWS; the Lambda default is only a fallback
# when the env var is unset, so the two must stay byte-identical.
_ROTATION_TF = Path(__file__).resolve().parent.parent / "terraform/aws-ecs/secret-rotation.tf"


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

    def test_rds_default_contains_required_chars(self):
        """The RDS Lambda default exclusion set must include every required char."""
        source = (_RDS_DIR / "index.py").read_text()
        for ch in REQUIRED_EXCLUDED:
            assert ch in _extract_default(source), (
                f"RDS Lambda default is missing required char {ch!r}"
            )

    def test_documentdb_default_contains_required_chars(self):
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

    def test_terraform_env_matches_lambda_defaults(self):
        """Terraform env EXCLUDE_CHARACTERS must match the Lambda defaults.

        The env value in secret-rotation.tf is what runs in AWS; the Lambda
        default only applies when the env var is unset. If they drift, prod
        excludes a different set than the tested default. Both .tf occurrences
        (documentdb + rds rotation) must equal the Lambda default.
        """
        lambda_default = _extract_default((_RDS_DIR / "index.py").read_text())
        tf_values = _extract_tf_exclude_values(_ROTATION_TF.read_text())
        assert tf_values, "Could not find EXCLUDE_CHARACTERS in secret-rotation.tf"
        for value in tf_values:
            assert value == lambda_default, (
                f"secret-rotation.tf EXCLUDE_CHARACTERS {value!r} "
                f"!= Lambda default {lambda_default!r}"
            )


def _extract_tf_exclude_values(source: str) -> list[str]:
    """Extract every EXCLUDE_CHARACTERS value from a Terraform source string.

    Returns the raw literal body (with ``\\"`` left escaped). Both HCL and the
    Python Lambda source escape a double quote as ``\\"`` inside a double-quoted
    string, and :func:`_extract_default` also returns the raw body, so the two
    are directly comparable without unescaping either side.
    """
    import re

    return re.findall(
        r'EXCLUDE_CHARACTERS\s*=\s*"((?:[^"\\]|\\.)*)"',
        source,
    )


def _extract_default(source: str) -> str:
    """Extract the default value from os.environ.get('EXCLUDE_CHARACTERS', ...).

    Handles escaped quotes inside the string literal (e.g. \").
    """
    import re

    match = re.search(
        r'''os\.environ\.get\(\s*["']EXCLUDE_CHARACTERS["']\s*,\s*(["'])((?:[^\\]|\\.)*?)\1''',
        source,
    )
    assert match, "Could not find EXCLUDE_CHARACTERS default in source"
    return match.group(2)
