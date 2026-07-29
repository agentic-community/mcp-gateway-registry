"""Guard against drift between the two label-cardinality limiter copies.

The in-process limiter (``registry/observability/label_bounding.py``) and the
metrics-service processor's limiter
(``metrics-service/app/core/processor.py``) are intentional siblings across a
hard deployable boundary: the metrics-service is a separate container that
cannot import ``registry``, so the bounding logic is duplicated rather than
shared. For the two emission paths to bound identically, their shared constants
(safe charset, sentinels, and the length/cardinality defaults) must stay equal.

This test parses both source files with ``ast`` -- it does NOT import the
metrics-service module, which pulls in dependencies (aiosqlite, ...) not present
in the registry test environment -- and asserts the shared constants match. If
either copy changes a constant without the other, this fails.
"""

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

# The constants that MUST be identical across both limiter copies for the two
# emission paths to bound labels the same way.
_SHARED_CONSTANTS: tuple[str, ...] = (
    "_MAX_LABEL_CARDINALITY",
    "_MAX_LABEL_LENGTH",
    "_OVERFLOW_LABEL_VALUE",
    "_EMPTY_LABEL_VALUE",
    "_SAFE_LABEL_CHARS",
)

_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
_REGISTRY_LIMITER: Path = _REPO_ROOT / "registry" / "observability" / "label_bounding.py"
_METRICS_LIMITER: Path = _REPO_ROOT / "metrics-service" / "app" / "core" / "processor.py"


def _extract_constants(
    source_path: Path,
) -> dict[str, str]:
    """Return a mapping of shared-constant name to a normalized source form.

    Parses the module with ``ast`` and, for each name in ``_SHARED_CONSTANTS``,
    records the unparsed source of its assigned expression. Using the expression
    source (rather than a runtime value) keeps the comparison independent of any
    environment overrides read at import time.

    Args:
        source_path: Path to the Python source file to parse.

    Returns:
        Mapping of constant name to the unparsed source of its value.
    """
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign | ast.Assign):
            continue
        targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
        for target in targets:
            if isinstance(target, ast.Name) and target.id in _SHARED_CONSTANTS:
                if node.value is not None:
                    found[target.id] = ast.unparse(node.value)
    return found


def test_both_limiter_source_files_exist() -> None:
    """Both sibling files must exist for the parity check to be meaningful."""
    assert _REGISTRY_LIMITER.is_file(), f"missing {_REGISTRY_LIMITER}"
    assert _METRICS_LIMITER.is_file(), f"missing {_METRICS_LIMITER}"


def test_shared_constants_are_in_sync() -> None:
    """The shared cardinality-limiter constants must match across both copies."""
    registry_constants = _extract_constants(_REGISTRY_LIMITER)
    metrics_constants = _extract_constants(_METRICS_LIMITER)

    # Every shared constant must be present in both files.
    for name in _SHARED_CONSTANTS:
        assert name in registry_constants, (
            f"{name} not found in {_REGISTRY_LIMITER.name}; the sibling parity "
            f"guard cannot verify it"
        )
        assert name in metrics_constants, (
            f"{name} not found in {_METRICS_LIMITER.name}; the sibling parity "
            f"guard cannot verify it"
        )

    # And their values must be identical.
    for name in _SHARED_CONSTANTS:
        assert registry_constants[name] == metrics_constants[name], (
            f"Sibling limiter drift on {name}: "
            f"{_REGISTRY_LIMITER.name} has {registry_constants[name]!r} but "
            f"{_METRICS_LIMITER.name} has {metrics_constants[name]!r}. "
            f"Keep the two limiter copies in sync (see the module docstrings)."
        )
