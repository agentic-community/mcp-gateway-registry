"""Compose parity check for the OpenBao AppRole credentials.

``registry/secrets/factory.py`` supports ``OPENBAO_AUTH_METHOD=approle`` and reads
``OPENBAO_ROLE_ID`` and ``OPENBAO_SECRET_ID`` from the process environment. The
Docker Compose files forwarded every other ``OPENBAO_*`` variable to the registry
service but not those two, so the approle branch could never succeed on the
Compose path: the registry started healthy and every vault request failed with
``OPENBAO_AUTH_METHOD=approle requires OPENBAO_ROLE_ID and OPENBAO_SECRET_ID``
(issue #1771). The only working alternative was a static root token.

This asserts that every variable the factory reads for OpenBao reaches the
registry in every compose file, derived from the factory source rather than from
a hand-maintained list, so a future auth method cannot repeat the gap silently.
"""

import re
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.unit]

COMPOSE_FILES = [
    "docker-compose.yml",
    "docker-compose.prebuilt.yml",
    "docker-compose.podman.yml",
]

# The only service that constructs the secret store (auth-server does not).
VAULT_CONSUMER = "registry"

# The two credentials the approle branch needs together.
APPROLE_VARS = ["OPENBAO_ROLE_ID", "OPENBAO_SECRET_ID"]


@pytest.fixture(scope="module")
def repo_root() -> Path:
    """Repository root directory."""
    return Path(__file__).parent.parent.parent


def _service_environment(compose_path: Path, service: str) -> dict[str, str]:
    """Return a service's environment as a mapping.

    Compose accepts both the ``KEY: value`` mapping form and the ``- KEY=value``
    list form; normalise to a dict so assertions work against either.
    """
    data = yaml.safe_load(compose_path.read_text())
    env = data["services"][service].get("environment", {}) or {}
    if isinstance(env, dict):
        return {str(k): str(v) for k, v in env.items()}
    normalised: dict[str, str] = {}
    for entry in env:
        key, _, value = str(entry).partition("=")
        normalised[key] = value
    return normalised


def _openbao_vars_read_by_factory(repo_root: Path) -> set[str]:
    """Every ``OPENBAO_*`` name the secret-store factory reads from the environment.

    Taken from the source so the test tracks the code, not a copy of it. Covers
    both ``os.environ.get("OPENBAO_X")`` and the ``settings.openbao_x`` fields
    that pydantic-settings populates from ``OPENBAO_X``.
    """
    source = (repo_root / "registry" / "secrets" / "factory.py").read_text()
    from_environ = set(re.findall(r'os\.environ\.get\(\s*"(OPENBAO_[A-Z_]+)"', source))
    from_settings = {
        f"OPENBAO_{name.upper()}" for name in re.findall(r"settings\.openbao_([a-z_]+)", source)
    }
    return from_environ | from_settings


@pytest.mark.parametrize("compose_filename", COMPOSE_FILES)
@pytest.mark.parametrize("var", APPROLE_VARS)
def test_registry_receives_approle_credentials(
    repo_root: Path,
    compose_filename: str,
    var: str,
):
    """Both AppRole credentials must reach the registry in every compose file.

    Without them the approle branch raises on every vault request, so
    ``OPENBAO_AUTH_METHOD=approle`` is unusable and operators are pushed onto a
    static root token instead of a scoped AppRole policy.
    """
    env = _service_environment(repo_root / compose_filename, VAULT_CONSUMER)

    assert var in env, (
        f"{compose_filename}: the '{VAULT_CONSUMER}' service does not receive {var}, so "
        f"OPENBAO_AUTH_METHOD=approle fails with "
        f"'requires OPENBAO_ROLE_ID and OPENBAO_SECRET_ID' on every vault request."
    )


@pytest.mark.parametrize("compose_filename", COMPOSE_FILES)
@pytest.mark.parametrize("var", APPROLE_VARS)
def test_approle_credentials_interpolate_from_operator_env(
    repo_root: Path,
    compose_filename: str,
    var: str,
):
    """The value must interpolate ``${VAR:-}`` from ``.env``, with an empty default.

    A hard-coded value would ignore the operator's setting; a required default
    (``:?``) would break the ``token`` and ``kubernetes`` paths, which do not
    need these two at all.
    """
    env = _service_environment(repo_root / compose_filename, VAULT_CONSUMER)
    value = env[var]

    assert value == f"${{{var}:-}}", (
        f"{compose_filename}: {var} is set to '{value}' instead of '${{{var}:-}}'. It must "
        f"interpolate from .env and default to empty so non-approle auth methods are unaffected."
    )


@pytest.mark.parametrize("compose_filename", COMPOSE_FILES)
def test_every_openbao_var_the_factory_reads_is_forwarded(
    repo_root: Path,
    compose_filename: str,
):
    """Generalisation of the above: no ``OPENBAO_*`` the factory reads may be missing.

    ``OPENBAO_ROLE_ID`` / ``OPENBAO_SECRET_ID`` went missing because the compose
    list was maintained by hand next to a factory that grew a new auth method.
    Deriving the expected set from the factory source closes that gap for the
    next auth method too.
    """
    expected = _openbao_vars_read_by_factory(repo_root)
    assert expected, "no OPENBAO_* reads found in factory.py -- has the file moved?"

    env = _service_environment(repo_root / compose_filename, VAULT_CONSUMER)
    missing = sorted(v for v in expected if v not in env)

    assert not missing, (
        f"{compose_filename}: registry/secrets/factory.py reads {missing} but the "
        f"'{VAULT_CONSUMER}' service does not forward them."
    )


def test_env_example_documents_approle_credentials(repo_root: Path):
    """``.env.example`` is where operators discover variables; both must be listed."""
    content = (repo_root / ".env.example").read_text()

    for var in APPROLE_VARS:
        assert re.search(rf"^{var}=", content, re.MULTILINE), (
            f".env.example does not list {var}; an operator has no way to discover it "
            f"short of reading registry/secrets/factory.py."
        )


@pytest.mark.parametrize(
    "doc_path",
    ["docs/egress-credential-vault.md", "docs/unified-parameter-reference.md"],
)
def test_docs_variable_tables_list_approle_credentials(repo_root: Path, doc_path: str):
    """Both parameter tables must carry the two variables and mention ``approle``."""
    content = (repo_root / doc_path).read_text()

    for var in APPROLE_VARS:
        assert f"`{var}`" in content, f"{doc_path}: variable table does not list {var}."
    assert "approle" in content, f"{doc_path}: never mentions the approle auth method."


def test_docs_do_not_claim_openbao_role_is_required_for_approle(repo_root: Path):
    """The approle branch never reads ``OPENBAO_ROLE``; the docs must not say it does.

    The vault doc used to state ``OPENBAO_ROLE`` was "required for
    kubernetes/approle auth", which sent operators to set a variable the approle
    path ignores while the two it actually needs stayed undocumented.
    """
    content = (repo_root / "docs" / "egress-credential-vault.md").read_text()
    role_rows = [line for line in content.splitlines() if line.startswith("| `OPENBAO_ROLE`")]

    assert role_rows, "OPENBAO_ROLE row missing from docs/egress-credential-vault.md"
    # The defect was the literal claim "required for `kubernetes`/`approle`". The
    # row may still *mention* approle to say the opposite, so match the claim,
    # not the word.
    wrong_claim = re.compile(r"required for\s+`kubernetes`\s*/\s*`approle`", re.IGNORECASE)
    for row in role_rows:
        assert not wrong_claim.search(row), (
            f"docs/egress-credential-vault.md still presents OPENBAO_ROLE as an approle "
            f"requirement: {row.strip()}"
        )
        assert "kubernetes" in row, (
            f"OPENBAO_ROLE row no longer names kubernetes auth: {row.strip()}"
        )
