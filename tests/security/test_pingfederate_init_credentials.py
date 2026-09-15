"""
Regression tests for IdP bootstrap credentials: the PingFederate init script must
never seed hardcoded/default credentials, must require every credential from the
environment (fail closed when unset, weak, padded-weak, or whitespace-only), must
not relax password requirements, must rotate a pre-existing weak credential
instead of skipping it, and must not map a hardcoded demo user to a privileged
group.

The static assertions are content/ordering checks over the script; the
behavioral tests run the script in a temp sandbox (so it never sources a real
``.env`` and never reaches the PingFederate/Mongo calls) and assert it exits
non-zero and seeds nothing when a required credential is unset or weak.
"""

import os
import re
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

SCRIPT_REL = "pingfederate/setup/init-pingfederate.sh"

# A full set of strong credentials; individual tests override one entry to
# exercise the fail-closed paths.
STRONG_CREDS = {
    "PF_ADMIN_PASS": "S7rong-Pf-Admin-Pass",
    "PINGFEDERATE_CLIENT_SECRET": "S7rong-Client-Secret",
    "PF_REGISTRY_ADMIN_PASSWORD": "S7rong-Registry-Admin",
}

# Every step marker. None may appear when the script fails closed: validation runs
# before the banner, so even step 1 must not be reached. Covering [3/8] and [4/8]
# matters because those are real admin-API writes (PUT /serverSettings and PUT
# /oauth/authServerSettings) authenticated with PF_ADMIN_PASS.
SEED_MARKERS = (
    "[1/8]",
    "[2/8]",
    "[3/8]",
    "[4/8]",
    "[5/8]",
    "[6/8]",
    "[7/8]",
    "[8/8]",
    "idp_user_groups",
    "registry-admins",
)


@pytest.fixture(scope="module")
def repo_root() -> Path:
    """Get repository root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="module")
def script_text(repo_root: Path) -> str:
    return (repo_root / SCRIPT_REL).read_text()


# Files the script needs at their repo-relative paths. The shared denylist is a
# hard dependency: the script fails closed if it is absent rather than validating
# against an empty list.
SUPPORT_FILES = ("scripts/weak-credentials.sh",)


def _run_sandboxed(repo_root: Path, tmp_path: Path, env: dict[str, str]):
    """Copy the script into a clean tree (no .env) and run it with ``env`` only."""
    dest = tmp_path / SCRIPT_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(repo_root / SCRIPT_REL, dest)
    for support in SUPPORT_FILES:
        target = tmp_path / support
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo_root / support, target)
    # Inherit PATH so bash/python3 resolve on Homebrew and Nix layouts too;
    # everything else is a fully controlled environment.
    run_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **env}
    return subprocess.run(
        ["bash", str(dest)],
        env=run_env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _assert_failed_closed(result, context: str):
    assert result.returncode != 0, f"{context}: expected non-zero exit.\n{result.stderr}"
    combined = result.stdout + result.stderr
    assert not any(m in combined for m in SEED_MARKERS), (
        f"{context}: script reached a seeding step: {combined!r}"
    )


# --------------------------------------------------------------------------
# Static assertions
# --------------------------------------------------------------------------


def test_no_weak_credential_fallbacks(script_text: str):
    """No `${VAR:-weakdefault}` credential fallbacks remain in the script."""
    offenders = [
        pat
        for pat in (":-changeme", ":-2FederateM0re", ":-admin123", ":-2federatem0re")
        if pat in script_text
    ]
    assert not offenders, f"{SCRIPT_REL}: weak default-credential fallback(s) present: {offenders}."


def test_no_hardcoded_seed_passwords(script_text: str):
    """The hardcoded admin123/changeme demo passwords must not be seeded."""
    assert "'admin', 'admin123'" not in script_text, (
        f"{SCRIPT_REL}: still seeds the hardcoded ('admin', 'admin123') login."
    )
    assert "'testuser', 'changeme'" not in script_text, (
        f"{SCRIPT_REL}: still seeds the hardcoded ('testuser', 'changeme') login."
    )
    # The seeded password must come from the environment, never a literal.
    assert "os.environ['PF_SEED_ADMIN_PW']" in script_text, (
        f"{SCRIPT_REL}: seeded admin password must be read from the environment."
    )


def test_client_secret_not_interpolated_into_json(script_text: str):
    """The OAuth client secret must be serialized from env, not spliced into JSON.

    Raw interpolation lets a JSON-escaped value decode server-side to a weak
    secret that bypassed validation, and breaks on quotes/backslashes.
    """
    assert '\\"secret\\": \\"${PF_CLIENT_SECRET}\\"' not in script_text, (
        f"{SCRIPT_REL}: client secret is interpolated into JSON source."
    )
    assert "os.environ['PF_CLIENT_SECRET']" in script_text, (
        f"{SCRIPT_REL}: client payload must read the secret from the environment."
    )


def test_no_relaxed_password_requirements(script_text: str):
    """The seeded credential must not relax PingFederate's password policy."""
    assert "Relax Password Requirements" not in script_text, (
        f"{SCRIPT_REL}: still sets 'Relax Password Requirements'; the seeded "
        "credential must satisfy the password policy."
    )


def test_no_hardcoded_testuser_group_seed(script_text: str):
    """No hardcoded demo user is seeded into idp_user_groups (a purge is fine)."""
    assert "username: 'testuser', groups" not in script_text, (
        f"{SCRIPT_REL}: still seeds a hardcoded 'testuser' into idp_user_groups."
    )
    assert "public-mcp-users" not in script_text, (
        f"{SCRIPT_REL}: still maps a demo user to a group."
    )
    # Re-running must actively purge any legacy testuser mapping it once seeded.
    assert "deleteOne({username: 'testuser'" in script_text, (
        f"{SCRIPT_REL}: does not purge a legacy 'testuser' idp_user_groups mapping."
    )


def test_requires_all_credentials(script_text: str):
    """Every credential is validated up front via the fail-closed validator."""
    for var in STRONG_CREDS:
        assert f'require_strong_secret "{var}"' in script_text, (
            f"{SCRIPT_REL}: {var} is not validated by require_strong_secret."
        )


def test_credentials_validated_before_any_seeding(script_text: str):
    """Validation must precede EVERY step, including the admin-API writes.

    Steps 3 and 4 (`PUT /serverSettings`, `PUT /oauth/authServerSettings`) are
    real writes authenticated with `PF_ADMIN_PASS`, so an ordering guard that
    only covers the seeding steps would stay green while the script mutated
    server configuration using an unvalidated credential.
    """
    last_validation = script_text.rfind('require_strong_secret "')
    assert last_validation != -1
    for marker in SEED_MARKERS:
        pos = script_text.find(marker)
        assert pos != -1, f"marker {marker!r} missing"
        assert last_validation < pos, (
            f"{SCRIPT_REL}: credential validation must precede {marker!r}."
        )


def test_admin_credential_is_rotated_not_skipped(script_text: str):
    """A pre-existing admin row is overwritten (rotated), never left in place."""
    assert "if 'admin' not in existing" not in script_text, (
        f"{SCRIPT_REL}: the skip-if-exists idiom leaves a legacy weak credential "
        "in place; the admin credential must be rotated on every run."
    )
    assert "_username(row) == 'admin'" in script_text, (
        f"{SCRIPT_REL}: expected an upsert that overwrites an existing admin row."
    )


def test_legacy_testuser_row_is_removed(script_text: str):
    """Re-running the script must strip any legacy demo 'testuser' PCV row."""
    assert "_username(r) != 'testuser'" in script_text, (
        f"{SCRIPT_REL}: does not remove a legacy 'testuser' row from the PCV."
    )


# The credentials init-pingfederate.sh requires with no default. Every one of
# them must appear in the onboarding template with an EMPTY value: a template
# that ships a denylisted literal (e.g. the historical
# `PINGFEDERATE_CLIENT_SECRET=changeme`) turns `cp .env.example .env` into a
# guaranteed bootstrap failure, and a variable missing from the template leaves
# operators to discover a hard requirement by hitting it.
REQUIRED_ENV_TEMPLATE_KEYS = [
    "PF_ADMIN_PASS",
    "PINGFEDERATE_CLIENT_SECRET",
    "PF_REGISTRY_ADMIN_PASSWORD",
]


@pytest.mark.parametrize("key", REQUIRED_ENV_TEMPLATE_KEYS)
def test_env_template_ships_required_credential_blank(repo_root: Path, key: str):
    """`.env.example` must declare each required credential, with no value."""
    lines = (repo_root / ".env.example").read_text().splitlines()
    assignments = [line for line in lines if line.startswith(f"{key}=")]
    assert assignments, (
        f".env.example does not declare {key}, which {SCRIPT_REL} requires with no default."
    )
    for line in assignments:
        value = line.split("=", 1)[1].strip()
        assert not value, (
            f".env.example ships {line!r}; a shipped value for {key} is a usable "
            "credential and would be rejected by the bootstrap denylist anyway."
        )


COMPOSE_FILES_WITH_PINGFEDERATE = [
    "docker-compose.yml",
    "docker-compose.prebuilt.yml",
    "docker-compose.podman.yml",
]


@pytest.mark.parametrize("compose_file", COMPOSE_FILES_WITH_PINGFEDERATE)
def test_compose_bootstraps_pf_admin_from_operator_password(
    repo_root: Path,
    compose_file: str,
):
    """The bundled container's console admin must come from `PF_ADMIN_PASS`.

    The Ping server profile resolves the administrator password from
    ``${PING_IDENTITY_PASSWORD:=2FederateM0re}``. Without an explicit
    ``PING_IDENTITY_PASSWORD``, the console boots with the vendor default -- the
    exact literal the bootstrap denylists -- so a correct ``PF_ADMIN_PASS`` is
    rejected as weak while a strong one 401s. Passing it through keeps the
    container and the validated variable in sync.
    """
    content = (repo_root / compose_file).read_text()
    assert "PING_IDENTITY_PASSWORD=${PF_ADMIN_PASS:-}" in content, (
        f"{compose_file}: the pingfederate service must pass "
        "PING_IDENTITY_PASSWORD=${PF_ADMIN_PASS:-} so the console administrator "
        "is bootstrapped from the operator-supplied password, not the vendor "
        "default."
    )
    assert "PING_IDENTITY_PASSWORD=${PF_ADMIN_PASS:-2" not in content, (
        f"{compose_file}: PING_IDENTITY_PASSWORD must not carry a fallback value."
    )


# --------------------------------------------------------------------------
# Behavioral (fail-closed) assertions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("missing", sorted(STRONG_CREDS))
def test_fails_closed_when_credential_unset(repo_root: Path, tmp_path: Path, missing: str):
    """An unset required credential aborts the run before anything is seeded."""
    env = {k: v for k, v in STRONG_CREDS.items() if k != missing}
    result = _run_sandboxed(repo_root, tmp_path, env)
    assert missing in result.stderr, f"error must name {missing}: {result.stderr!r}"
    _assert_failed_closed(result, f"unset {missing}")


@pytest.mark.parametrize("var", sorted(STRONG_CREDS))
@pytest.mark.parametrize("weak", ["changeme", "admin123", "2FederateM0re", "password"])
def test_fails_closed_on_weak_credential(repo_root: Path, tmp_path: Path, var: str, weak: str):
    """A known-weak value in ANY credential aborts before anything is seeded."""
    env = {**STRONG_CREDS, var: weak}
    result = _run_sandboxed(repo_root, tmp_path, env)
    _assert_failed_closed(result, f"weak {var}={weak!r}")


@pytest.mark.parametrize("var", sorted(STRONG_CREDS))
def test_fails_closed_on_padded_weak_credential(repo_root: Path, tmp_path: Path, var: str):
    """A padded weak value (e.g. '  changeme  ') must not bypass the denylist."""
    env = {**STRONG_CREDS, var: "  changeme  "}
    result = _run_sandboxed(repo_root, tmp_path, env)
    _assert_failed_closed(result, f"padded weak {var}")


@pytest.mark.parametrize("var", sorted(STRONG_CREDS))
def test_fails_closed_on_whitespace_only_credential(repo_root: Path, tmp_path: Path, var: str):
    """A whitespace-only value (even if long) is treated as unset."""
    env = {**STRONG_CREDS, var: " \t \t \t \t \t"}
    result = _run_sandboxed(repo_root, tmp_path, env)
    _assert_failed_closed(result, f"whitespace-only {var}")


@pytest.mark.parametrize("var", sorted(STRONG_CREDS))
def test_fails_closed_on_short_credential(repo_root: Path, tmp_path: Path, var: str):
    """A too-short credential is rejected (length floor)."""
    env = {**STRONG_CREDS, var: "Ab1!"}
    result = _run_sandboxed(repo_root, tmp_path, env)
    assert "at least" in result.stderr, result.stderr
    _assert_failed_closed(result, f"short {var}")


class _ErrorHandler(BaseHTTPRequestHandler):
    """Mock PingFederate admin API that always returns HTTP 400."""

    def _respond(self):
        self.send_response(400)
        self.end_headers()
        self.wfile.write(b"{}")

    do_GET = _respond
    do_PUT = _respond
    do_POST = _respond

    def log_message(self, *args):  # noqa: D401 - silence the test server
        pass


def _extract_pf_api(script_text: str) -> str:
    match = re.search(r"^pf_api\(\) \{.*?^\}", script_text, re.S | re.M)
    assert match, "pf_api function definition not found in the script"
    return match.group(0)


def test_api_write_fails_closed_on_http_error(script_text: str, tmp_path: Path):
    """A non-2xx response to an admin-API write aborts the run (no false success).

    This is the upgrade-remediation guarantee: if PingFederate rejects the
    rotated credential (e.g. password policy) or any write errors, the script
    must fail closed rather than report success while a legacy weak credential
    survives.
    """
    server = HTTPServer(("127.0.0.1", 0), _ErrorHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        harness = tmp_path / "pf_api_harness.sh"
        harness.write_text(
            "set -e\n"
            "PF_ADMIN_USER=u; PF_ADMIN_PASS=p\n"
            f'PF_ADMIN_URL="http://127.0.0.1:{port}"\n' + _extract_pf_api(script_text) + "\n"
            'pf_api PUT "/passwordCredentialValidators/simple" "{}" > /dev/null\n'
            "echo SEEDED-ANYWAY\n"
        )
        result = subprocess.run(["bash", str(harness)], capture_output=True, text=True, timeout=30)
    finally:
        server.shutdown()
    assert result.returncode != 0, "a failed PingFederate write must abort the script"
    assert "SEEDED-ANYWAY" not in result.stdout, (
        "script continued past a failed write; a legacy credential could survive."
    )
    assert "failed (HTTP 400)" in result.stderr, result.stderr
