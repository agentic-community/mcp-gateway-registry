"""
Regression tests for SA-9: the Keycloak provisioning paths must not ship weak or
predictable credentials, and the `mcp-gateway-web` client must not be granted
wildcard CORS origins or a wildcard post-logout redirect allowlist.

All assertions are static reads of the setup scripts, the Helm provisioner, the
ECS provisioner and the realm descriptor, so they hold without a live Keycloak.
"""

import json
import re
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def repo_root() -> Path:
    """Get repository root directory."""
    return Path(__file__).parent.parent.parent


# Both copies of the Keycloak init script must satisfy the SA-9 invariants.
INIT_SCRIPTS = [
    "keycloak/setup/init-keycloak.sh",
    "terraform/aws-ecs/scripts/init-keycloak.sh",
]

# Weak default-password patterns that must never reappear in the init scripts.
FORBIDDEN_FALLBACKS = [
    ":-changeme",
    ":-testpass",
    ":-lob1pass",
    ":-lob2pass",
]


@pytest.mark.parametrize("script_path", INIT_SCRIPTS)
def test_init_script_has_no_weak_password_fallbacks(
    repo_root: Path,
    script_path: str,
):
    """No `${VAR:-weakdefault}` password fallbacks in the init scripts."""
    content = (repo_root / script_path).read_text()
    offenders = [pat for pat in FORBIDDEN_FALLBACKS if pat in content]
    assert not offenders, (
        f"{script_path}: weak default-password fallback(s) present: {offenders}. "
        "Require the password env var instead (fail closed when unset)."
    )


@pytest.mark.parametrize("script_path", INIT_SCRIPTS)
def test_init_script_creates_no_testuser(
    repo_root: Path,
    script_path: str,
):
    """The insecure `testuser` demo account must not be created."""
    content = (repo_root / script_path).read_text()
    assert "testuser" not in content, (
        f"{script_path}: still references 'testuser'; the weak demo account was removed."
    )


@pytest.mark.parametrize("script_path", INIT_SCRIPTS)
def test_init_script_users_are_temporary(
    repo_root: Path,
    script_path: str,
):
    """Every created credential forces a reset on first login (no temporary:false)."""
    content = (repo_root / script_path).read_text()
    assert '"temporary": false' not in content, (
        f"{script_path}: a user credential is created with 'temporary': false; "
        "all created users must use 'temporary': true to force a first-login reset."
    )


def test_primary_init_requires_admin_password(repo_root: Path):
    """The primary init script fails closed when INITIAL_ADMIN_PASSWORD is unset."""
    content = (repo_root / "keycloak/setup/init-keycloak.sh").read_text()
    assert 'if [ -z "$INITIAL_ADMIN_PASSWORD" ]' in content, (
        "keycloak/setup/init-keycloak.sh must guard on INITIAL_ADMIN_PASSWORD being set."
    )


def test_ecs_init_requires_lob_passwords(repo_root: Path):
    """The ECS init script fails closed when LOB user passwords are unset."""
    content = (repo_root / "terraform/aws-ecs/scripts/init-keycloak.sh").read_text()
    assert 'if [ -z "$LOB1_USER_PASSWORD" ] || [ -z "$LOB2_USER_PASSWORD" ]' in content, (
        "terraform/aws-ecs/scripts/init-keycloak.sh must guard on LOB user passwords."
    )


def test_realm_import_has_no_testuser(repo_root: Path):
    """The realm import must not ship a testuser with a weak default password."""
    data = json.loads((repo_root / "keycloak/import/realm-config.json").read_text())
    usernames = {u.get("username") for u in data.get("users", [])}
    assert "testuser" not in usernames, "realm-config.json still defines a 'testuser'."


def test_realm_import_ships_no_credentials(repo_root: Path):
    """The realm descriptor must ship no password material at all.

    Keycloak's realm import has no fail-closed mode for a credential: an
    unresolved ``${VAR}`` placeholder is re-emitted verbatim
    (``StringPropertyReplacer`` appends ``"${" + key + "}"`` when the resolver
    returns null) and ``RepresentationToModel.createCredentials`` then stores
    that literal text as the user's real password. ``"temporary": true`` does
    not help either -- the import path never reads it. So the only way this
    descriptor cannot leak a predictable credential is to carry none: the admin
    password is set by ``keycloak/setup/init-keycloak.sh``, which does guard on
    ``INITIAL_ADMIN_PASSWORD`` being non-empty.
    """
    content = (repo_root / "keycloak/import/realm-config.json").read_text()
    assert "INITIAL_ADMIN_PASSWORD" not in content, (
        "realm-config.json references INITIAL_ADMIN_PASSWORD; the descriptor must "
        "ship no credential, because an unresolved placeholder becomes the "
        "literal password on import."
    )
    data = json.loads(content)
    for user in data.get("users", []):
        for cred in user.get("credentials", []):
            assert not cred.get("value"), (
                f"realm-config.json user {user.get('username')!r} ships a "
                f"credential value {cred.get('value')!r}; remove it and let "
                "init-keycloak.sh set the password from a validated env var."
            )


def test_realm_import_uses_import_placeholder_syntax(repo_root: Path):
    """Realm-import placeholders must be `${VAR}`, never `${env.VAR}`.

    ``AbstractFileBasedImportProvider`` resolves placeholders with
    ``property -> System.getenv(property)`` -- it does NOT strip an ``env.``
    prefix (only the theme resolver, ``SystemEnvProperties``, does). So
    ``${env.REGISTRY_URL}`` looks up the variable named literally
    ``env.REGISTRY_URL``, always misses, and is stored as literal text.
    """
    content = (repo_root / "keycloak/import/realm-config.json").read_text()
    assert "${env." not in content, (
        "realm-config.json uses ${env.VAR}, which never resolves during realm "
        "import and is stored as literal text. Use ${VAR}."
    )


# Every provisioner that creates `mcp-gateway-web`. The realm descriptor is a
# latent import file; the other three are the paths that actually run.
WEB_CLIENT_PROVISIONERS = [
    "keycloak/import/realm-config.json",
    "keycloak/setup/init-keycloak.sh",
    "charts/keycloak-configure/templates/configmap.yaml",
    "terraform/aws-ecs/scripts/init-keycloak.sh",
]


def _client_block(content: str, client_id: str) -> str:
    """The text defining `client_id`, from its `clientId` up to the next one.

    Anchoring matters: every provisioner also declares `mcp-gateway-m2m`, and a
    bare first-match regex would silently assert against whichever client happens
    to be declared first.
    """
    start = content.find(f'"clientId": "{client_id}"')
    assert start != -1, f"client {client_id!r} not declared"
    nxt = content.find('"clientId"', start + 1)
    return content[start:] if nxt == -1 else content[start:nxt]


def _resolved_entries(fragment: str, content: str) -> list[str]:
    """Every individual allowlist entry reachable from `fragment`, normalized.

    The ECS provisioner does not inline its values: it builds `web_origins` and
    `post_logout_uris` as shell strings and interpolates them into the client JSON.
    Matching only the JSON there captures a stray quote, which is truthy and
    contains no `+` -- so an array-only assertion would pass while never inspecting
    the initializer where a `+` would actually be reintroduced. So follow every
    `$var` to its assignments, then split on all three delimiters this repo uses
    (`,` in JSON arrays, `##` in Keycloak multi-valued attributes, newline between
    fragments) and strip the surrounding quotes -- otherwise a shell-quoted `'+'`
    reads as the three-character string `'+'` and slips past an equality check.
    """
    texts = [fragment]
    for var in set(re.findall(r"\$\{?([a-z_][a-z0-9_]*)\}?", fragment)):
        texts += re.findall(rf"^\s*(?:local\s+)?{re.escape(var)}\s*=\s*(.+)$", content, re.M)
    entries = []
    for text in texts:
        for raw in re.split(r"##|,|\n", text):
            entry = raw.strip().strip("'\"").strip()
            if entry:
                entries.append(entry)
    return entries


@pytest.mark.parametrize("provisioner", WEB_CLIENT_PROVISIONERS)
def test_web_client_has_no_wildcard_web_origin(repo_root: Path, provisioner: str):
    """No provisioner may grant the `+` wildcard web origin.

    ``WebOriginsUtils.resolveValidWebOrigins`` expands ``"+"`` into the origin of
    every registered redirect URI, so CORS silently widens whenever a redirect is
    added (e.g. a CloudFront domain). Nothing needs it: the browser only performs
    top-level navigations to Keycloak, and the auth-server does the code-for-token
    exchange server-side over internal DNS.
    """
    content = (repo_root / provisioner).read_text()
    block = _client_block(content, "mcp-gateway-web")
    match = re.search(r'"webOrigins"\s*:\s*\[(.*?)\]', block, re.S)
    assert match, f"{provisioner} declares no webOrigins for mcp-gateway-web"
    entries = _resolved_entries(match.group(1), content)
    assert "+" not in entries, (
        f"{provisioner} grants the '+' wildcard web origin on mcp-gateway-web "
        f"(entries: {entries}); list the allowed origins explicitly."
    )


@pytest.mark.parametrize("provisioner", WEB_CLIENT_PROVISIONERS)
def test_web_client_post_logout_redirect_is_pinned(repo_root: Path, provisioner: str):
    """Post-logout redirects must be an explicit allowlist, never `+` or unset.

    ``OIDCAdvancedConfigWrapper.getPostLogoutRedirectUris`` treats ``"+"`` -- and
    an absent/empty attribute -- as "union with redirectUris", which includes the
    auth-server OAuth callback. That is not a valid logout landing page.
    """
    content = (repo_root / provisioner).read_text()
    block = _client_block(content, "mcp-gateway-web")
    match = re.search(r'"post\.logout\.redirect\.uris"\s*:\s*"(.*)"', block)
    assert match, (
        f"{provisioner} does not set post.logout.redirect.uris on mcp-gateway-web; "
        "an unset attribute falls back to redirectUris."
    )
    entries = _resolved_entries(match.group(1), content)
    assert "+" not in entries, (
        f"{provisioner} uses a wildcard post-logout redirect (entries: {entries})."
    )
    # Guard against the allowlist collapsing to a single unresolved interpolation:
    # every provisioner must name at least the two loopback landing pages plus its
    # deployment URL, so a real list always yields 3+ entries.
    assert len(entries) >= 3, (
        f"{provisioner} post-logout allowlist did not resolve to an explicit list "
        f"(entries: {entries}); it must name each landing origin."
    )


# The provisioners that actually execute against a live Keycloak. The realm
# descriptor is excluded: it is a latent import file, not a running code path.
EXECUTABLE_PROVISIONERS = [
    "keycloak/setup/init-keycloak.sh",
    "charts/keycloak-configure/templates/configmap.yaml",
    "terraform/aws-ecs/scripts/init-keycloak.sh",
]


@pytest.mark.parametrize("provisioner", EXECUTABLE_PROVISIONERS)
def test_provisioner_updates_an_existing_client(repo_root: Path, provisioner: str):
    """A provisioner must UPDATE `mcp-gateway-web`, not only create it.

    These scripts run against realms that already exist. A bare `POST .../clients`
    gets HTTP 409 on a re-run, so every hardening change above would be a silent
    no-op for existing deployments while the script still reported success. The
    sibling assertions in this module are static file reads, so without this guard
    the suite would certify a change the provisioner never applies.
    """
    content = (repo_root / provisioner).read_text()
    # The URL must terminate at the client id: a PUT to a SUB-resource such as
    # .../clients/{id}/default-client-scopes/{id} exists in all three scripts and
    # would satisfy a looser pattern without ever replacing the representation.
    client_put = re.compile(
        r"-X\s+PUT\s+\"\$\{KEYCLOAK_URL\}/admin/realms/\$\{REALM\}/clients/"
        r"\$\{[A-Za-z_][A-Za-z0-9_]*\}\""
    )
    assert client_put.search(content), (
        f"{provisioner} never PUTs a full client representation, so it cannot "
        "apply webOrigins/post-logout changes to an existing realm; add a "
        "409->PUT upsert (see upsert_client in keycloak/setup/init-keycloak.sh)."
    )


@pytest.mark.parametrize("provisioner", EXECUTABLE_PROVISIONERS)
def test_provisioner_updates_an_existing_realm(repo_root: Path, provisioner: str):
    """A provisioner must apply realm settings to an EXISTING realm, not skip it.

    Same defect class as the client upsert: every one of these scripts used to
    treat an already-present realm as "nothing to do", so a realm-level security
    setting added here (`bruteForceProtected`) would never reach a deployment whose
    realm predates the change, while the script still reported success.
    """
    content = (repo_root / provisioner).read_text()
    realm_put = re.compile(r"-X\s+PUT\s+\"\$\{KEYCLOAK_URL\}/admin/realms/\$\{REALM\}\"")
    assert realm_put.search(content), (
        f"{provisioner} never PUTs the realm representation, so realm settings "
        "such as bruteForceProtected cannot reach an existing realm; PUT "
        "/admin/realms/{realm} when creation returns 409."
    )
    # A PUT that exists but is unreachable is worse than none, because the
    # assertion above would certify it. Both former early-outs bypassed it: a
    # pre-POST existence probe that returned 0, and a 409 treated as "Continuing".
    for dead_end in ("Realm already exists. Skipping creation", "Realm already exists. Continuing"):
        assert dead_end not in content, (
            f"{provisioner} still short-circuits on an existing realm "
            f"({dead_end!r}), which makes the realm PUT unreachable."
        )


# Variables that decide WHICH Keycloak is targeted or WHAT gets written into it.
# A stale `.env` must not win over the caller for any of them.
CALLER_PRECEDENCE_VARS = ["KEYCLOAK_ADMIN_URL", "REGISTRY_URL", "AUTH_SERVER_EXTERNAL_URL"]


def test_compose_provisioner_prefers_caller_urls_over_dotenv(repo_root: Path):
    """Caller-exported URLs must win over `.env` in `init-keycloak.sh`.

    Now that an existing client is PUT rather than skipped, these variables decide
    whether a re-run preserves or destroys a deployed realm: Keycloak replaces
    `redirectUris`/`webOrigins` wholesale on a client PUT (`RepresentationToModel`
    `collectionToSet`), so on a host that also has a Compose `.env` the file's
    localhost values would overwrite a CloudFront deployment and break login with
    `invalid_redirect_uri`. `KEYCLOAK_ADMIN_URL` is listed for the inverse reason:
    a stale `.env` would retarget the whole run at a local Keycloak.
    """
    content = (repo_root / "keycloak/setup/init-keycloak.sh").read_text()
    match = re.search(r"^\s*CALLER_PRECEDENCE_VARS=\"([^\"]*)\"", content, re.M)
    assert match, (
        "init-keycloak.sh does not declare CALLER_PRECEDENCE_VARS, so a stale .env "
        "can overwrite a deployed client's URIs on the new client PUT."
    )
    declared = match.group(1).split()
    for var in CALLER_PRECEDENCE_VARS:
        assert var in declared, (
            f"{var} is missing from CALLER_PRECEDENCE_VARS in init-keycloak.sh; a "
            f"stale .env value for it would win over the caller."
        )


def test_post_deploy_exports_urls_to_the_provisioner(repo_root: Path):
    """`post-deploy.sh` must EXPORT the CDK URLs, not only write them to `.env`.

    The precedence above only fires if the caller's values are actually in the
    child's environment. `post-deploy.sh` invokes the provisioner with
    `bash "$init_script"` and writes `.env` only when absent, so without an
    explicit export the snapshot is empty, the precedence never triggers, and a
    pre-existing Compose `.env` silently wins.
    """
    content = (repo_root / "infra/scripts/post-deploy.sh").read_text()
    for var in CALLER_PRECEDENCE_VARS:
        assert re.search(rf"^\s*export {var}=", content, re.M), (
            f"infra/scripts/post-deploy.sh does not export {var}, so "
            "init-keycloak.sh cannot prefer it over a pre-existing .env."
        )


# `sslRequired` is null-guarded in Keycloak's `DefaultExportImportManager.updateRealm`,
# so a realm PUT that omits it preserves whatever the realm already has. The three
# provisioners want opposite things, and both are deliberate:
#   - Helm ENFORCES "none" on every run. That topology reaches Keycloak over plain
#     HTTP on the in-cluster service address, and "external" makes Keycloak answer
#     "HTTPS required" (the EKS pod CIDR 100.64/10 is not treated as private), which
#     breaks the token and admin calls the Job itself makes. TLS terminates at the ALB.
#   - Compose and ECS OMIT it, so an operator who hardened the realm keeps their value.
SSL_REQUIRED_EXPECTATION = {
    "charts/keycloak-configure/templates/configmap.yaml": "none",
    "keycloak/setup/init-keycloak.sh": None,
    "terraform/aws-ecs/scripts/init-keycloak.sh": None,
}


@pytest.mark.parametrize("provisioner", EXECUTABLE_PROVISIONERS)
def test_realm_ssl_required_matches_topology(repo_root: Path, provisioner: str):
    """Each provisioner's realm body handles `sslRequired` per its topology."""
    expected = SSL_REQUIRED_EXPECTATION[provisioner]
    content = (repo_root / provisioner).read_text()
    put_bodies = re.findall(
        r"-X PUT \"\$\{KEYCLOAK_URL\}/admin/realms/\$\{REALM\}\"(?:[^)]*?)-d \"\$(\w+)\"",
        content,
        re.S,
    )
    assert put_bodies, f"{provisioner} issues no realm PUT"
    for body_var in set(put_bodies):
        match = re.search(rf"{body_var}='(\{{.*?\n\s*\}})'", content, re.S)
        assert match, f"{provisioner}: cannot resolve realm PUT body ${body_var}"
        body = json.loads(match.group(1))
        actual = body.get("sslRequired")
        assert actual == expected, (
            f"{provisioner} realm PUT body (${body_var}) has sslRequired={actual!r}, "
            f"expected {expected!r}. Helm must enforce 'none' because its in-cluster "
            f"HTTP calls fail with 'HTTPS required' otherwise; Compose and ECS must "
            f"omit it so an operator's hardened value survives an upgrade."
        )


@pytest.mark.parametrize("provisioner", EXECUTABLE_PROVISIONERS)
def test_clients_restrict_full_scope(repo_root: Path, provisioner: str):
    """Both gateway clients must set `fullScopeAllowed: false`.

    With full scope allowed -- Keycloak's default when the field is omitted -- a
    token minted for these clients carries EVERY role mapping the user holds,
    including client roles belonging to other clients. Verified on a live realm:
    granting a user `realm-management:manage-users` made
    `resource_access["realm-management"]` appear in an `mcp-gateway-web` access
    token, so a gateway-scoped token becomes replayable against Keycloak's own
    Admin API. Setting it false emptied `realm_access` and `resource_access`
    entirely. Nothing in this repo authorizes on roles (authorization is
    group-driven via the `groups` mapper), so restricting scope removes the
    exposure without changing behaviour.
    """
    content = (repo_root / provisioner).read_text()
    for client in ("web_client_json", "m2m_client_json"):
        match = re.search(rf"{client}='(\{{.*?\n\s*\}})'", content, re.S)
        assert match, f"{provisioner}: cannot resolve ${client}"
        # These bodies interpolate shell variables (including whole JSON arrays, as
        # the ECS provisioner does for redirectUris), so they are not parseable as
        # JSON. Read the one field out of the block instead.
        field = re.search(r'"fullScopeAllowed"\s*:\s*(true|false)', match.group(1))
        assert field and field.group(1) == "false", (
            f"{provisioner} ${client} has fullScopeAllowed="
            f"{field.group(1) if field else '<absent>'}; set it false so tokens do "
            "not carry every role the user holds, including other clients' roles."
        )


@pytest.mark.parametrize("provisioner", EXECUTABLE_PROVISIONERS)
def test_realm_enables_brute_force_protection(repo_root: Path, provisioner: str):
    """Each provisioned realm must throttle repeated failed logins.

    The realm descriptor sets `bruteForceProtected: true`, but it is never
    imported, so the value only takes effect if the provisioners set it too --
    otherwise it defaults to false and the browser login form (which needs no
    client secret) is unthrottled. This is the precondition that makes retaining
    the direct-access grant on a confidential client defensible.
    """
    content = (repo_root / provisioner).read_text()
    assert '"bruteForceProtected": true' in content, (
        f"{provisioner} does not enable bruteForceProtected on the realm it "
        "creates; failed logins would be unthrottled."
    )
