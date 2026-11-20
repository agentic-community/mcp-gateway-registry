"""Exercise production shell mapper boundaries against a fake Keycloak API."""

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HELM = "charts/keycloak-configure/templates/configmap.yaml"
HELPER = "keycloak/setup/group-path-preflight.sh"
MODELS = "/admin/realms/mcp-gateway/clients/client-uuid/protocol-mappers/models"
MAPPER = {
    "id": "mapper-id",
    "name": "groups",
    "protocol": "openid-connect",
    "protocolMapper": "oidc-group-membership-mapper",
    "config": {"full.path": "false", "claim.name": "groups", "custom": "preserved"},
}
CURRENT_MAPPER = {
    **MAPPER,
    "config": {**MAPPER["config"], "full.path": "true", "claim.name": "mcpgw_group_paths"},
}
MUTATORS = [
    ("keycloak/setup/init-keycloak.sh", '_upsert_groups_mapper "$TOKEN" "$MODELS" client'),
    ("keycloak/setup/upgrade-realm-for-dcr.sh", 'setup_dcr_groups_mapper "$TOKEN"'),
    (
        "terraform/aws-ecs/scripts/init-keycloak.sh",
        '_upsert_groups_mapper "$TOKEN" client-uuid client',
    ),
    ("terraform/aws-ecs/scripts/user_mgmt.sh", "add_groups_mapper client-uuid"),
    ("cli/user_mgmt.sh", "add_groups_mapper client-uuid"),
    ("keycloak/setup/setup-agent-service-account.sh", "ensure_groups_mapper"),
    ("keycloak/setup/setup-federation-service-account.sh", "_add_groups_mapper"),
    ("keycloak/setup/setup-m2m-service-account.sh", "add_groups_mapper"),
    ("setup/idp/keycloak/setup-ide-public-client.sh", "_add_groups_mapper"),
    (HELM, '_upsert_groups_mapper "$TOKEN" "$KEYCLOAK_URL$MODELS" client'),
]
CONVERTERS = [
    (path, '_ensure_groups_mapper_full_path "$TOKEN" "$MODELS" "$EXISTING_MAPPER" client')
    for path in (MUTATORS[0][0], MUTATORS[1][0], HELM)
]

# Only the curl process is replaced. The scripts' real HTTP parsing, group-tree
# walking, collision decision, mapper JSON, and mutation ordering all execute.
CURL_STUB = r"""
import json
import os
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

args = sys.argv[1:]
def option(name, default=""):
    return args[args.index(name) + 1] if name in args else default

method = option("-X", "GET")
url = next(arg for arg in args if arg.startswith(("http://", "https://")))
parsed = urlsplit(url)
path = parsed.path
query = parse_qs(parsed.query)
body_arg = option("-d", option("--data", option("--data-binary", "")))
if body_arg.startswith("@"):
    body_arg = sys.stdin.read() if body_arg == "@-" else Path(body_arg[1:]).read_text()
authorization = ""
for index, argument in enumerate(args[:-1]):
    if argument in ("-H", "--header"):
        header = args[index + 1]
        if header.startswith("@"):
            header = Path(header[1:]).read_text().strip()
        if header.startswith("Authorization:"):
            authorization = header
with Path(os.environ["API_LOG"]).open("a") as log:
    log.write(json.dumps({"method": method, "path": path, "query": query, "body": body_arg, "authorization": authorization}) + "\n")
status = 200
body = []
scenario = os.environ["GROUP_SCENARIO"]
mapper = json.loads(os.environ["FIXTURE_MAPPER_JSON"])
if scenario in ("bulk-clear", "bulk-unchanged", "bulk-old-full-path", "bulk-custom"):
    mapper = dict(mapper, config=dict(mapper["config"], **{
        "full.path": "false" if scenario in ("bulk-clear", "bulk-custom") else "true",
        "claim.name": "mcpgw_group_paths" if scenario == "bulk-unchanged" else
            "custom_groups" if scenario == "bulk-custom" else "groups"}))
mappers = [mapper] if os.environ["MAPPER_MODE"] == "update" else []
if scenario in ("foreign-path-mapper", "foreign-dot-path-mapper"):
    mappers.append({"id": "foreign", "name": "foreign", "protocol": "openid-connect",
                    "protocolMapper": "oidc-hardcoded-claim-mapper",
                    "config": {"claim.name": "mcpgw_group_paths.admin" if scenario == "foreign-dot-path-mapper" else "mcpgw_group_paths",
                               "claim.value": "/registry-admins"}})
root = "/admin/realms/mcp-gateway/groups"
parent = {"id": "parent", "name": "finance", "subGroupCount": 1}
slash = {"id": "slash", "name": "finance/read", "subGroupCount": 0}
child = {"id": "child", "name": "read", "subGroupCount": 0}
first = int(query.get("first", [0])[0])
maximum = int(query.get("max", [100])[0])
raw = None

if path == "/admin/serverinfo":
    policy_type = "org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy"
    body = {"systemInfo": {"version": "26.2.4" if scenario in ("roles-missing", "exchange-refused", "exchange-removed", "cimd-old") else "26.8.2"},
            "componentTypes": {policy_type: [] if scenario == "roles-missing" else [{"id": "default-roles"}]}}
    body["providers"] = {"client-registration": {"providers": {"openid-connect": {}}}}
    if scenario.startswith("exchange-"):
        body["componentTypes"]["org.keycloak.services.clientpolicy.condition.ClientPolicyConditionProvider"] = [{"id": "grant-type"}]
        body["features"] = [{"name": "TOKEN_EXCHANGE_STANDARD_V2", "enabled": True}]
    if scenario.startswith("cimd-"):
        body["features"] = [] if scenario == "cimd-feature-missing" else [{"name": "CIMD", "enabled": True}]
        body["componentTypes"]["org.keycloak.services.clientpolicy.condition.ClientPolicyConditionProvider"] = [] if scenario == "cimd-condition-missing" else [{"id": name} for name in ("client-id-uri", "client-scopes", "client-roles", "client-access-type", "client-type", "client-attributes", "grant-type")]
        body["componentTypes"]["org.keycloak.services.clientpolicy.executor.ClientPolicyExecutorProvider"] = [] if scenario == "cimd-executor-missing" else [{"id": name} for name in ("client-id-metadata-document", "reject-request", "pkce-enforcer", "secure-client-uris-pattern")]
elif path.endswith("/client-policies/policies"):
    body = {"policies": [{"name": "mcp-restrict-token-exchange"}, {"name": "unrelated"}]}
    if scenario.startswith("cimd-"):
        body = {"policies": [{"name": "unrelated"}, {"name": "cimd-vendors-policy"}]}
    policy_state = Path(os.environ["API_LOG"]).with_suffix(".policies.json")
    if policy_state.exists():
        body = json.loads(policy_state.read_text())
    if method == "PUT":
        status = 403 if scenario == "exchange-refused" else 204
        if status == 204:
            policy_state.write_text(body_arg)
elif path.endswith("/client-policies/profiles"):
    body = {"profiles": [{"name": "unrelated-profile"}, {"name": "cimd-vendors"}]}
    profile_state = Path(os.environ["API_LOG"]).with_suffix(".profiles.json")
    if profile_state.exists():
        body = json.loads(profile_state.read_text())
    if method == "PUT":
        status = 204
        profile_state.write_text(body_arg)
elif path.endswith("/components"):
    state = Path(os.environ["API_LOG"]).with_suffix(".components.json")
    policies = json.loads(state.read_text()) if state.exists() else []
    if method == "POST":
        if scenario == "roles-write-failure":
            status = 403
        else:
            status = 201
            if scenario != "roles-readback-failure":
                policies.append(dict(json.loads(body_arg), id="policy-" + str(len(policies))))
                state.write_text(json.dumps(policies))
    body = policies
elif path == "/admin/realms/mcp-gateway":
    body = {"id": "realm-id"}
elif path == "/admin/realms/team-realm":
    state = Path(os.environ["API_LOG"]).with_suffix(".realm.json")
    if method == "GET":
        status = 200 if scenario == "custom-realm-existing" or state.exists() else 404
        body = {"id": "realm-id", "realm": "team-realm"}
    else:
        status = 204
        state.write_text(body_arg)
elif path == "/admin/realms" and method == "POST":
    state = Path(os.environ["API_LOG"]).with_suffix(".realm.json")
    status = 201
    state.write_text(body_arg)
elif path == root:
    if scenario == "transport-error":
        sys.exit(28)
    if scenario == "root-error":
        status, body = 503, {"error": "unavailable"}
    elif scenario == "invalid-json":
        raw = "not JSON"
    elif scenario == "invalid-group":
        body = [{"id": "missing-name", "subGroupCount": 0}]
    elif scenario in ("empty", "bulk-clear", "bulk-unchanged", "bulk-old-full-path", "bulk-custom"):
        body = []
    elif scenario in ("slash", "foreign-path-mapper", "foreign-dot-path-mapper", "mixed-migration", "inherited-path-mapper", "inherited-dot-path-mapper", "optional-path-mapper"):
        body = [slash]
    elif scenario == "old-clear":
        body = [{"id": "plain", "name": "plain", "subGroups": []}]
    elif scenario == "old-clash":
        body = [dict(slash, subGroups=[]), {"id": "parent", "name": "finance", "subGroups": [dict(child, subGroups=[])]}]
        body[0].pop("subGroupCount")
        body[1]["subGroups"][0].pop("subGroupCount")
    elif scenario == "missing-count-clash":
        body = [slash, {"id": "parent", "name": "finance"}]
    elif scenario == "paged-root-clash":
        body = [{"id": f"filler-{i}", "name": f"filler-{i}", "subGroupCount": 0} for i in range(99)] + [slash, parent]
    elif scenario == "paged-child-clash":
        body = [slash, dict(parent, subGroupCount=101)]
    else:
        body = [slash, parent]
    if isinstance(body, list):
        body = body[first:first + maximum]
elif path == root + "/parent/children":
    if scenario == "child-error":
        status, body = 403, {"error": "forbidden"}
    elif scenario == "invalid-child":
        body = [{"id": "child", "name": None}]
    elif scenario == "paged-child-clash":
        body = [{"id": f"child-{i}", "name": f"child-{i}", "subGroupCount": 0} for i in range(100)] + [child]
        body = body[first:first + maximum]
    elif scenario == "old-clash":
        # Old API contract: children are embedded; no children endpoint exists.
        status = 404
    else:
        body = [child] if first == 0 else []
elif path.endswith("/protocol/openid-connect/token"):
    body = {"access_token": "fixture-token", "expires_in": 60}
elif path.endswith("/clients") and method == "GET":
    client_id = query.get("clientId", ["mcp-gateway-web"])[0]
    if scenario.startswith("custom-realm-"):
        body = []
    elif scenario == "sweep":
        body = [{"id": "browser", "clientId": "registered-browser", "enabled": True, "publicClient": True},
                {"id": "builtin", "clientId": "account", "enabled": True}]
    elif scenario in ("bulk-clear", "bulk-unchanged", "bulk-old-full-path", "bulk-custom"):
        body = [{"id": "one", "clientId": "one", "protocolMappers": [mapper]},
                {"id": "two", "clientId": "two", "protocolMappers": [mapper]}]
    else:
        body = [{"id": client_id + "-uuid", "clientId": client_id}]
elif scenario.startswith("custom-realm-") and path == "/admin/realms/team-realm/clients" and method == "POST":
    status = 201
elif scenario == "sweep" and path.endswith("/registration-access-token") and method == "POST":
    body = {"registrationAccessToken": "replacement-token"}
elif scenario == "sweep" and "/clients/" in path and method == "PUT":
    status = 204
elif "/clients/" in path and "/roles" in path:
    role_state = Path(os.environ["API_LOG"]).with_suffix(".roles.json")
    roles = json.loads(role_state.read_text()) if role_state.exists() else {}
    if method == "POST":
        role = dict(json.loads(body_arg), clientRole=True, composite=False)
        roles[path + "/" + role["name"]] = role
        role_state.write_text(json.dumps(roles))
        status = 201
    else:
        status = 200 if path in roles else 404
        body = roles.get(path, {})
elif path.endswith(("/default-default-client-scopes", "/default-optional-client-scopes")):
    body = []
elif path.endswith(("/default-client-scopes", "/optional-client-scopes")):
    attached = scenario in ("mixed-migration", "inherited-path-mapper", "inherited-dot-path-mapper") or (
        scenario == "optional-path-mapper" and path.endswith("/optional-client-scopes"))
    body = [{"id": "basic", "name": "basic"}] if attached else []
elif path.endswith("/client-scopes"):
    body = [{"id": "basic", "name": "basic"}]
    if scenario.startswith("bulk-"):
        body.append({"id": "custom-scope", "name": "custom-scope"})
elif path.endswith("/client-scopes/basic"):
    body = {"id": "basic", "protocolMappers": mappers}
elif "/protocol-mappers/models" in path:
    if method == "GET":
        if "/client-scopes/" in path and scenario in ("mixed-migration", "inherited-path-mapper", "inherited-dot-path-mapper", "optional-path-mapper"):
            body = [dict(mapper, config=dict(mapper["config"], **{
                "claim.name": "mcpgw_group_paths", "full.path": "true"}))]
            if scenario in ("inherited-path-mapper", "optional-path-mapper"):
                body[0]["protocolMapper"] = "oidc-hardcoded-claim-mapper"
            if scenario == "inherited-dot-path-mapper":
                body[0]["config"]["claim.name"] = "mcpgw_group_paths.admin"
        else:
            body = mappers
    else:
        status = 201 if method == "POST" else 204
else:
    print("Unexpected Keycloak request: " + path, file=sys.stderr)
    sys.exit(97)

if option("-o") != "/dev/null":
    sys.stdout.write(raw if raw is not None else json.dumps(body, separators=(",", ":")))
write_out = option("-w").replace(r"\n", "\n").replace("%{http_code}", str(status))
sys.stdout.write(write_out)
"""


def _run_shell(
    tmp_path: Path,
    path: str,
    command: str,
    scenario: str,
    mode: str = "update",
    *,
    mapper: dict | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
    """Load a script without auto-running main; execute its real mapper function."""
    source = (REPO_ROOT / path).read_text()
    if path == HELM:
        source = textwrap.dedent(source.split("  script.sh: |\n", 1)[1].rsplit("{{- end", 1)[0])
        # Package installation belongs to the image boundary, not this API test.
        source = re.sub(r"^apt .*\n", "", source, flags=re.MULTILINE)
    source = re.sub(r'^main(?: "\$@")?\s*$', "", source, flags=re.MULTILINE)
    script_path = tmp_path / path
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(source)
    helper_path = tmp_path / HELPER
    helper_path.parent.mkdir(parents=True, exist_ok=True)
    if path != HELPER:
        helper_path.write_text((REPO_ROOT / HELPER).read_text())
    credential_helper = helper_path.with_name("credential-io.sh")
    credential_helper.write_text((REPO_ROOT / "keycloak/setup/credential-io.sh").read_text())
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    curl = fake_bin / "curl"
    curl.write_text(f"#!{sys.executable}\n" + CURL_STUB)
    curl.chmod(0o755)
    log = tmp_path / "api.jsonl"
    fixture_mapper = MAPPER if mapper is None else mapper
    env = dict(os.environ)
    env.update(
        PATH=f"{fake_bin}{os.pathsep}{env['PATH']}",
        API_LOG=str(log),
        GROUP_SCENARIO=scenario,
        MAPPER_MODE=mode,
        EXISTING_MAPPER=json.dumps(fixture_mapper),
        FIXTURE_MAPPER_JSON=json.dumps(fixture_mapper),
        KEYCLOAK_ADMIN_PASSWORD="fixture-password",
        SCRIPT_UNDER_TEST=str(script_path),
        MODELS=MODELS,
        ADMIN_URL="http://keycloak.invalid",
    )
    result = subprocess.run(  # nosec B603 B607 - production shell against a local API stub
        [
            "bash",
            "-c",
            'source "$SCRIPT_UNDER_TEST" -a fixture\n'
            "KEYCLOAK_URL=http://keycloak.invalid\nADMIN_URL=$KEYCLOAK_URL\n"
            "REALM=mcp-gateway\nTOKEN=fixture-token\nCLIENT_ID=client-uuid\nCLIENT_UUID=client-uuid\n"
            + command,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    requests = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, requests


def _mutations(requests: list[dict]) -> list[dict]:
    return [
        request
        for request in requests
        if request["method"] != "GET" and "/protocol-mappers/models" in request["path"]
    ]


@pytest.fixture(autouse=True)
def _require_jq() -> None:
    if shutil.which("jq") is None:
        pytest.skip("jq is not installed")


@pytest.mark.parametrize(("path", "command"), MUTATORS)
@pytest.mark.parametrize("mode", ["create", "update"])
def test_collision_prevents_every_mapper_mutation(
    tmp_path: Path, path: str, command: str, mode: str
) -> None:
    """Neither new nor existing client/scope mappers can bypass the preflight."""
    result, requests = _run_shell(tmp_path, path, command, "clash", mode)
    assert result.returncode != 0, result.stdout + result.stderr
    assert _mutations(requests) == []


@pytest.mark.parametrize(("path", "command"), CONVERTERS)
def test_bulk_converter_cannot_bypass_collision_gate(
    tmp_path: Path, path: str, command: str
) -> None:
    result, requests = _run_shell(tmp_path, path, command, "clash")
    assert result.returncode != 0, result.stdout + result.stderr
    assert _mutations(requests) == []


@pytest.mark.parametrize(("path", "command"), MUTATORS)
@pytest.mark.parametrize("mode", ["create", "update"])
def test_noncolliding_legacy_slash_names_allow_mapper_changes(
    tmp_path: Path, path: str, command: str, mode: str
) -> None:
    """A slash-bearing legacy name alone is supported, not denied as a collision."""
    result, requests = _run_shell(tmp_path, path, command, "slash", mode)
    assert result.returncode == 0, result.stdout + result.stderr
    mutations = _mutations(requests)
    assert len(mutations) == 1
    assert mutations[0]["method"] == ("POST" if mode == "create" else "PUT")
    config = json.loads(mutations[0]["body"])["config"]
    assert config["full.path"] == "true"
    assert config["claim.name"] == "mcpgw_group_paths"
    if mode == "update":
        assert config["custom"] == "preserved"


@pytest.mark.parametrize("path", [HELPER, HELM])
@pytest.mark.parametrize(
    "scenario", ["paged-root-clash", "paged-child-clash", "old-clash", "missing-count-clash"]
)
def test_enumerates_all_group_api_shapes(tmp_path: Path, path: str, scenario: str) -> None:
    result, requests = _run_shell(
        tmp_path, path, '_check_group_paths "$KEYCLOAK_URL" "$REALM" "$TOKEN"', scenario
    )
    assert result.returncode != 0, result.stdout + result.stderr
    if scenario.startswith("paged-"):
        assert any(request["query"].get("first") == ["100"] for request in requests)
    if scenario == "old-clash":
        assert not any(request["path"].endswith("/children") for request in requests)


@pytest.mark.parametrize("path", [HELPER, HELM])
@pytest.mark.parametrize("scenario", ["empty", "old-clear"])
def test_empty_and_plain_legacy_realms_are_supported(
    tmp_path: Path, path: str, scenario: str
) -> None:
    result, _ = _run_shell(
        tmp_path, path, '_check_group_paths "$KEYCLOAK_URL" "$REALM" "$TOKEN"', scenario
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(("path", "command"), MUTATORS)
@pytest.mark.parametrize(
    "scenario", ["root-error", "child-error", "invalid-json", "invalid-group", "invalid-child"]
)
def test_enumeration_failure_prevents_mapper_mutation(
    tmp_path: Path, path: str, command: str, scenario: str
) -> None:
    result, requests = _run_shell(tmp_path, path, command, scenario)
    assert result.returncode != 0, result.stdout + result.stderr
    assert _mutations(requests) == []


def test_terraform_web_and_m2m_flow_stops_before_any_mapper_write(tmp_path: Path) -> None:
    """The Terraform first-party phase runs before its delegated realm upgrade."""
    result, requests = _run_shell(
        tmp_path,
        "terraform/aws-ecs/scripts/init-keycloak.sh",
        'setup_groups_mapper "$TOKEN"',
        "clash",
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert _mutations(requests) == []


@pytest.mark.parametrize(("path", "command"), MUTATORS)
def test_transport_failure_prevents_mapper_mutation(
    tmp_path: Path, path: str, command: str
) -> None:
    result, requests = _run_shell(tmp_path, path, command, "transport-error")
    assert result.returncode != 0, result.stdout + result.stderr
    assert _mutations(requests) == []


@pytest.mark.parametrize(("path", "command"), CONVERTERS)
def test_noncolliding_bulk_conversion_preserves_mapper_config(
    tmp_path: Path, path: str, command: str
) -> None:
    result, requests = _run_shell(tmp_path, path, command, "slash")
    assert result.returncode == 0, result.stdout + result.stderr
    mutations = _mutations(requests)
    assert len(mutations) == 1
    assert json.loads(mutations[0]["body"])["config"] == {
        "full.path": "true",
        "claim.name": "mcpgw_group_paths",
        "custom": "preserved",
    }


@pytest.mark.parametrize("path", [MUTATORS[0][0], MUTATORS[1][0], HELM])
@pytest.mark.parametrize(
    "scenario", ["bulk-clear", "bulk-unchanged", "bulk-old-full-path", "bulk-custom"]
)
def test_bulk_mapper_conversion_scans_groups_once_only_when_needed(tmp_path, path, scenario):
    result, requests = _run_shell(tmp_path, path, 'update_client_groups_mappers "$TOKEN"', scenario)
    assert result.returncode == 0, result.stdout + result.stderr
    group_reads = [r for r in requests if r["path"].endswith("/groups")]
    assert len(group_reads) == (1 if scenario in ("bulk-clear", "bulk-old-full-path") else 0)
    changes = _mutations(requests)
    assert {r["path"] for r in changes} == (
        {
            f"/admin/realms/mcp-gateway/clients/{client}/protocol-mappers/models/mapper-id"
            for client in ("one", "two")
        }
        | {
            f"/admin/realms/mcp-gateway/client-scopes/{scope}/protocol-mappers/models/mapper-id"
            for scope in ("basic", "custom-scope")
        }
        if scenario in ("bulk-clear", "bulk-old-full-path")
        else set()
    )
    for change in changes:
        assert json.loads(change["body"])["config"]["claim.name"] == "mcpgw_group_paths"


@pytest.mark.parametrize(("path", "command"), MUTATORS + CONVERTERS)
def test_full_path_legacy_claim_still_migrates(tmp_path, path, command):
    old = {**MAPPER, "config": {**MAPPER["config"], "full.path": "true"}}
    result, requests = _run_shell(tmp_path, path, command, "slash", mapper=old)
    assert result.returncode == 0, result.stdout + result.stderr
    changes = _mutations(requests)
    assert len(changes) == 1
    payload = json.loads(changes[0]["body"])
    assert payload["config"]["claim.name"] == "mcpgw_group_paths"
    assert payload["config"]["full.path"] == "true"
    assert payload["config"]["custom"] == "preserved"


@pytest.mark.parametrize(("path", "command"), MUTATORS)
@pytest.mark.parametrize("scenario", ["foreign-path-mapper", "foreign-dot-path-mapper"])
def test_foreign_new_claim_emitter_prevents_mapper_write(tmp_path, path, command, scenario):
    result, requests = _run_shell(tmp_path, path, command, scenario)
    assert result.returncode != 0, result.stdout + result.stderr
    assert _mutations(requests) == []


@pytest.mark.parametrize(("path", "command"), MUTATORS[3:9])
@pytest.mark.parametrize(
    "scenario",
    [
        "mixed-migration",
        "inherited-path-mapper",
        "inherited-dot-path-mapper",
        "optional-path-mapper",
    ],
)
def test_client_writer_checks_inherited_claim_provenance(tmp_path, path, command, scenario):
    result, requests = _run_shell(tmp_path, path, command, scenario)
    changes = _mutations(requests)
    if scenario != "mixed-migration":
        assert result.returncode != 0, result.stdout + result.stderr
        assert changes == []
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(changes) == 1
        assert json.loads(changes[0]["body"])["config"]["claim.name"] == "mcpgw_group_paths"


@pytest.mark.parametrize(("path", "command"), MUTATORS)
@pytest.mark.parametrize(
    "config",
    [
        {"claim.name": "custom_groups", "full.path": "false"},
        {"claim.name": "mcpgw_group_paths", "full.path": "false"},
    ],
)
def test_named_mapper_cannot_repurpose_custom_or_invalid_reserved_claim(
    tmp_path, path, command, config
):
    result, requests = _run_shell(
        tmp_path, path, command, "slash", mapper={**MAPPER, "config": config}
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert _mutations(requests) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["create", "update"])
@pytest.mark.parametrize(
    "scenario",
    [
        "slash",
        "clash",
        "paged-root-clash",
        "paged-child-clash",
        "legacy-clash",
        "root-error",
        "child-error",
        "invalid-group",
        "invalid-child",
    ],
)
async def test_registry_mapper_checks_complete_group_tree_before_mutation(mode, scenario):
    import httpx

    from registry.utils import keycloak_manager as manager

    requests = []
    slash = {"id": "slash", "name": "finance/read", "subGroupCount": 0}
    child = {"id": "child", "name": "read", "subGroupCount": 0}
    parent = {"id": "parent", "name": "finance", "subGroupCount": 1}

    def respond(request):
        requests.append(request)
        path = request.url.path
        first = int(request.url.params.get("first", "0"))
        if path.endswith("/groups"):
            if scenario == "root-error":
                return httpx.Response(503)
            groups = [slash, parent]
            if scenario == "slash":
                groups = [slash]
            elif scenario == "invalid-group":
                groups = [{"id": "missing-name"}]
            elif scenario == "legacy-clash":
                groups = [
                    {"id": "slash", "name": "finance/read", "subGroups": []},
                    {
                        "id": "parent",
                        "name": "finance",
                        "subGroups": [{"id": "child", "name": "read", "subGroups": []}],
                    },
                ]
            elif scenario == "paged-root-clash":
                groups = [
                    {"id": f"filler-{i}", "name": f"filler-{i}", "subGroupCount": 0}
                    for i in range(99)
                ] + groups
            elif scenario == "paged-child-clash":
                groups = [slash, dict(parent, subGroupCount=101)]
            return httpx.Response(200, json=groups[first : first + 100])
        if path.endswith("/parent/children"):
            if scenario == "child-error":
                return httpx.Response(403)
            children = [child]
            if scenario == "invalid-child":
                children = [{"id": "child", "name": None}]
            elif scenario == "paged-child-clash":
                children = [
                    {"id": f"child-{i}", "name": f"child-{i}", "subGroupCount": 0}
                    for i in range(100)
                ] + children
            return httpx.Response(200, json=children[first : first + 100])
        if path.endswith(("/default-client-scopes", "/optional-client-scopes")):
            return httpx.Response(200, json=[])
        if "/protocol-mappers/models" in path:
            if request.method == "GET":
                return httpx.Response(200, json=[MAPPER] if mode == "update" else [])
            return httpx.Response(201 if request.method == "POST" else 204)
        raise AssertionError(f"Unexpected request: {request.method} {path}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        if scenario == "slash":
            await manager._ensure_groups_mapper(client, "token", "client-uuid")
        else:
            with pytest.raises((manager.KeycloakAdminError, httpx.HTTPStatusError)):
                await manager._ensure_groups_mapper(client, "token", "client-uuid")
    mutations = [request for request in requests if request.method != "GET"]
    if scenario != "slash":
        assert mutations == []
        return
    assert len(mutations) == 1
    assert mutations[0].method == ("POST" if mode == "create" else "PUT")
    payload = json.loads(mutations[0].content)
    assert payload["config"]["full.path"] == "true"
    assert payload["config"]["claim.name"] == "mcpgw_group_paths"
    if mode == "update":
        assert payload["config"]["custom"] == "preserved"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("local_mapper", "inherited_mapper", "scope_kind", "allowed", "changes"),
    [
        (MAPPER, CURRENT_MAPPER, "default-client-scopes", True, 1),
        (MAPPER, CURRENT_MAPPER, "optional-client-scopes", True, 1),
        (
            {**MAPPER, "config": {**MAPPER["config"], "full.path": "true"}},
            None,
            "default-client-scopes",
            True,
            1,
        ),
        (CURRENT_MAPPER, None, "default-client-scopes", True, 0),
        (
            MAPPER,
            {**CURRENT_MAPPER, "protocolMapper": "oidc-hardcoded-claim-mapper"},
            "default-client-scopes",
            False,
            0,
        ),
        (
            MAPPER,
            {**CURRENT_MAPPER, "protocolMapper": "oidc-hardcoded-claim-mapper"},
            "optional-client-scopes",
            False,
            0,
        ),
        (
            MAPPER,
            {
                **CURRENT_MAPPER,
                "config": {"claim.name": "mcpgw_group_paths.admin", "full.path": "true"},
            },
            "optional-client-scopes",
            False,
            0,
        ),
        (
            MAPPER,
            {**CURRENT_MAPPER, "config": {"claim.name": "mcpgw_group_paths", "full.path": "false"}},
            "default-client-scopes",
            False,
            0,
        ),
        (
            {**CURRENT_MAPPER, "config": {"claim.name": "mcpgw_group_paths", "full.path": "false"}},
            None,
            "default-client-scopes",
            False,
            0,
        ),
        (
            {**MAPPER, "config": {"claim.name": "custom_groups", "full.path": "true"}},
            None,
            "default-client-scopes",
            False,
            0,
        ),
        ({**CURRENT_MAPPER, "name": "foreign"}, None, "default-client-scopes", False, 0),
        (
            {**CURRENT_MAPPER, "name": "foreign", "protocolMapper": "oidc-hardcoded-claim-mapper"},
            None,
            "default-client-scopes",
            False,
            0,
        ),
    ],
)
async def test_registry_mapper_preserves_claim_provenance(
    local_mapper, inherited_mapper, scope_kind, allowed, changes
):
    import httpx

    from registry.utils import keycloak_manager as manager

    mutations = []

    def respond(request):
        path = request.url.path
        if request.method != "GET":
            mutations.append(request)
            return httpx.Response(204)
        if path.endswith("/groups"):
            return httpx.Response(200, json=[])
        if path.endswith(("/default-client-scopes", "/optional-client-scopes")):
            scopes = [{"id": "basic"}] if inherited_mapper and path.endswith(scope_kind) else []
            return httpx.Response(200, json=scopes)
        if "/client-scopes/basic/protocol-mappers/models" in path:
            return httpx.Response(200, json=[inherited_mapper])
        if path.endswith("/clients/client-uuid/protocol-mappers/models"):
            return httpx.Response(200, json=[local_mapper])
        raise AssertionError(f"Unexpected request: {path}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        if allowed:
            await manager._ensure_groups_mapper(client, "token", "client-uuid")
        else:
            with pytest.raises(manager.KeycloakAdminError):
                await manager._ensure_groups_mapper(client, "token", "client-uuid")
    assert len(mutations) == changes
    for mutation in mutations:
        payload = json.loads(mutation.content)
        assert payload["name"] == "groups"
        assert payload["config"] == CURRENT_MAPPER["config"]


@pytest.mark.asyncio
async def test_registry_memberships_keep_admin_named_subgroups_namespaced():
    import httpx

    from registry.utils import keycloak_manager as manager

    memberships = [
        {"id": "admin", "name": "registry-admins", "path": "/registry-admins"},
        {"id": "nested", "name": "registry-admins", "path": "/team/registry-admins"},
        {"id": "slashes", "name": "/registry-admins", "path": "//registry-admins"},
    ]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=memberships))
    ) as client:
        names = await manager._get_user_groups(client, "token", "user-id")
    assert names == ["registry-admins", "team/registry-admins", "/registry-admins"]


@pytest.mark.asyncio
async def test_registry_group_removal_resolves_full_path_not_leaf_name():
    import httpx

    from registry.utils import keycloak_manager as manager

    deleted = []

    def respond(request):
        if request.method == "DELETE":
            deleted.append(request.url.path)
            return httpx.Response(204)
        return httpx.Response(
            200,
            json=[
                {"id": "admin", "name": "registry-admins", "subGroups": []},
                {
                    "id": "team",
                    "name": "team",
                    "subGroups": [{"id": "nested", "name": "registry-admins", "subGroups": []}],
                },
            ],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        await manager._remove_user_from_groups_by_name(
            client, "token", "user-id", ["team/registry-admins"]
        )
    assert deleted == [f"/admin/realms/{manager.KEYCLOAK_REALM}/users/user-id/groups/nested"]
