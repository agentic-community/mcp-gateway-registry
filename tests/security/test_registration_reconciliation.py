"""Exercise registration cutovers through the scripts' real Admin API boundary."""

import json

import pytest

from tests.security import test_group_path_collisions as shell

PROVISIONERS = [shell.MUTATORS[0][0], shell.MUTATORS[1][0], shell.HELM]

# Reuse credential-file parsing and request logging, replacing only the HTTP peer.
CURL_STUB = (
    shell.CURL_STUB.split("status = 200\n", 1)[0]
    + r"""
state_path = Path(os.environ["API_LOG"]).with_suffix(".state.json")
state = json.loads(state_path.read_text()) if state_path.exists() else json.loads(os.environ["REGISTRATION_STATE"])
scenario = os.environ["GROUP_SCENARIO"]
status, body = 200, {}
base = "/admin/realms/mcp-gateway"
if path.endswith("/protocol/openid-connect/token"):
    body = {"access_token": "fixture-token", "expires_in": 60}
elif path == "/admin/serverinfo":
    body = {"systemInfo": {"version": "26.8.2"}, "features": [{"name": "CIMD", "enabled": True}],
            "providers": {"client-registration": {"providers": {"openid-connect": {}}}},
            "componentTypes": {
                "org.keycloak.services.clientpolicy.condition.ClientPolicyConditionProvider": [{"id": name} for name in ("client-scopes", "client-roles", "client-id-uri", "client-access-type", "client-type", "client-attributes", "grant-type")],
                "org.keycloak.services.clientpolicy.executor.ClientPolicyExecutorProvider": [{"id": name} for name in ("reject-request", "client-id-metadata-document", "pkce-enforcer", "secure-client-uris-pattern")],
                "org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy": [{"id": "default-roles"}]}}
    if scenario == "native-policy-missing":
        body["componentTypes"]["org.keycloak.services.clientpolicy.executor.ClientPolicyExecutorProvider"] = []
    if scenario == "old-server":
        body["componentTypes"]["org.keycloak.services.clientpolicy.executor.ClientPolicyExecutorProvider"] = []
        body["systemInfo"]["version"] = "26.2.4"
    if scenario == "default-roles-missing":
        body["componentTypes"]["org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy"] = []
elif path == base:
    body = {"id": "realm-id"}
elif path.startswith(base + "/client-policies/"):
    key = path.rsplit("/", 1)[1]
    if method == "PUT":
        status = 204
        if scenario != "policy-readback-failed":
            state[key] = json.loads(body_arg)[key]
            if scenario == "policy-readback-expanded" and key == "policies":
                for policy in state[key]:
                    if policy["name"] == "mcp-registration-scope-guard-policy":
                        policy["conditions"][0]["configuration"]["scopes"].append("basic")
        metadata_name = "cimd-vendors" if key == "profiles" else "cimd-vendors-policy"
        metadata = next((entry for entry in json.loads(body_arg)[key] if entry["name"] == metadata_name), None)
        if metadata and (metadata.get("executors") or metadata.get("conditions")):
            if scenario == "metadata-" + key + "-write-error":
                state[key] = [entry for entry in state[key] if entry["name"] != metadata_name]
                status = 500
            elif scenario == "metadata-" + key + "-readback-missing":
                state[key] = [entry for entry in state[key] if entry["name"] != metadata_name]
    body = {key: state[key]}
elif path == base + "/client-scopes":
    body = state["catalog"]
    if scenario in ("is.parameterized.scope", "is.dynamic.scope"):
        body = [dict(scope, attributes={scenario: "true"}) for scope in body]
elif path.startswith(base + "/default-default-client-scopes") or path.startswith(base + "/default-optional-client-scopes"):
    parts = path.removeprefix(base + "/").split("/")
    if method == "DELETE":
        status = 204
        if scenario != "default-readback-failed":
            state[parts[0]] = [scope for scope in state[parts[0]] if scope["id"] != parts[1]]
    body = state[parts[0]]
elif path == base + "/clients":
    first = int(query.get("first", [0])[0])
    maximum = int(query.get("max", [100])[0])
    state["client_inventories"] += first == 0
    if scenario == "late-cimd" and state["client_inventories"] == 2 and first == 0:
        state["clients"]["late"] = {"id": "late", "clientId": "https://withdrawn.example/late.json", "enabled": True, "attributes": {"cimd.cache.expiry.time.in.sec": "300"}}
    body = list(state["clients"].values())[first:first + maximum]
elif path.startswith(base + "/clients/"):
    parts = path.removeprefix(base + "/clients/").split("/")
    client = state["clients"][parts[0]]
    if len(parts) == 1:
        if method == "PUT":
            status = 204
            if scenario != "client-readback-failed":
                client.update(json.loads(body_arg))
        body = client
    elif parts[1] == "registration-access-token":
        client["registrationAccessToken"] = "rotated-token"
        body = {"registrationAccessToken": "rotated-token"}
    elif parts[1] == "roles":
        roles = state["roles"].setdefault(parts[0], {})
        if method == "POST":
            status = 201
            role = dict(json.loads(body_arg), clientRole=True, composite=False)
            if scenario != "role-readback-failed":
                roles[role["name"]] = role
        elif method == "DELETE":
            status = 204
            if scenario != "role-readback-failed":
                roles.pop(parts[2], None)
        else:
            status = 200 if parts[2] in roles else 404
            body = roles.get(parts[2], {})
    else:
        key = parts[1]
        if method == "DELETE":
            status = 204
            if scenario != "scope-readback-failed":
                state["scopes"][parts[0]][key] = [scope for scope in state["scopes"][parts[0]][key] if scope["id"] != parts[2]]
        body = state["scopes"].get(parts[0], {}).get(key, [])
elif path == base + "/components":
    if method == "POST":
        status = 201
        state["components"].append(dict(json.loads(body_arg), id="component-" + str(len(state["components"]))))
    body = state["components"]
elif path.startswith(base + "/components/") and method == "PUT":
    status = 204
    replacement = json.loads(body_arg)
    state["components"] = [replacement if c["id"] == replacement["id"] else c for c in state["components"]]
else:
    print("Unexpected API request: " + path, file=sys.stderr)
    sys.exit(97)
state_path.write_text(json.dumps(state))
if option("-o") != "/dev/null":
    sys.stdout.write(json.dumps(body, separators=(",", ":")))
sys.stdout.write(option("-w").replace(r"\n", "\n").replace("%{http_code}", str(status)))
"""
)


def client(uuid, client_id, **fields):
    return {"id": uuid, "clientId": client_id, "enabled": True, **fields}


def run(
    tmp_path, monkeypatch, path, command, clients=(), scopes=None, scenario="supported", roles=None
):
    state = {
        "clients": {c["id"]: c for c in clients},
        "scopes": scopes or {},
        "profiles": [
            {"name": "unrelated-profile"},
            {"name": "cimd-vendors"},
            {
                "name": "mcp-cimd-runtime-guard",
                "executors": [{"executor": "reject-request", "configuration": {}}],
            },
        ],
        "policies": [
            {"name": "unrelated-policy"},
            {"name": "cimd-vendors-policy"},
            {
                "name": "mcp-cimd-runtime-guard-policy",
                "enabled": True,
                "mode": "STRICT",
                "profiles": ["mcp-cimd-runtime-guard"],
                "conditions": [],
            },
        ],
        "client_inventories": 0,
        "components": [],
        "roles": roles or {},
        "catalog": [
            {"id": name.replace("/", "-"), "name": name}
            for name in ("basic", "profile", "email", "phone", "kc-admin-api", "mcp-servers/read")
        ],
        "default-default-client-scopes": [
            {"id": name, "name": name} for name in ("basic", "profile", "kc-admin-api")
        ],
        "default-optional-client-scopes": [
            {"id": name.replace("/", "-"), "name": name}
            for name in ("email", "phone", "mcp-servers/read")
        ],
    }
    monkeypatch.setenv("REGISTRATION_STATE", json.dumps(state))
    monkeypatch.setattr(shell, "CURL_STUB", CURL_STUB)
    result, requests = shell._run_shell(tmp_path, path, command, scenario)
    persisted = tmp_path / "api.state.json"
    return result, requests, json.loads(persisted.read_text()) if persisted.exists() else state


def cimd(uuid="cimd", host="vendor.example", **fields):
    return client(
        uuid,
        f"https://{host}/client.json",
        attributes={"cimd.cache.expiry.time.in.sec": "300"},
        **fields,
    )


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize("dcr", ["false", "true"])
def test_cimd_off_revokes_even_when_dcr_on_or_kept(tmp_path, monkeypatch, path, dcr):
    record = cimd()
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        f"ENABLE_DCR_CONFIG={dcr}\nENABLE_CIMD_CONFIG=false\nDCR_SWEEP_ACKNOWLEDGED=true\n"
        f'DCR_KEEP_CLIENTS={record["clientId"]}\nremove_cimd_client_policy "$TOKEN"',
        [record, client("ordinary", "https://vendor.example/ordinary")],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert state["clients"]["cimd"]["enabled"] is False
    assert state["clients"]["cimd"]["notBefore"] > 0
    assert state["clients"]["ordinary"]["enabled"] is True
    assert not any("logout" in r["path"] or "sessions" in r["path"] for r in requests)
    assert {p["name"] for p in state["policies"]} == {
        "unrelated-policy",
        "mcp-registration-scope-guard-policy",
        "mcp-public-pkce-policy",
        "mcp-no-remote-request-objects",
        "mcp-cimd-runtime-guard-policy",
    }
    assert {p["name"] for p in state["profiles"]} == {
        "unrelated-profile",
        "mcp-registration-scope-guard",
        "mcp-public-pkce",
        "mcp-no-remote-request-objects",
        "mcp-cimd-runtime-guard",
    }


@pytest.mark.parametrize("path", PROVISIONERS)
def test_cimd_trust_withdrawal_is_paginated_and_does_not_reenable(tmp_path, monkeypatch, path):
    clients = [client(f"filler-{i}", f"filler-{i}") for i in range(100)]
    clients += [
        cimd("withdrawn", "old.example"),
        cimd("kept", "vendor.example"),
        cimd("disabled", "sub.vendor.example", enabled=False),
    ]
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        "ENABLE_DCR_CONFIG=true\nENABLE_CIMD_CONFIG=true\nDCR_SWEEP_ACKNOWLEDGED=true\n"
        'CIMD_TRUSTED_DOMAINS=\'["vendor.example"]\'\nreconcile_cimd_clients "$TOKEN"',
        clients,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert state["clients"]["withdrawn"]["enabled"] is False
    assert state["clients"]["kept"]["enabled"] is True
    assert state["clients"]["disabled"]["enabled"] is False
    assert [r["query"]["first"] for r in requests if r["path"].endswith("/clients")] == [
        ["0"],
        ["100"],
    ]
    assert state["clients"]["disabled"]["notBefore"] > 0


@pytest.mark.parametrize("path", PROVISIONERS)
def test_cimd_revocation_requires_review_even_with_keep(tmp_path, monkeypatch, path):
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        'ENABLE_DCR_CONFIG=true\nDCR_KEEP_CLIENTS=https://vendor.example/client.json\nreconcile_cimd_clients "$TOKEN"',
        [cimd()],
    )
    assert result.returncode != 0
    assert "DCR_SWEEP_ACKNOWLEDGED=true" in result.stderr
    assert state["clients"]["cimd"]["enabled"] is True
    assert not any(r["method"] == "PUT" for r in requests)


@pytest.mark.parametrize("path", PROVISIONERS)
def test_cimd_cleanup_refuses_without_native_policy_support(tmp_path, monkeypatch, path):
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        'ENABLE_DCR_CONFIG=false\nDCR_SWEEP_ACKNOWLEDGED=true\nremove_cimd_client_policy "$TOKEN"',
        [cimd()],
        scenario="old-server",
    )
    assert result.returncode != 0
    assert state["clients"]["cimd"]["enabled"] is True
    assert any(p["name"] == "cimd-vendors-policy" for p in state["policies"])
    assert not any(r["method"] in ("PUT", "DELETE") and "/clients/" in r["path"] for r in requests)


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize("flag", ["ENABLE_DCR_CONFIG", "ENABLE_CIMD_CONFIG"])
def test_native_providers_are_required_before_enabling(tmp_path, monkeypatch, path, flag):
    result, requests, _ = run(
        tmp_path,
        monkeypatch,
        path,
        f'{flag}=true\npreflight_dcr_upgrade "$TOKEN"',
        scenario="native-policy-missing",
    )
    assert result.returncode != 0
    assert not any(r["method"] == "PUT" for r in requests)


@pytest.mark.parametrize("path", PROVISIONERS)
def test_simultaneous_dcr_cimd_is_refused_before_mutation(tmp_path, monkeypatch, path):
    result, requests, _ = run(
        tmp_path,
        monkeypatch,
        path,
        'ENABLE_DCR_CONFIG=true\nENABLE_CIMD_CONFIG=true\npreflight_dcr_upgrade "$TOKEN"',
    )
    assert result.returncode != 0
    assert "Enable DCR or CIMD, not both" in result.stderr
    assert not any(r["method"] in ("POST", "PUT", "DELETE") for r in requests)


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize(
    "attribute",
    [
        "request.uris",
        "jwks.url",
        "sectorIdentifierUri",
        "backchannel.logout.url",
        "ciba.backchannel.client.notification.endpoint",
    ],
)
def test_dcr_guard_refuses_enabled_clients_with_server_fetch_urls(
    tmp_path, monkeypatch, path, attribute
):
    record = client("old-client", "old-client", attributes={attribute: "http://localhost/private"})
    result, _, state = run(
        tmp_path,
        monkeypatch,
        path,
        'ENABLE_DCR_CONFIG=true\nENABLE_CIMD_CONFIG=false\nconfigure_registration_security_policy "$TOKEN"',
        [record],
    )
    assert result.returncode != 0
    assert state["clients"]["old-client"]["enabled"] is True


@pytest.mark.parametrize("path", PROVISIONERS)
def test_cimd_only_removes_dcr_uri_guard_without_removing_unrelated_policies(
    tmp_path, monkeypatch, path
):
    result, _, state = run(
        tmp_path,
        monkeypatch,
        path,
        "ENABLE_DCR_CONFIG=false\nENABLE_CIMD_CONFIG=true\nDCR_SWEEP_ACKNOWLEDGED=true\n"
        "CIMD_TRUSTED_DOMAINS='[\"vendor.example\"]'\n"
        'configure_cimd_client_policy "$TOKEN"',
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(p["name"] == "mcp-no-remote-request-objects" for p in state["policies"])
    assert not any(p["name"] == "mcp-no-remote-request-objects" for p in state["profiles"])
    assert any(p["name"] == "unrelated-policy" for p in state["policies"])


@pytest.mark.parametrize("path", PROVISIONERS)
def test_dcr_only_guards_each_server_fetch_url_field(tmp_path, monkeypatch, path):
    result, _, state = run(
        tmp_path,
        monkeypatch,
        path,
        'ENABLE_DCR_CONFIG=true\nENABLE_CIMD_CONFIG=false\nconfigure_registration_security_policy "$TOKEN"',
    )
    assert result.returncode == 0, result.stdout + result.stderr
    profile = next(p for p in state["profiles"] if p["name"] == "mcp-no-remote-request-objects")
    config = profile["executors"][0]["configuration"]
    assert config["allowed-patterns"] == []
    assert config["client-uri-fields"] == [
        "requestUris",
        "jwksUri",
        "sectorIdentifierUri",
        "backchannelLogoutUrl",
        "cibaClientNotificationEndpoint",
    ]


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize("scenario", ["policy-readback-failed", "policy-readback-expanded"])
def test_guard_readback_must_match_exact_scopes(tmp_path, monkeypatch, path, scenario):
    result, requests, _ = run(
        tmp_path,
        monkeypatch,
        path,
        'ENABLE_DCR_CONFIG=true\nconfigure_dcr_trusted_hosts "$TOKEN"',
        scenario=scenario,
    )
    assert result.returncode != 0
    assert "readback failed" in result.stderr
    assert not any(r["method"] == "PUT" and "/components/" in r["path"] for r in requests)


@pytest.mark.parametrize("path", PROVISIONERS)
def test_guard_catalog_and_both_realm_defaults_follow_allowlist(tmp_path, monkeypatch, path):
    trusted = client("reviewed", "reviewed-machine")
    existing = {
        "reviewed": {"optional-client-scopes": [{"id": "kc-admin-api", "name": "kc-admin-api"}]}
    }
    result, _, state = run(
        tmp_path,
        monkeypatch,
        path,
        'ENABLE_DCR_CONFIG=false\nENABLE_CIMD_CONFIG=true\nDCR_ALLOWED_SCOPES="openid basic email"\n'
        'REGISTRATION_TRUSTED_CLIENT_IDS=\'["reviewed-machine"]\'\nconfigure_cimd_client_policy "$TOKEN"',
        [trusted],
        existing,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    profiles = {p["name"]: p for p in state["profiles"]}
    assert profiles["mcp-registration-scope-guard"]["executors"] == [
        {"executor": "reject-request", "configuration": {}}
    ]
    metadata = profiles["cimd-vendors"]["executors"][0]
    assert metadata["executor"] == "client-id-metadata-document"
    assert "allowed-scopes" not in metadata["configuration"]
    assert [s["name"] for s in state["default-default-client-scopes"]] == ["basic"]
    assert [s["name"] for s in state["default-optional-client-scopes"]] == ["email"]
    assert state["scopes"] == existing
    assert "mcp-registration-trusted" in state["roles"]["reviewed"]
    policy = next(
        p for p in state["policies"] if p["name"] == "mcp-registration-scope-guard-policy"
    )
    assert policy["mode"] == "STRICT"
    assert policy["conditions"] == [
        {
            "condition": "client-scopes",
            "configuration": {
                "type": "Any",
                "scopes": ["kc-admin-api", "mcp-servers/read", "phone", "profile"],
            },
        },
        {
            "condition": "client-roles",
            "configuration": {"roles": ["mcp-registration-trusted"], "is-negative-logic": True},
        },
    ]


@pytest.mark.parametrize("path", PROVISIONERS)
def test_existing_registered_scopes_are_narrowed_independent_of_keep_or_shape(
    tmp_path, monkeypatch, path
):
    clients = [
        cimd(
            "dcr",
            publicClient=False,
            serviceAccountsEnabled=True,
            clientAuthenticatorType="client-secret",
        ),
        cimd("managed", "second.example"),
        client("machine", "mcp-gateway-m2m", registrationAccessToken="rotated-by-setup"),
        client("reviewed", "operator-machine", registrationAccessToken="rotated-by-setup"),
    ]
    scopes = {
        c["id"]: {
            "default-client-scopes": [
                {"id": "basic", "name": "basic"},
                {"id": "profile", "name": "profile"},
            ],
            "optional-client-scopes": [{"id": "admin", "name": "kc-admin-api"}],
        }
        for c in clients
    }
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        'DCR_ALLOWED_SCOPES="openid basic"\nDCR_KEEP_CLIENTS=shape-attacker\n'
        'REGISTRATION_TRUSTED_CLIENT_IDS=\'["operator-machine"]\'\nreconcile_registered_client_scopes "$TOKEN"',
        clients,
        scopes,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for uuid in ("dcr", "managed"):
        assert state["scopes"][uuid] == {
            "default-client-scopes": [{"id": "basic", "name": "basic"}],
            "optional-client-scopes": [],
        }
    for uuid in ("machine", "reviewed"):
        assert state["scopes"][uuid] == scopes[uuid]
    assert all("/clients/" in r["path"] for r in requests if r["method"] == "DELETE")


@pytest.mark.parametrize("path", PROVISIONERS)
def test_unknown_machine_provenance_fails_with_inventory(tmp_path, monkeypatch, path):
    record = client(
        "unknown",
        "looks-like-machine",
        publicClient=False,
        serviceAccountsEnabled=True,
        clientAuthenticatorType="client-secret",
    )
    result, requests, _ = run(
        tmp_path,
        monkeypatch,
        path,
        'DCR_KEEP_CLIENTS=looks-like-machine\nreconcile_registered_client_scopes "$TOKEN"',
        [record],
        {"unknown": {"optional-client-scopes": [{"id": "admin", "name": "kc-admin-api"}]}},
    )
    assert result.returncode != 0
    assert "looks-like-machine" in result.stderr
    assert "REGISTRATION_TRUSTED_CLIENT_IDS" in result.stderr
    assert not any(r["method"] == "DELETE" for r in requests)


@pytest.mark.parametrize("path", PROVISIONERS)
def test_cimd_is_not_part_of_dcr_off_shape_sweep(tmp_path, monkeypatch, path):
    result, _, state = run(
        tmp_path,
        monkeypatch,
        path,
        'ENABLE_CIMD_CONFIG=true\nDCR_SWEEP_ACKNOWLEDGED=true\ndisable_unlisted_clients "$TOKEN"',
        [cimd()],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert state["clients"]["cimd"]["enabled"] is True
    assert "registrationAccessToken" not in state["clients"]["cimd"]


@pytest.mark.parametrize("path", PROVISIONERS)
def test_client_revocation_readback_failure_is_fatal(tmp_path, monkeypatch, path):
    result, _, _ = run(
        tmp_path,
        monkeypatch,
        path,
        'DCR_SWEEP_ACKNOWLEDGED=true\nreconcile_cimd_clients "$TOKEN"',
        [cimd()],
        scenario="client-readback-failed",
    )
    assert result.returncode != 0


@pytest.mark.parametrize("path", PROVISIONERS)
def test_scope_readback_failure_is_fatal(tmp_path, monkeypatch, path):
    result, _, _ = run(
        tmp_path,
        monkeypatch,
        path,
        'reconcile_registered_client_scopes "$TOKEN"',
        [cimd("dcr")],
        {"dcr": {"default-client-scopes": [{"id": "admin", "name": "kc-admin-api"}]}},
        scenario="scope-readback-failed",
    )
    assert result.returncode != 0


@pytest.mark.parametrize("path", PROVISIONERS)
def test_cimd_revocation_precedes_unknown_scope_failure(tmp_path, monkeypatch, path):
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        'ENABLE_DCR_CONFIG=false\nDCR_SWEEP_ACKNOWLEDGED=true\nreconcile_registration_security "$TOKEN"',
        [cimd(), client("unknown", "unknown-browser")],
        {"unknown": {"default-client-scopes": [{"id": "admin", "name": "kc-admin-api"}]}},
    )
    assert result.returncode != 0
    assert "cannot establish registration provenance" in result.stderr
    assert state["clients"]["cimd"]["enabled"] is False
    assert state["clients"]["unknown"]["enabled"] is False
    assert "registrationAccessToken" in state["clients"]["unknown"]
    assert not any("protocol-mappers" in request["path"] for request in requests)


@pytest.mark.parametrize("path", PROVISIONERS)
def test_malformed_managed_cimd_id_is_revoked(tmp_path, monkeypatch, path):
    record = client(
        "malformed",
        "http://vendor.example/client.json",
        attributes={"cimd.cache.expiry.time.in.sec": "300"},
    )
    result, _, state = run(
        tmp_path,
        monkeypatch,
        path,
        "ENABLE_CIMD_CONFIG=true\nCIMD_TRUSTED_DOMAINS='[\"vendor.example\"]'\n"
        'DCR_SWEEP_ACKNOWLEDGED=true\nreconcile_cimd_clients "$TOKEN"',
        [record],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert state["clients"]["malformed"]["enabled"] is False


@pytest.mark.parametrize("path", PROVISIONERS)
def test_managed_cimd_provenance_overrides_exact_scope_trust(tmp_path, monkeypatch, path):
    record = cimd()
    result, _, state = run(
        tmp_path,
        monkeypatch,
        path,
        "REGISTRATION_TRUSTED_CLIENT_IDS='[\"https://vendor.example/client.json\"]'\n"
        'DCR_ALLOWED_SCOPES="openid basic"\nreconcile_registered_client_scopes "$TOKEN"',
        [record],
        {"cimd": {"default-client-scopes": [{"id": "admin", "name": "kc-admin-api"}]}},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert state["scopes"]["cimd"]["default-client-scopes"] == []


@pytest.mark.parametrize("path", PROVISIONERS)
def test_reconciliation_is_idempotent_in_one_setup(tmp_path, monkeypatch, path):
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        "ENABLE_DCR_CONFIG=false\nDCR_SWEEP_ACKNOWLEDGED=true\n"
        'reconcile_registration_security "$TOKEN"\nreconcile_registration_security "$TOKEN"',
        [cimd()],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert state["clients"]["cimd"]["enabled"] is False
    assert (
        len([r for r in requests if r["method"] == "PUT" and r["path"].endswith("/clients/cimd")])
        == 1
    )


def test_terraform_delegates_cleanup_before_first_party_mutations():
    source = (shell.REPO_ROOT / "terraform/aws-ecs/scripts/init-keycloak.sh").read_text()
    main = source.split("main() {", 1)[1]
    assert main.index('bash "$DCR_SCRIPT" --reconcile-registration') < main.index(
        'harden_master_realm "$TOKEN"'
    )
    assert main.index('bash "$DCR_SCRIPT" --restrict-group-claims') < main.index(
        'create_clients "$TOKEN"'
    )
    assert (
        main.count('REGISTRATION_TRUSTED_CLIENT_IDS="${REGISTRATION_TRUSTED_CLIENT_IDS:-[]}"') == 3
    )


@pytest.mark.parametrize("path", PROVISIONERS)
def test_reserved_disabled_audience_drops_inherited_scopes(tmp_path, monkeypatch, path):
    record = client("audience", "mcp-gateway", enabled=False, bearerOnly=True)
    result, _, state = run(
        tmp_path,
        monkeypatch,
        path,
        'reconcile_registered_client_scopes "$TOKEN"',
        [record],
        {"audience": {"optional-client-scopes": [{"id": "phone", "name": "phone"}]}},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert state["scopes"]["audience"]["optional-client-scopes"] == []
    assert state["clients"]["audience"]["enabled"] is False


@pytest.mark.parametrize("path", PROVISIONERS)
def test_native_role_exception_is_exact_and_withdrawable(tmp_path, monkeypatch, path):
    role = {"name": "mcp-registration-trusted", "clientRole": True, "composite": False}
    records = [
        client("trusted", "reviewed-machine"),
        client("withdrawn", "formerly-reviewed"),
        client("shape", "shape-attacker", serviceAccountsEnabled=True, publicClient=False),
        cimd("metadata"),
    ]
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        'REGISTRATION_TRUSTED_CLIENT_IDS=\'["reviewed-machine", "https://vendor.example/client.json"]\'\n'
        'configure_registration_security_policy "$TOKEN"',
        records,
        roles={uuid: {role["name"]: role} for uuid in ("withdrawn", "shape", "metadata")},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert role["name"] in state["roles"]["trusted"]
    assert all(not state["roles"][uuid] for uuid in ("withdrawn", "shape", "metadata"))
    role_create = next(
        i for i, r in enumerate(requests) if r["method"] == "POST" and r["path"].endswith("/roles")
    )
    roles_disabled = [
        json.loads(r["body"])["subType"]
        for r in requests[:role_create]
        if r["method"] == "POST" and r["path"].endswith("/components")
    ]
    assert sorted(roles_disabled) == ["anonymous", "authenticated"]


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize("scenario", ["role-readback-failed", "default-readback-failed"])
def test_native_state_readback_failures_are_fatal(tmp_path, monkeypatch, path, scenario):
    result, _, _ = run(
        tmp_path,
        monkeypatch,
        path,
        'configure_registration_security_policy "$TOKEN"',
        [client("machine", "mcp-gateway-m2m")],
        scenario=scenario,
    )
    assert result.returncode != 0


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize("kind", ["default", "optional"])
def test_unmarked_historical_registration_requires_scope_review(tmp_path, monkeypatch, path, kind):
    result, requests, _ = run(
        tmp_path,
        monkeypatch,
        path,
        'DCR_KEEP_CLIENTS=legacy-dcr\nreconcile_registered_client_scopes "$TOKEN"',
        [client("legacy", "legacy-dcr")],
        {"legacy": {f"{kind}-client-scopes": [{"id": "phone", "name": "phone"}]}},
    )
    assert result.returncode != 0
    assert "legacy-dcr" in result.stderr and "REGISTRATION_TRUSTED_CLIENT_IDS" in result.stderr
    assert not any(r["method"] == "DELETE" for r in requests)


@pytest.mark.parametrize("path", PROVISIONERS)
def test_native_guard_refreshes_catalog_after_first_party_creation(tmp_path, path):
    source = (shell.REPO_ROOT / path).read_text()
    if path == shell.MUTATORS[1][0]:
        # Upgrade runs against an already-provisioned realm; Terraform calls it last.
        source = (shell.REPO_ROOT / "terraform/aws-ecs/scripts/init-keycloak.sh").read_text()
        main = source.split("main() {", 1)[1]
        assert main.index('create_clients "$TOKEN"') < main.index('bash "$DCR_SCRIPT";')
    else:
        main = source.split("main() {", 1)[1]
        assert main.index('create_scopes "$TOKEN"') < main.index(
            'configure_registration_security_policy "$TOKEN"'
        )


@pytest.mark.parametrize("path", PROVISIONERS)
def test_stock_server_with_registration_disabled_still_installs_guard(tmp_path, monkeypatch, path):
    result, _, state = run(
        tmp_path,
        monkeypatch,
        path,
        'ENABLE_DCR_CONFIG=false\nENABLE_CIMD_CONFIG=false\nreconcile_registration_security "$TOKEN"',
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert any(p["name"] == "mcp-registration-scope-guard-policy" for p in state["policies"])
    assert not any(p["name"] == "cimd-vendors-policy" for p in state["policies"])
    assert {s["name"] for s in state["default-default-client-scopes"]} == {"basic", "profile"}
    assert {s["name"] for s in state["default-optional-client-scopes"]} == {"email"}


@pytest.mark.parametrize("path", PROVISIONERS)
def test_off_state_missing_default_roles_skips_role_exceptions(tmp_path, monkeypatch, path):
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        'ENABLE_DCR_CONFIG=false\nENABLE_CIMD_CONFIG=false\nconfigure_registration_security_policy "$TOKEN"',
        [client("machine", "mcp-gateway-m2m")],
        scenario="default-roles-missing",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not state["roles"]
    assert not any(r["method"] == "POST" and not r["path"].endswith("/token") for r in requests)


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize("flag", ["ENABLE_DCR_CONFIG", "ENABLE_CIMD_CONFIG"])
def test_registration_requires_default_roles_before_scope_exception(
    tmp_path, monkeypatch, path, flag
):
    result, requests, _ = run(
        tmp_path,
        monkeypatch,
        path,
        f'{flag}=true\nconfigure_registration_security_policy "$TOKEN"',
        [client("machine", "mcp-gateway-m2m")],
        scenario="default-roles-missing",
    )
    assert result.returncode != 0
    assert not any(
        r["method"] in ("POST", "PUT", "DELETE") and not r["path"].endswith("/token")
        for r in requests
    )


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize("attribute", ["is.parameterized.scope", "is.dynamic.scope"])
@pytest.mark.parametrize("flag", ["ENABLE_DCR_CONFIG", "ENABLE_CIMD_CONFIG"])
def test_parameterized_scopes_refuse_registration_before_mutation(
    tmp_path, monkeypatch, path, attribute, flag
):
    result, requests, _ = run(
        tmp_path,
        monkeypatch,
        path,
        f'{flag}=true\npreflight_dcr_upgrade "$TOKEN"',
        scenario=attribute,
    )
    assert result.returncode != 0
    assert "ordinary client scopes" in result.stderr
    assert not any(
        r["method"] in ("POST", "PUT", "DELETE") and not r["path"].endswith("/token")
        for r in requests
    )


@pytest.mark.parametrize("path", PROVISIONERS)
def test_parameterized_scopes_do_not_block_off_state_cleanup(tmp_path, monkeypatch, path):
    result, _, state = run(
        tmp_path,
        monkeypatch,
        path,
        "ENABLE_DCR_CONFIG=false\nENABLE_CIMD_CONFIG=false\nDCR_SWEEP_ACKNOWLEDGED=true\n"
        'reconcile_registration_security "$TOKEN"',
        [cimd()],
        scenario="is.parameterized.scope",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert state["clients"]["cimd"]["enabled"] is False


@pytest.mark.parametrize("path", PROVISIONERS)
def test_scope_trust_does_not_mutate_protected_builtin_roles(tmp_path, monkeypatch, path):
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        'configure_registration_security_policy "$TOKEN"',
        [client("builtin", "realm-management"), client("machine", "mcp-gateway-m2m")],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any("/clients/builtin/roles" in request["path"] for request in requests)
    assert "mcp-registration-trusted" in state["roles"]["machine"]


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize("enabled", ["false", "true"])
def test_lifecycle_installs_guard_before_closing_creation_and_revoking_clients(
    tmp_path, monkeypatch, path, enabled
):
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        f"ENABLE_CIMD_CONFIG={enabled}\nCIMD_TRUSTED_DOMAINS='[\"vendor.example\"]'\n"
        'DCR_SWEEP_ACKNOWLEDGED=true\n_prepare_cimd_lifecycle "$TOKEN"',
        [cimd("withdrawn", "withdrawn.example")],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    writes = [(i, r) for i, r in enumerate(requests) if r["method"] == "PUT"]
    guard = next(
        i
        for i, r in writes
        if r["path"].endswith("/policies")
        and any(
            p["name"] == "mcp-cimd-runtime-guard-policy" and p.get("conditions")
            for p in json.loads(r["body"])["policies"]
        )
    )
    close = next(
        i
        for i, r in writes
        if r["path"].endswith("/policies")
        and not any(p["name"] == "cimd-vendors-policy" for p in json.loads(r["body"])["policies"])
    )
    revoke = next(i for i, r in writes if r["path"].endswith("/clients/withdrawn"))
    assert guard < close < revoke
    assert any(r["path"].endswith("/clients") for r in requests[:guard])
    assert state["clients"]["withdrawn"]["enabled"] is False
    runtime = next(p for p in state["policies"] if p["name"] == "mcp-cimd-runtime-guard-policy")
    assert len(runtime["conditions"]) == 2
    configuration = runtime["conditions"][1]["configuration"]
    assert not configuration.get("is-negative-logic", False)
    assert json.loads(configuration["attributes"]) == [
        {"key": "display.on.consent.screen", "value": "true"}
    ]
    assert not any(r["method"] == "DELETE" for r in requests)


@pytest.mark.parametrize("path", PROVISIONERS)
def test_lifecycle_refuses_unacknowledged_revocation_before_guard_mutation(
    tmp_path, monkeypatch, path
):
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        'ENABLE_CIMD_CONFIG=false\nDCR_SWEEP_ACKNOWLEDGED=false\n_prepare_cimd_lifecycle "$TOKEN"',
        [cimd()],
    )
    assert result.returncode != 0
    assert state["clients"]["cimd"]["enabled"] is True
    assert not any(r["method"] in ("PUT", "DELETE") for r in requests)


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize("ack", ["false", "true"])
def test_lifecycle_repeats_inventory_after_installing_guard(tmp_path, monkeypatch, path, ack):
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        f'ENABLE_CIMD_CONFIG=false\nDCR_SWEEP_ACKNOWLEDGED={ack}\n_prepare_cimd_lifecycle "$TOKEN"',
        scenario="late-cimd",
    )
    assert state["client_inventories"] >= 2
    assert (result.returncode == 0) == (ack == "true"), result.stdout + result.stderr
    assert state["clients"]["late"]["enabled"] == (ack == "false")
    assert any(
        p["name"] == "mcp-cimd-runtime-guard-policy" and p["conditions"] for p in state["policies"]
    )
    assert not any(p["name"] == "cimd-vendors-policy" for p in state["policies"])
    assert not any(r["method"] == "DELETE" for r in requests)


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize("collection", ["profiles", "policies"])
@pytest.mark.parametrize("failure", ["write-error", "readback-missing"])
def test_metadata_install_failure_retains_all_cimd_denial(
    tmp_path, monkeypatch, path, collection, failure
):
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        "ENABLE_CIMD_CONFIG=true\nCIMD_TRUSTED_DOMAINS='[\"vendor.example\"]'\n"
        'DCR_SWEEP_ACKNOWLEDGED=true\nconfigure_cimd_client_policy "$TOKEN"',
        [cimd()],
        scenario=f"metadata-{collection}-{failure}",
    )
    assert result.returncode != 0
    assert state["clients"]["cimd"]["enabled"] is True
    runtime = next(p for p in state["policies"] if p["name"] == "mcp-cimd-runtime-guard-policy")
    assert runtime["enabled"] is True
    assert runtime["mode"] == "STRICT"
    assert runtime["profiles"] == ["mcp-cimd-runtime-guard"]
    assert len(runtime["conditions"]) == 2
    condition = runtime["conditions"][1]
    assert condition["condition"] == "client-attributes"
    assert not condition["configuration"].get("is-negative-logic", False)
    assert json.loads(condition["configuration"]["attributes"]) == [
        {"key": "display.on.consent.screen", "value": "true"}
    ]
    assert any(
        r["method"] == "PUT"
        and any(
            p["name"] == ("cimd-vendors" if collection == "profiles" else "cimd-vendors-policy")
            and (p.get("executors") or p.get("conditions"))
            for p in json.loads(r["body"])[collection]
        )
        for r in requests
        if r["path"].endswith("/" + collection)
    )
