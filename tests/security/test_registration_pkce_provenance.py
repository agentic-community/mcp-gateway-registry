"""Exercise native registration PKCE and CIMD provenance through the Admin API."""

import json
import shutil

import pytest

from tests.security import test_group_path_collisions as shell
from tests.security.test_registration_reconciliation import PROVISIONERS, cimd, client

REALM_PATH = "/admin/realms/mcp-gateway"
MARKER = "mcpgw.cimd.provenance.v1"
CONDITION_TYPE = "org.keycloak.services.clientpolicy.condition.ClientPolicyConditionProvider"
EXECUTOR_TYPE = "org.keycloak.services.clientpolicy.executor.ClientPolicyExecutorProvider"

# Keep the shared curl argument parser, credential handling, and request log.
CURL_STUB = (
    shell.CURL_STUB.split("status = 200\n", 1)[0]
    + r"""
state_path = Path(os.environ["API_LOG"]).with_suffix(".state.json")
state = json.loads(state_path.read_text()) if state_path.exists() else json.loads(os.environ["PKCE_PROVENANCE_STATE"])
scenario = os.environ["GROUP_SCENARIO"]
base = "/admin/realms/mcp-gateway"
status, body, raw = 200, {}, None
if path.endswith("/protocol/openid-connect/token"):
    body = {"access_token": "fixture-token", "expires_in": 60}
elif path == "/admin/serverinfo":
    body = state["serverinfo"]
    if scenario == "serverinfo-http-error":
        status = 503
    elif scenario == "serverinfo-invalid-json":
        raw = "{invalid"
elif path.startswith(base + "/client-policies/"):
    key = path.rsplit("/", 1)[1]
    if key not in ("profiles", "policies"):
        sys.exit(97)
    if method == "PUT":
        status = 204
        if scenario == key + "-write-error":
            status = 500
        elif scenario != key + "-readback-missing":
            state[key] = json.loads(body_arg)[key]
            if scenario == key + "-readback-corrupt":
                for entry in state[key]:
                    if entry["name"] == "mcp-public-pkce":
                        entry["executors"][0]["configuration"]["auto-configure"] = False
                    if entry["name"] == "mcp-public-pkce-policy":
                        entry["mode"] = "LENIENT"
        state["written"].append(key)
    body = {key: state[key]}
    if method == "GET" and key in state["written"] and scenario == key + "-readback-http-error":
        status = 503
elif path == base:
    if method == "PUT":
        status = 204
        if scenario == "marker-write-error":
            status = 500
        elif scenario != "marker-readback-missing":
            state["realm"].update(json.loads(body_arg))
    body = state["realm"]
    if method == "GET" and scenario == "realm-read-error":
        status = 503
elif path == base + "/clients":
    first = int(query.get("first", [0])[0])
    maximum = int(query.get("max", [100])[0])
    body = list(state["clients"].values())[first:first + maximum]
    if scenario == "client-list-error":
        status = 503
    elif scenario == "client-list-malformed":
        body = {"clients": body}
    elif scenario == "client-later-page-error" and first > 0:
        status = 503
elif path.startswith(base + "/clients/"):
    from urllib.parse import unquote
    uuid = unquote(path.removeprefix(base + "/clients/"))
    body = state["clients"].get(uuid, {})
    status = 200 if uuid in state["clients"] else 404
    if method == "DELETE":
        if scenario == "client-delete-error" and uuid == "second":
            status = 500
        elif scenario == "client-delete-readback-failed":
            status = 204
        else:
            state["clients"].pop(uuid, None)
            status, body = 204, {}
else:
    print("Unexpected API request: " + path, file=sys.stderr)
    sys.exit(97)
state_path.write_text(json.dumps(state))
if option("-o") != "/dev/null":
    sys.stdout.write(raw if raw is not None else json.dumps(body, separators=(",", ":")))
sys.stdout.write(option("-w").replace(r"\n", "\n").replace("%{http_code}", str(status)))
"""
)


@pytest.fixture(autouse=True)
def require_jq():
    if shutil.which("jq") is None:
        pytest.skip("jq is not installed")


def serverinfo():
    return {
        "systemInfo": {"version": "26.8.2"},
        "features": [{"name": "CIMD", "enabled": True}],
        "providers": {
            "client-registration": {
                "providers": {"openid-connect": {}, "install": {}, "saml2-entity-descriptor": {}}
            }
        },
        "componentTypes": {
            CONDITION_TYPE: [
                {"id": name}
                for name in (
                    "client-access-type",
                    "client-type",
                    "client-scopes",
                    "client-roles",
                    "client-id-uri",
                )
            ],
            EXECUTOR_TYPE: [
                {"id": name}
                for name in ("pkce-enforcer", "reject-request", "client-id-metadata-document")
            ],
            "org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy": [
                {"id": "default-roles"}
            ],
        },
    }


def run(
    tmp_path,
    monkeypatch,
    path,
    command,
    *,
    clients=(),
    attributes=None,
    info=None,
    scenario="supported",
    profiles=None,
    policies=None,
):
    state = {
        "serverinfo": serverinfo() if info is None else info,
        "realm": {
            "id": "realm-id",
            "realm": "mcp-gateway",
            "attributes": {"unrelated-setting": "preserve", **(attributes or {})},
        },
        "clients": {entry["id"]: entry for entry in clients},
        "profiles": [{"name": "unrelated-profile"}] if profiles is None else profiles,
        "policies": [{"name": "unrelated-policy"}] if policies is None else policies,
        "written": [],
    }
    monkeypatch.setenv("PKCE_PROVENANCE_STATE", json.dumps(state))
    monkeypatch.setattr(shell, "CURL_STUB", CURL_STUB)
    result, requests = shell._run_shell(
        tmp_path,
        path,
        "ENABLE_CIMD_CONFIG=true\nDCR_SWEEP_ACKNOWLEDGED=false\n" + command,
        scenario,
    )
    persisted = tmp_path / "api.state.json"
    return result, requests, json.loads(persisted.read_text()) if persisted.exists() else state


def mutations(requests):
    return [
        request
        for request in requests
        if request["method"] in ("POST", "PUT", "DELETE", "PATCH")
        and not request["path"].endswith("/protocol/openid-connect/token")
    ]


@pytest.mark.parametrize("path", PROVISIONERS)
def test_oidc_registration_provider_inventory_is_read_only(tmp_path, monkeypatch, path):
    result, requests, _ = run(
        tmp_path, monkeypatch, path, '_require_oidc_registration_only "$TOKEN"'
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert any(request["path"] == "/admin/serverinfo" for request in requests)
    assert mutations(requests) == []


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize(
    "inventory",
    [
        {"default": {}, "openid-connect": {}},
        {"default": None, "openid-connect": {}},
        {"install": {}},
        {},
        None,
        [],
        ["openid-connect"],
        "openid-connect",
    ],
)
def test_oidc_registration_rejects_unsafe_or_malformed_inventory(
    tmp_path, monkeypatch, path, inventory
):
    info = serverinfo()
    info["providers"]["client-registration"]["providers"] = inventory
    result, requests, _ = run(
        tmp_path, monkeypatch, path, '_require_oidc_registration_only "$TOKEN"', info=info
    )
    assert result.returncode != 0
    assert mutations(requests) == []


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize(
    "info", [{}, {"providers": {}}, {"providers": {"client-registration": {}}}]
)
def test_oidc_registration_rejects_absent_inventory(tmp_path, monkeypatch, path, info):
    result, requests, _ = run(
        tmp_path, monkeypatch, path, '_require_oidc_registration_only "$TOKEN"', info=info
    )
    assert result.returncode != 0
    assert mutations(requests) == []


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize(
    "helper",
    [
        "_require_oidc_registration_only",
        "_preflight_registration_security",
        "_reset_untrusted_cimd_cache",
    ],
)
@pytest.mark.parametrize("scenario", ["serverinfo-http-error", "serverinfo-invalid-json"])
def test_registration_serverinfo_failure_prevents_mutation(
    tmp_path, monkeypatch, path, helper, scenario
):
    result, requests, _ = run(tmp_path, monkeypatch, path, f'{helper} "$TOKEN"', scenario=scenario)
    assert result.returncode != 0
    assert mutations(requests) == []


@pytest.mark.parametrize("path", PROVISIONERS)
def test_pkce_policy_requires_public_oidc_s256_and_preserves_unrelated_entries(
    tmp_path, monkeypatch, path
):
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        '_configure_public_pkce_policy "$TOKEN" && _configure_public_pkce_policy "$TOKEN"',
        profiles=[{"name": "unrelated-profile"}, {"name": "mcp-public-pkce", "executors": []}],
        policies=[
            {"name": "unrelated-policy"},
            {"name": "mcp-public-pkce-policy", "enabled": False},
        ],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert {"name": "unrelated-profile"} in state["profiles"]
    assert {"name": "unrelated-policy"} in state["policies"]
    profiles = [entry for entry in state["profiles"] if entry["name"] == "mcp-public-pkce"]
    policies = [entry for entry in state["policies"] if entry["name"] == "mcp-public-pkce-policy"]
    assert len(profiles) == len(policies) == 1
    assert profiles[0]["executors"] == [
        {"executor": "pkce-enforcer", "configuration": {"auto-configure": True}}
    ]
    policy = policies[0]
    assert policy["enabled"] is True
    assert policy["mode"] == "STRICT"
    assert policy["profiles"] == ["mcp-public-pkce"]
    assert len(policy["conditions"]) == 2
    assert {entry["condition"]: entry["configuration"] for entry in policy["conditions"]} == {
        "client-access-type": {"type": ["public"]},
        "client-type": {"protocol": "openid-connect"},
    }
    for key in ("profiles", "policies"):
        endpoint = f"{REALM_PATH}/client-policies/{key}"
        writes = [
            i
            for i, request in enumerate(requests)
            if request["path"] == endpoint and request["method"] == "PUT"
        ]
        assert writes
        assert any(
            request["path"] == endpoint and request["method"] == "GET"
            for request in requests[writes[-1] + 1 :]
        )


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize(
    ("kind", "missing"),
    [
        (CONDITION_TYPE, "client-access-type"),
        (CONDITION_TYPE, "client-type"),
        (EXECUTOR_TYPE, "pkce-enforcer"),
    ],
)
def test_pkce_missing_native_capability_prevents_mutation(
    tmp_path, monkeypatch, path, kind, missing
):
    info = serverinfo()
    info["componentTypes"][kind] = [
        item for item in info["componentTypes"][kind] if item["id"] != missing
    ]
    result, requests, _ = run(
        tmp_path, monkeypatch, path, '_preflight_registration_security "$TOKEN"', info=info
    )
    assert result.returncode != 0
    assert mutations(requests) == []


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize("collection", ["profiles", "policies"])
@pytest.mark.parametrize(
    "failure", ["write-error", "readback-missing", "readback-corrupt", "readback-http-error"]
)
def test_pkce_write_and_readback_failures_are_fatal(
    tmp_path, monkeypatch, path, collection, failure
):
    result, requests, _ = run(
        tmp_path,
        monkeypatch,
        path,
        '_configure_public_pkce_policy "$TOKEN"',
        scenario=f"{collection}-{failure}",
    )
    assert result.returncode != 0
    if collection == "profiles":
        assert not any(
            request["method"] == "PUT" and request["path"].endswith("/policies")
            for request in requests
        )


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize("enabled", [True, False])
def test_untrusted_cimd_reset_requires_ack_for_every_cached_client(
    tmp_path, monkeypatch, path, enabled
):
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        '_reset_untrusted_cimd_cache "$TOKEN"',
        clients=[cimd(enabled=enabled)],
    )
    assert result.returncode != 0
    assert mutations(requests) == []
    assert MARKER not in state["realm"]["attributes"]
    assert "cimd" in state["clients"]


@pytest.mark.parametrize("path", PROVISIONERS)
def test_untrusted_cimd_reset_deletes_all_marked_clients_before_receipt(
    tmp_path, monkeypatch, path
):
    ordinary = [client(f"ordinary-{i}", f"ordinary-{i}") for i in range(101)]
    ordinary += [
        client("https-ordinary", "https://vendor.example/ordinary.json"),
        client("first-party", "mcp-gateway-web", publicClient=True),
    ]
    marked = [
        cimd("first", host="claude.ai"),
        cimd("second"),
        client("forged", "non-url-forged-client", attributes={"cimd.cache.expiry.time.in.sec": ""}),
    ]
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        "DCR_SWEEP_ACKNOWLEDGED=true\nDCR_KEEP_CLIENTS=all\n"
        'CIMD_TRUSTED_DOMAINS=\'["claude.ai"]\'\n_reset_untrusted_cimd_cache "$TOKEN"',
        clients=ordinary + marked,
        policies=[{"name": "unrelated-policy"}, {"name": "cimd-vendors-policy", "enabled": True}],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert state["clients"] == {entry["id"]: entry for entry in ordinary}
    assert state["realm"]["attributes"] == {"unrelated-setting": "preserve", MARKER: "true"}
    deletes = [(i, request) for i, request in enumerate(requests) if request["method"] == "DELETE"]
    assert {request["path"] for _, request in deletes} == {
        f"{REALM_PATH}/clients/{entry['id']}" for entry in marked
    }
    marker_writes = [
        i
        for i, request in enumerate(requests)
        if request["method"] == "PUT" and request["path"] == REALM_PATH
    ]
    assert len(marker_writes) == 1
    assert max(i for i, _ in deletes) < marker_writes[0]
    assert state["policies"] == [{"name": "unrelated-policy"}]
    close_policy = next(
        i
        for i, request in enumerate(requests)
        if request["method"] == "PUT" and request["path"].endswith("/policies")
    )
    assert close_policy < min(i for i, _ in deletes)
    assert any(
        int(request["query"].get("first", [0])[0]) > 0
        for request in requests
        if request["path"] == REALM_PATH + "/clients"
    )


@pytest.mark.parametrize("path", PROVISIONERS)
def test_untrusted_cimd_empty_reset_sets_receipt_without_ack(tmp_path, monkeypatch, path):
    record = client("ordinary", "ordinary", attributes={"another": "attribute"})
    result, requests, state = run(
        tmp_path, monkeypatch, path, '_reset_untrusted_cimd_cache "$TOKEN"', clients=[record]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert state["realm"]["attributes"][MARKER] == "true"
    assert state["clients"] == {"ordinary": record}
    assert not any(request["method"] == "DELETE" for request in requests)


@pytest.mark.parametrize("path", PROVISIONERS)
def test_provenance_receipt_preserves_new_native_cimd_cache(tmp_path, monkeypatch, path):
    record = cimd()
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        '_reset_untrusted_cimd_cache "$TOKEN" && _reset_untrusted_cimd_cache "$TOKEN"',
        clients=[record],
        attributes={MARKER: "true"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert state["clients"] == {"cimd": record}
    assert mutations(requests) == []


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize("marker", ["false", "", "TRUE"])
def test_incomplete_provenance_receipt_does_not_bypass_reset(tmp_path, monkeypatch, path, marker):
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        '_reset_untrusted_cimd_cache "$TOKEN"',
        clients=[cimd()],
        attributes={MARKER: marker},
    )
    assert result.returncode != 0
    assert mutations(requests) == []
    assert state["realm"]["attributes"][MARKER] == marker


@pytest.mark.parametrize("path", PROVISIONERS)
def test_acknowledged_disabled_cimd_record_requires_admin_review(tmp_path, monkeypatch, path):
    record = cimd(enabled=False)
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        'DCR_SWEEP_ACKNOWLEDGED=true\n_reset_untrusted_cimd_cache "$TOKEN"',
        clients=[record],
    )
    assert result.returncode != 0
    assert "administrator review" in result.stdout + result.stderr
    assert not any(request["method"] == "DELETE" for request in requests)
    assert MARKER not in state["realm"]["attributes"]
    assert state["clients"] == {"cimd": record}


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize(
    "scenario",
    ["realm-read-error", "client-list-error", "client-list-malformed", "client-later-page-error"],
)
def test_untrusted_cimd_inventory_failure_prevents_deletion_and_receipt(
    tmp_path, monkeypatch, path, scenario
):
    records = [cimd()] + [client(f"ordinary-{i}", f"ordinary-{i}") for i in range(101)]
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        'DCR_SWEEP_ACKNOWLEDGED=true\n_reset_untrusted_cimd_cache "$TOKEN"',
        clients=records,
        scenario=scenario,
    )
    assert result.returncode != 0
    assert mutations(requests) == []
    assert MARKER not in state["realm"]["attributes"]
    assert state["clients"] == {record["id"]: record for record in records}


@pytest.mark.parametrize("path", PROVISIONERS)
def test_failed_cimd_delete_never_sets_provenance_receipt(tmp_path, monkeypatch, path):
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        'DCR_SWEEP_ACKNOWLEDGED=true\n_reset_untrusted_cimd_cache "$TOKEN"',
        clients=[cimd("first"), cimd("second"), client("ordinary", "ordinary")],
        scenario="client-delete-error",
    )
    assert result.returncode != 0
    assert "first" not in state["clients"]
    assert set(state["clients"]) == {"second", "ordinary"}
    assert MARKER not in state["realm"]["attributes"]
    assert not any(
        request["method"] == "PUT" and request["path"] == REALM_PATH for request in requests
    )


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize("scenario", ["marker-write-error", "marker-readback-missing"])
def test_failed_cimd_receipt_write_or_readback_is_fatal(tmp_path, monkeypatch, path, scenario):
    result, _, state = run(
        tmp_path,
        monkeypatch,
        path,
        'DCR_SWEEP_ACKNOWLEDGED=true\n_reset_untrusted_cimd_cache "$TOKEN"',
        clients=[cimd()],
        scenario=scenario,
    )
    assert result.returncode != 0
    assert state["clients"] == {}
    assert MARKER not in state["realm"]["attributes"]


@pytest.mark.parametrize("path", PROVISIONERS)
def test_cimd_delete_readback_requires_absent_client(tmp_path, monkeypatch, path):
    record = cimd()
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        'DCR_SWEEP_ACKNOWLEDGED=true\n_reset_untrusted_cimd_cache "$TOKEN"',
        clients=[record],
        scenario="client-delete-readback-failed",
    )
    assert result.returncode != 0
    assert state["clients"] == {"cimd": record}
    assert MARKER not in state["realm"]["attributes"]
    assert not any(
        request["method"] == "PUT" and request["path"] == REALM_PATH for request in requests
    )


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize("failure", ["write-error", "readback-missing", "readback-http-error"])
def test_cimd_creation_must_close_before_cache_deletion(tmp_path, monkeypatch, path, failure):
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        'DCR_SWEEP_ACKNOWLEDGED=true\n_reset_untrusted_cimd_cache "$TOKEN"',
        clients=[cimd()],
        policies=[{"name": "cimd-vendors-policy", "enabled": True}],
        scenario=f"policies-{failure}",
    )
    assert result.returncode != 0
    assert "cimd" in state["clients"]
    assert MARKER not in state["realm"]["attributes"]
    assert not any(request["method"] == "DELETE" for request in requests)
