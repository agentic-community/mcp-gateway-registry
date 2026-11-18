"""Exercise native registration PKCE and CIMD lifecycle through the Admin API."""

import json
import shutil

import pytest

from tests.security import test_group_path_collisions as shell
from tests.security.test_registration_reconciliation import PROVISIONERS, cimd, client

REALM_PATH = "/admin/realms/mcp-gateway"
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
                    if entry["name"] == "mcp-cimd-runtime-guard":
                        entry["executors"] = []
                    if entry["name"] == "mcp-cimd-runtime-guard-policy":
                        entry["mode"] = "LENIENT"
        state["written"].append(key)
    body = {key: state[key]}
    if method == "GET" and key in state["written"] and scenario == key + "-readback-http-error":
        status = 503
    if method == "GET" and scenario == key + "-read-error":
        status = 503
    if method == "GET" and scenario == key + "-invalid-json":
        raw = "{invalid"
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
    if method == "PUT":
        state["clients"][uuid].update(json.loads(body_arg))
        body = state["clients"][uuid]
        status = 204
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
                    "client-attributes",
                    "grant-type",
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
    info=None,
    scenario="supported",
    profiles=None,
    policies=None,
):
    state = {
        "serverinfo": serverinfo() if info is None else info,
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
        "_check_cimd_first_enable",
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
@pytest.mark.parametrize("ack", ["false", "true"])
def test_first_enable_refuses_every_cached_client(tmp_path, monkeypatch, path, enabled, ack):
    record = cimd(enabled=enabled)
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        f'DCR_SWEEP_ACKNOWLEDGED={ack}\n_check_cimd_first_enable "$TOKEN"',
        clients=[record],
    )
    assert result.returncode != 0
    assert mutations(requests) == []
    assert state["clients"] == {"cimd": record}


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize("enabled", ["false", "true"])
def test_first_guard_install_refuses_cache_in_both_modes(tmp_path, monkeypatch, path, enabled):
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        f'ENABLE_CIMD_CONFIG={enabled}\nDCR_SWEEP_ACKNOWLEDGED=true\n_prepare_cimd_lifecycle "$TOKEN"',
        clients=[cimd(enabled=False)],
    )
    assert result.returncode != 0
    assert mutations(requests) == []
    assert "cimd" in state["clients"]


@pytest.mark.parametrize("path", PROVISIONERS)
def test_first_enable_detects_empty_marker_on_later_page(tmp_path, monkeypatch, path):
    records = [client(f"ordinary-{i}", f"ordinary-{i}") for i in range(101)]
    records.append(client("forged", "not-a-url", attributes={"cimd.cache.expiry.time.in.sec": ""}))
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        '_check_cimd_first_enable "$TOKEN"',
        clients=records,
    )
    assert result.returncode != 0
    assert mutations(requests) == []
    assert state["clients"] == {record["id"]: record for record in records}
    assert any(
        int(r["query"].get("first", [0])[0]) > 0 for r in requests if r["path"].endswith("/clients")
    )


@pytest.mark.parametrize("path", PROVISIONERS)
def test_first_enable_without_cache_is_read_only(tmp_path, monkeypatch, path):
    record = client("ordinary", "https://vendor.example/client.json")
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        '_check_cimd_first_enable "$TOKEN"',
        clients=[record],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert mutations(requests) == []
    assert state["clients"] == {"ordinary": record}
    assert not any(r["path"] == REALM_PATH for r in requests)


@pytest.mark.parametrize("path", PROVISIONERS)
def test_existing_guard_skips_first_enable_cache_refusal(tmp_path, monkeypatch, path):
    result, requests, _ = run(
        tmp_path,
        monkeypatch,
        path,
        '_check_cimd_first_enable "$TOKEN"',
        clients=[cimd()],
        policies=[{"name": "mcp-cimd-runtime-guard-policy"}],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert mutations(requests) == []
    assert not any(r["path"].endswith("/clients") for r in requests)
    inventory = next(i for i, r in enumerate(requests) if r["path"] == "/admin/serverinfo")
    policies = next(i for i, r in enumerate(requests) if r["path"].endswith("/policies"))
    assert inventory < policies


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize(
    "scenario",
    [
        "client-list-error",
        "client-list-malformed",
        "client-later-page-error",
        "policies-read-error",
        "policies-invalid-json",
    ],
)
def test_first_enable_inventory_failure_prevents_mutation(tmp_path, monkeypatch, path, scenario):
    records = [client(f"ordinary-{i}", f"ordinary-{i}") for i in range(101)]
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        '_check_cimd_first_enable "$TOKEN"',
        clients=records,
        scenario=scenario,
    )
    assert result.returncode != 0
    assert mutations(requests) == []
    assert state["clients"] == {record["id"]: record for record in records}


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize("collection", ["profiles", "policies"])
@pytest.mark.parametrize(
    "failure", ["write-error", "readback-missing", "readback-corrupt", "readback-http-error"]
)
def test_runtime_guard_write_and_readback_failures_are_fatal(
    tmp_path, monkeypatch, path, collection, failure
):
    result, requests, _ = run(
        tmp_path,
        monkeypatch,
        path,
        'ENABLE_CIMD_CONFIG=false\n_configure_cimd_runtime_guard "$TOKEN"',
        scenario=f"{collection}-{failure}",
    )
    assert result.returncode != 0
    assert not any(r["method"] == "DELETE" or "/clients/" in r["path"] for r in requests)
    if collection == "profiles":
        assert not any(r["method"] == "PUT" and r["path"].endswith("/policies") for r in requests)


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize(
    "kind,missing",
    [
        (CONDITION_TYPE, "client-attributes"),
        (CONDITION_TYPE, "grant-type"),
        (EXECUTOR_TYPE, "reject-request"),
    ],
)
def test_runtime_guard_capability_failure_prevents_sweeps(
    tmp_path, monkeypatch, path, kind, missing
):
    info = serverinfo()
    info["componentTypes"][kind] = [
        entry for entry in info["componentTypes"][kind] if entry["id"] != missing
    ]
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        'ENABLE_CIMD_CONFIG=false\nDCR_SWEEP_ACKNOWLEDGED=true\n_prepare_cimd_lifecycle "$TOKEN"',
        clients=[cimd()],
        info=info,
        policies=[{"name": "mcp-cimd-runtime-guard-policy"}],
    )
    assert result.returncode != 0
    assert mutations(requests) == []
    assert state["clients"]["cimd"]["enabled"] is True


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize("scenario", ["serverinfo-http-error", "serverinfo-invalid-json"])
def test_existing_guard_does_not_bypass_provider_inventory(tmp_path, monkeypatch, path, scenario):
    result, requests, _ = run(
        tmp_path,
        monkeypatch,
        path,
        '_check_cimd_first_enable "$TOKEN"',
        policies=[{"name": "mcp-cimd-runtime-guard-policy"}],
        scenario=scenario,
    )
    assert result.returncode != 0
    assert mutations(requests) == []


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize("collection", ["profiles", "policies"])
@pytest.mark.parametrize("failure", ["write-error", "readback-corrupt", "readback-http-error"])
def test_failed_runtime_guard_prevents_creation_shutdown_and_revocation(
    tmp_path, monkeypatch, path, collection, failure
):
    record = cimd()
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        'ENABLE_CIMD_CONFIG=false\nDCR_SWEEP_ACKNOWLEDGED=true\n_prepare_cimd_lifecycle "$TOKEN"',
        clients=[record],
        scenario=f"{collection}-{failure}",
        policies=[{"name": "mcp-cimd-runtime-guard-policy"}, {"name": "cimd-vendors-policy"}],
    )
    assert result.returncode != 0
    assert state["clients"] == {"cimd": record}
    assert any(p["name"] == "cimd-vendors-policy" for p in state["policies"])
    assert not any(
        r["method"] == "DELETE" or (r["method"] == "PUT" and "/clients/" in r["path"])
        for r in requests
    )


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize("enabled", ["false", "true"])
def test_runtime_guard_matches_native_markers_and_exact_hosts(tmp_path, monkeypatch, path, enabled):
    result, requests, state = run(
        tmp_path,
        monkeypatch,
        path,
        f'ENABLE_CIMD_CONFIG={enabled}\nCIMD_TRUSTED_DOMAINS=\'["vendor.example", "other.example", "localhost", "127.0.0.1"]\'\n'
        '_configure_cimd_runtime_guard "$TOKEN" && _configure_cimd_runtime_guard "$TOKEN"',
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert {"name": "unrelated-profile"} in state["profiles"]
    assert {"name": "unrelated-policy"} in state["policies"]
    profiles = [p for p in state["profiles"] if p["name"] == "mcp-cimd-runtime-guard"]
    policies = [p for p in state["policies"] if p["name"] == "mcp-cimd-runtime-guard-policy"]
    assert len(profiles) == len(policies) == 1
    assert [e["executor"] for e in profiles[0]["executors"]] == ["reject-request"]
    policy = policies[0]
    assert policy["enabled"] is True
    assert policy["mode"] == "STRICT"
    assert policy["profiles"] == ["mcp-cimd-runtime-guard"]
    assert policy["conditions"][0] == {
        "condition": "grant-type",
        "configuration": {"grant_types": [], "is-negative-logic": True},
    }
    conditions = policy["conditions"][1:]
    assert all(c["condition"] == "client-attributes" for c in conditions)
    positive = [
        c["configuration"]
        for c in conditions
        if not c["configuration"].get("is-negative-logic", False)
    ]
    negative = [
        c["configuration"]
        for c in conditions
        if c["configuration"].get("is-negative-logic") is True
    ]
    assert len(positive) == 1
    assert json.loads(positive[0]["attributes"]) == [
        {"key": "display.on.consent.screen", "value": "true"}
    ]
    hosts = {"vendor.example", "other.example"} if enabled == "true" else set()
    assert len(negative) == len(hosts)
    assert {json.loads(c["attributes"])[0]["value"] for c in negative} == {
        f"The client's hostname is {host}" for host in hosts
    }
    assert all(
        len(json.loads(c["attributes"])) == 1
        and json.loads(c["attributes"])[0]["key"] == "consent.screen.text"
        for c in negative
    )
    assert not any(r["method"] == "DELETE" or r["path"] == REALM_PATH for r in requests)
