"""Fail-closed realm lifecycle behavior through the production shell API boundary."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.security.test_group_path_collisions import HELM, MUTATORS, _run_shell

PROVISIONERS = [MUTATORS[0][0], MUTATORS[1][0], HELM]


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize(
    "scenario", ["roles-missing", "roles-write-failure", "roles-readback-failure"]
)
def test_dcr_role_policy_must_be_available_and_verified(tmp_path, path, scenario):
    result, requests = _run_shell(tmp_path, path, 'configure_dcr_default_roles "$TOKEN"', scenario)
    assert result.returncode != 0, result.stdout + result.stderr
    if scenario == "roles-missing":
        assert not any(
            request["method"] != "GET" and not request["path"].endswith("/token")
            for request in requests
        )


@pytest.mark.parametrize("path", PROVISIONERS)
def test_dcr_role_policy_installed_for_both_registration_subtypes(tmp_path, path):
    result, requests = _run_shell(
        tmp_path, path, 'configure_dcr_default_roles "$TOKEN"', "roles-present"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    policies = [
        json.loads(request["body"])
        for request in requests
        if request["path"].endswith("/components") and request["method"] == "POST"
    ]
    assert {(policy["providerId"], policy["subType"]) for policy in policies} == {
        ("default-roles", "anonymous"),
        ("default-roles", "authenticated"),
    }
    assert all(policy["parentId"] == "realm-id" for policy in policies)


@pytest.mark.parametrize("path", PROVISIONERS)
def test_sweep_preflight_lists_targets_and_refuses_without_acknowledgment(tmp_path, path):
    result, requests = _run_shell(
        tmp_path, path, 'ENABLE_DCR_CONFIG=false; preflight_dcr_upgrade "$TOKEN"', "sweep"
    )
    assert result.returncode != 0
    assert "registered-browser" in result.stdout
    assert not any(
        request["method"] not in ("GET",) and not request["path"].endswith("/token")
        for request in requests
    )


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize(
    "choice", ["DCR_SWEEP_ACKNOWLEDGED=true", "DCR_KEEP_CLIENTS=registered-browser"]
)
def test_sweep_preflight_accepts_explicit_operator_choice(tmp_path, path, choice):
    result, requests = _run_shell(
        tmp_path,
        path,
        f'ENABLE_DCR_CONFIG=false; {choice}; preflight_dcr_upgrade "$TOKEN"',
        "sweep",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(
        request["method"] != "GET" and not request["path"].endswith("/token")
        for request in requests
    )


@pytest.mark.parametrize("path", PROVISIONERS)
def test_long_operations_refresh_cached_admin_token(tmp_path, path):
    command = r"""
_realm_fresh_token >/dev/null
(umask 077; printf '0\nexpired-token\n' > "$CREDENTIAL_IO_DIR/realm-token")
_check_group_paths "$KEYCLOAK_URL" "$REALM" "$TOKEN"
"""
    result, requests = _run_shell(tmp_path, path, command, "empty")
    assert result.returncode == 0, result.stdout + result.stderr
    assert len([r for r in requests if r["path"].endswith("/token")]) == 2
    groups = [r for r in requests if r["path"].endswith("/groups")]
    assert groups[0]["authorization"] == "Authorization: Bearer fixture-token"


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize("scenario", ["exchange-refused", "exchange-removed"])
def test_old_keycloak_keeps_existing_token_exchange_restrictions(tmp_path, path, scenario):
    result, requests = _run_shell(tmp_path, path, 'restrict_token_exchange "$TOKEN"', scenario)
    assert result.returncode != 0
    assert not any(r["method"] == "PUT" for r in requests)


def test_terraform_environment_cannot_replace_caller_target_or_sweep_choice(tmp_path):
    env_file = tmp_path / "terraform" / ".env"
    env_file.parent.mkdir()
    env_file.write_text(
        "KEYCLOAK_ADMIN_URL=http://wrong.invalid\nREGISTRY_URL=http://wrong.invalid\nAWS_REGION=wrong\nDCR_SWEEP_ACKNOWLEDGED=true\nKEYCLOAK_ADMIN_PASSWORD=wrong-password\n"
    )
    command = r"""
KEYCLOAK_ADMIN_URL=http://caller.invalid
REGISTRY_URL=https://caller.invalid
AUTH_SERVER_EXTERNAL_URL=https://caller.invalid
AWS_REGION=us-west-2
INITIAL_ADMIN_PASSWORD=synthetic
DCR_SWEEP_ACKNOWLEDGED=false
wait_for_keycloak() {
    [ "$KEYCLOAK_ADMIN_PASSWORD" = fixture-password ] || exit 42
    printf 'CALLER_VALUES:%s|%s|%s|%s\n' "$KEYCLOAK_ADMIN_URL" "$REGISTRY_URL" "$AWS_REGION" "$DCR_SWEEP_ACKNOWLEDGED"
    exit 0
}
main
"""
    result, requests = _run_shell(
        tmp_path, "terraform/aws-ecs/scripts/init-keycloak.sh", command, "empty"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "CALLER_VALUES:http://caller.invalid|https://caller.invalid|us-west-2|false"
        in result.stdout
    )
    assert requests == []


@pytest.mark.parametrize(
    "path", [MUTATORS[0][0], MUTATORS[1][0], "terraform/aws-ecs/scripts/init-keycloak.sh"]
)
@pytest.mark.parametrize(
    "with_dotenv, caller_mode",
    [
        (False, "supplied"),
        (True, "supplied"),
        (True, "absent"),
        (True, "empty-password"),
        (True, "empty-user"),
        (True, "empty-url"),
        (True, "empty-region"),
    ],
)
def test_real_bootstrap_invocation_uses_exported_admin_credentials(
    tmp_path, path, with_dotenv, caller_mode
):
    """Run main from a clean environment; stop at a deliberately refused login."""
    root = Path(__file__).resolve().parents[2]
    for relative in {
        path,
        "keycloak/setup/credential-io.sh",
        "keycloak/setup/group-path-preflight.sh",
    }:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    dotenv = tmp_path / ("terraform/.env" if path.startswith("terraform/") else ".env")
    if with_dotenv:
        dotenv.write_text(
            "KEYCLOAK_ADMIN_URL=http://wrong.invalid\n"
            "KEYCLOAK_ADMIN=wrong-user\n"
            "KEYCLOAK_ADMIN_PASSWORD=wrong-password\nAWS_REGION=us-east-1\n"
        )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "auth.json"
    curl = fake_bin / "curl"
    curl.write_text(
        f"#!{sys.executable}\n"
        + r"""
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
url = next(arg for arg in args if arg.startswith(("http://", "https://")))
if url.endswith("/admin/"):
    sys.exit(0)
if not url.endswith("/protocol/openid-connect/token"):
    sys.exit(97)
fields = {}
for index, arg in enumerate(args[:-1]):
    if arg != "--data-urlencode":
        continue
    value = args[index + 1]
    if "=" in value:
        name, value = value.split("=", 1)
    else:
        name, filename = value.split("@", 1)
        value = sys.stdin.read() if filename == "-" else Path(filename).read_text()
    fields[name] = value
Path(os.environ["AUTH_CAPTURE"]).write_text(json.dumps({"url": url, "fields": fields}))
print('{"error":"invalid_grant"}')
"""
    )
    curl.chmod(0o755)
    env = {
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "KEYCLOAK_ADMIN_URL": "http://caller.invalid",
        "KEYCLOAK_ADMIN": "caller-user",
        "KEYCLOAK_ADMIN_PASSWORD": "caller-password",
        "INITIAL_ADMIN_PASSWORD": "synthetic-initial-password",
        "AWS_REGION": "us-west-2",
        "AUTH_CAPTURE": str(capture),
    }
    if caller_mode == "absent":
        for name in (
            "KEYCLOAK_ADMIN_URL",
            "KEYCLOAK_ADMIN",
            "KEYCLOAK_ADMIN_PASSWORD",
            "AWS_REGION",
        ):
            del env[name]
    elif caller_mode.startswith("empty-"):
        name = {
            "empty-password": "KEYCLOAK_ADMIN_PASSWORD",
            "empty-user": "KEYCLOAK_ADMIN",
            "empty-url": "KEYCLOAK_ADMIN_URL",
            "empty-region": "AWS_REGION",
        }[caller_mode]
        if name == "AWS_REGION" and not path.startswith("terraform/"):
            pytest.skip("AWS_REGION is specific to Terraform setup")
        env[name] = ""
    result = subprocess.run(
        ["bash", str(tmp_path / path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    if caller_mode.startswith("empty-"):
        assert not capture.exists(), result.stdout + result.stderr
    else:
        assert capture.exists(), result.stdout + result.stderr
        request = json.loads(capture.read_text())
        selected = "wrong" if caller_mode == "absent" else "caller"
        assert (
            request["url"]
            == f"http://{selected}.invalid/realms/master/protocol/openid-connect/token"
        )
        assert request["fields"]["username"] == f"{selected}-user"
        assert request["fields"]["password"] == f"{selected}-password"
    assert "caller-password" not in result.stdout + result.stderr
    assert "wrong-password" not in result.stdout + result.stderr
    assert dotenv.exists() == with_dotenv


@pytest.mark.parametrize("scenario", ["custom-realm-new", "custom-realm-existing"])
def test_chart_creates_and_updates_selected_realm_before_client_provisioning(tmp_path, scenario):
    command = """
REALM=team-realm
KEYCLOAK_CLIENT_SECRET=synthetic-web-secret
KEYCLOAK_M2M_CLIENT_SECRET=synthetic-machine-secret
create_realm "$TOKEN" && create_clients "$TOKEN"
"""
    result, requests = _run_shell(tmp_path, HELM, command, scenario)
    assert result.returncode == 0, result.stdout + result.stderr
    realm_writes = [
        r
        for r in requests
        if r["method"] in ("POST", "PUT")
        and r["path"] in ("/admin/realms", "/admin/realms/team-realm")
    ]
    assert len(realm_writes) == 1
    assert realm_writes[0]["method"] == ("POST" if scenario == "custom-realm-new" else "PUT")
    assert json.loads(realm_writes[0]["body"])["realm"] == "team-realm"
    clients = [
        json.loads(r["body"])["clientId"]
        for r in requests
        if r["method"] == "POST" and r["path"] == "/admin/realms/team-realm/clients"
    ]
    assert set(clients) == {"mcp-gateway-web", "mcp-gateway-m2m"}
    assert not any(r["path"].startswith("/admin/realms/mcp-gateway") for r in requests)


@pytest.mark.parametrize("path", PROVISIONERS)
def test_older_external_keycloak_allows_dcr_off_without_standard_exchange(tmp_path, path):
    result, requests = _run_shell(
        tmp_path, path, 'ENABLE_DCR_CONFIG=false; preflight_dcr_upgrade "$TOKEN"', "roles-missing"
    )
    assert result.returncode == 0, result.stderr
    assert not any(r["method"] == "PUT" for r in requests)


@pytest.mark.parametrize("path", PROVISIONERS)
def test_older_external_keycloak_refuses_dcr_before_mutation(tmp_path, path):
    result, requests = _run_shell(
        tmp_path, path, 'ENABLE_DCR_CONFIG=true; preflight_dcr_upgrade "$TOKEN"', "roles-missing"
    )
    assert result.returncode != 0
    assert not any(r["method"] == "PUT" for r in requests)


@pytest.mark.parametrize("path", PROVISIONERS)
def test_old_standard_exchange_is_refused_before_any_realm_mutation(tmp_path, path):
    result, requests = _run_shell(
        tmp_path,
        path,
        'ENABLE_DCR_CONFIG=false; preflight_dcr_upgrade "$TOKEN"',
        "exchange-removed",
    )
    assert result.returncode != 0
    assert not any(r["method"] == "PUT" for r in requests)


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize(
    "scenario",
    ["cimd-old", "cimd-feature-missing", "cimd-condition-missing", "cimd-executor-missing"],
)
@pytest.mark.parametrize(
    "command",
    [
        'ENABLE_CIMD_CONFIG=true; preflight_dcr_upgrade "$TOKEN"',
        'configure_cimd_client_policy "$TOKEN"',
    ],
)
def test_unsupported_cimd_fails_before_realm_writes(tmp_path, path, scenario, command):
    result, requests = _run_shell(tmp_path, path, command, scenario)
    assert result.returncode != 0
    assert not any(
        request["method"] != "GET" and not request["path"].endswith("/token")
        for request in requests
    )


@pytest.mark.parametrize("path", PROVISIONERS)
def test_cimd_separates_metadata_hosts_from_loopback_callbacks(tmp_path, path):
    domains = ["vendor.example", "localhost", "127.0.0.1", "127.2.3.4", "::1", "[::1]"]
    command = (
        f"CIMD_TRUSTED_DOMAINS='{json.dumps(domains)}'\n" + 'configure_cimd_client_policy "$TOKEN"'
    )
    result, requests = _run_shell(tmp_path, path, command, "cimd-supported")
    assert result.returncode == 0, result.stdout + result.stderr
    documents = {
        request["path"].rsplit("/", 1)[-1]: json.loads(request["body"])
        for request in requests
        if request["method"] == "PUT"
    }
    profiles = documents["profiles"]["profiles"]
    policies = documents["policies"]["policies"]
    assert {profile["name"] for profile in profiles} == {
        "unrelated-profile",
        "cimd-vendors",
        "mcp-registration-scope-guard",
        "mcp-public-pkce",
        "mcp-cimd-runtime-guard",
    }
    assert {policy["name"] for policy in policies} == {
        "unrelated",
        "cimd-vendors-policy",
        "mcp-registration-scope-guard-policy",
        "mcp-public-pkce-policy",
        "mcp-cimd-runtime-guard-policy",
    }
    assert {"name": "unrelated-profile"} in profiles
    assert {"name": "unrelated"} in policies
    profile = next(profile for profile in profiles if profile["name"] == "cimd-vendors")
    policy = next(policy for policy in policies if policy["name"] == "cimd-vendors-policy")
    assert profile["executors"] == [
        {
            "executor": "client-id-metadata-document",
            "configuration": {
                "cimd-allow-permitted-domains": sorted(domains),
            },
        }
    ]
    assert policy["enabled"] is True
    assert policy["profiles"] == ["cimd-vendors"]
    assert policy["conditions"] == [
        {
            "condition": "client-id-uri",
            "configuration": {
                "client-id-uri-scheme": ["https"],
                "client-id-uri-allow-permitted-domains": ["vendor.example"],
            },
        }
    ]


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize(
    "domains",
    [
        '["localhost", "127.0.0.1", "::1", "[::1]"]',
        "[]",
        "{}",
        "[42]",
        '["*.vendor.example"]',
        '["vendor.example", "*.other.example"]',
    ],
)
def test_cimd_rejects_invalid_or_loopback_only_metadata_allowlist(tmp_path, path, domains):
    command = f"CIMD_TRUSTED_DOMAINS='{domains}'\n" + 'configure_cimd_client_policy "$TOKEN"'
    result, requests = _run_shell(tmp_path, path, command, "cimd-supported")
    assert result.returncode != 0
    assert not any(request["method"] == "PUT" for request in requests)


@pytest.mark.parametrize("path", PROVISIONERS)
def test_cimd_disable_retains_runtime_guard_and_unrelated_entries(tmp_path, path):
    result, requests = _run_shell(
        tmp_path, path, 'remove_cimd_client_policy "$TOKEN"', "cimd-supported"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    writes = [request for request in requests if request["method"] == "PUT"]
    assert all(request["path"].rsplit("/", 1)[-1] in {"profiles", "policies"} for request in writes)
    documents = {
        request["path"].rsplit("/", 1)[-1]: json.loads(request["body"]) for request in writes
    }
    final_policies = documents["policies"]["policies"]
    final_profiles = documents["profiles"]["profiles"]
    assert {policy["name"] for policy in final_policies} == {
        "unrelated",
        "mcp-registration-scope-guard-policy",
        "mcp-public-pkce-policy",
        "mcp-cimd-runtime-guard-policy",
        "mcp-no-remote-request-objects",
    }
    assert {profile["name"] for profile in final_profiles} == {
        "unrelated-profile",
        "mcp-registration-scope-guard",
        "mcp-public-pkce",
        "mcp-cimd-runtime-guard",
        "mcp-no-remote-request-objects",
    }
    assert {"name": "unrelated"} in final_policies
    assert {"name": "unrelated-profile"} in final_profiles


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize("keep", [False, True])
def test_dcr_off_invalidates_registration_token_even_for_explicitly_kept_client(
    tmp_path, path, keep
):
    command = "DCR_SWEEP_ACKNOWLEDGED=true\n"
    if keep:
        command += "DCR_KEEP_CLIENTS=registered-browser\n"
    command += 'disable_unlisted_clients "$TOKEN"'
    result, requests = _run_shell(tmp_path, path, command, "sweep")
    assert result.returncode == 0, result.stdout + result.stderr
    writes = [request for request in requests if request["method"] == "PUT"]
    assert [json.loads(request["body"]) for request in writes] == (
        [] if keep else [{"enabled": False}]
    )
    rotations = [
        request for request in requests if request["path"].endswith("/registration-access-token")
    ]
    assert [request["path"] for request in rotations] == [
        "/admin/realms/mcp-gateway/clients/browser/registration-access-token"
    ]
    assert all(request["method"] == "POST" for request in rotations)
    assert "replacement-token" not in result.stdout + result.stderr
