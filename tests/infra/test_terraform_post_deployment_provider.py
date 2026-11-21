"""Exercise the Terraform setup shell with isolated, read-only command fixtures."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "terraform/aws-ecs/scripts/post-deployment-setup.sh"


@pytest.fixture
def run_setup(tmp_path):
    bash = shutil.which("bash")
    jq = shutil.which("jq")
    if not bash or not jq:
        pytest.skip("Post-deployment shell regression requires bash and jq")

    scripts = tmp_path / "scripts"
    commands = tmp_path / "bin"
    scripts.mkdir()
    commands.mkdir()
    shutil.copy2(SCRIPT, scripts / SCRIPT.name)
    (commands / "jq").symlink_to(jq)
    log = tmp_path / "calls.log"

    def _executable(path, body):
        path.write_text("#!/bin/bash\nset -euo pipefail\n" + body)
        path.chmod(0o755)

    _executable(commands / "terraform", "exit 0\n")
    _executable(commands / "host", 'printf "dns %s\\n" "$*" >> "$CALL_LOG"\n')
    _executable(
        commands / "curl",
        'printf "http %s\\n" "$*" >> "$CALL_LOG"\nprintf 200\n',
    )
    _executable(commands / "sleep", "exit 90\n")
    _executable(
        commands / "aws",
        """printf "aws %s\\n" "$*" >> "$CALL_LOG"
case "$1 $2" in
    "sts get-caller-identity") printf '{}\\n' ;;
    "ecs describe-services")
        if [[ "$*" == *"deployments | length(@)"* ]]; then
            printf '1\\n'
        else
            printf '{"running":1,"desired":1}\\n'
        fi
        ;;
    "ecs update-service") printf '{}\\n' ;;
    *) exit 91 ;;
esac
""",
    )
    _executable(
        scripts / "save-terraform-outputs.sh",
        'cp "$FIXTURE_OUTPUTS" "$(dirname "$0")/terraform-outputs.json"\n',
    )
    _executable(
        scripts / "init-keycloak.sh",
        'printf "init-keycloak %s\\n" "$KEYCLOAK_ADMIN_URL" >> "$CALL_LOG"\n',
    )
    _executable(
        scripts / "run-documentdb-init.sh",
        'printf "init-scopes\\n" >> "$CALL_LOG"\n',
    )

    def _run(provider="auth0", keycloak_url=None, flags=(), passwords=False):
        outputs = {
            key: {"value": value}
            for key, value in {
                "vpc_id": "vpc-example",
                "ecs_cluster_name": "gateway-example",
                "ecs_cluster_arn": "arn:aws:ecs:us-east-1:123456789012:cluster/example",
                "mcp_gateway_url": "https://registry.example.test",
                "mcp_gateway_auth_url": "https://registry.example.test",
                "mcp_gateway_efs_id": "fs-example",
                "documentdb_cluster_endpoint": "db.example.test",
                "keycloak_url": keycloak_url,
            }.items()
        }
        if provider != "legacy":
            outputs["auth_provider"] = {"value": provider}
        fixture = tmp_path / "outputs.json"
        fixture.write_text(json.dumps(outputs))
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("AWS_", "KEYCLOAK_", "INITIAL_ADMIN_", "LOB1_", "LOB2_"))
        }
        env.update(
            PATH=f"{commands}:/usr/bin:/bin",
            CALL_LOG=str(log),
            FIXTURE_OUTPUTS=str(fixture),
            KEYCLOAK_ADMIN_URL="https://stale-keycloak.example.test",
        )
        if passwords:
            env.update(
                INITIAL_ADMIN_PASSWORD="EXAMPLE-admin-password",
                LOB1_USER_PASSWORD="EXAMPLE-lob1-password",
                LOB2_USER_PASSWORD="EXAMPLE-lob2-password",
            )
        result = subprocess.run(
            [bash, str(scripts / SCRIPT.name), *flags],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return result, log.read_text().splitlines()

    return _run


@pytest.mark.parametrize(
    "provider", ["auth0", "okta", "entra", "cognito", "pingfederate", "default"]
)
@pytest.mark.parametrize("stale_url", [None, "https://unused-keycloak.example.test"])
def test_inactive_keycloak_runs_registry_setup_without_keycloak_calls(
    run_setup, provider, stale_url
):
    result, calls = run_setup(provider=provider, keycloak_url=stale_url)

    assert result.returncode == 0, result.stdout + result.stderr
    assert all("keycloak" not in call for call in calls)
    assert "dns registry.example.test" in calls
    assert "init-scopes" in calls
    assert sum(call.startswith("aws ecs update-service") for call in calls) == 2
    assert any("https://registry.example.test/health" in call for call in calls)
    assert "Access Keycloak Admin:" not in result.stdout


@pytest.mark.parametrize("provider", ["keycloak", "legacy"])
def test_active_and_legacy_keycloak_keep_full_setup(run_setup, provider):
    result, calls = run_setup(
        provider=provider,
        keycloak_url="https://keycloak.example.test",
        passwords=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "dns keycloak.example.test" in calls
    assert any("--cluster keycloak --services keycloak" in call for call in calls)
    assert "init-keycloak https://keycloak.example.test" in calls
    assert any("https://keycloak.example.test/admin/" in call for call in calls)
    assert "init-scopes" in calls
    assert "Access Keycloak Admin: https://keycloak.example.test/admin" in result.stdout


def test_skip_keycloak_skips_only_initialization(run_setup):
    result, calls = run_setup(
        provider="keycloak",
        keycloak_url="https://keycloak.example.test",
        flags=("--skip-keycloak",),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(call.startswith("init-keycloak") for call in calls)
    assert "dns keycloak.example.test" in calls
    assert any("--cluster keycloak --services keycloak" in call for call in calls)
    assert any("https://keycloak.example.test/admin/" in call for call in calls)


@pytest.mark.parametrize("provider", [None, "", "unknown", 42, {"type": "auth0"}])
def test_malformed_explicit_provider_never_uses_legacy_keycloak_fallback(run_setup, provider):
    result, calls = run_setup(provider=provider, keycloak_url="https://keycloak.example.test")

    assert result.returncode != 0
    assert calls == ["aws sts get-caller-identity"]
    assert "auth_provider output" in result.stdout


@pytest.mark.parametrize("provider", ["keycloak", "legacy"])
def test_missing_keycloak_url_blocks_keycloak_and_ambiguous_legacy_setup(run_setup, provider):
    result, calls = run_setup(provider=provider)

    assert result.returncode != 0
    assert calls == ["aws sts get-caller-identity"]
