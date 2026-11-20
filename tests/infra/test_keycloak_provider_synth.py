"""Synthesize the complete CDK app to check provider-gated public resources."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INFRA = ROOT / "infra"
ACCOUNT = "111111111111"
REGION = "us-east-1"
DOMAIN = "example.test"


@pytest.fixture(scope="module")
def synth_app(tmp_path_factory):
    node = shutil.which("node")
    dependencies = INFRA / "node_modules"
    if not node or not (dependencies / "ts-node/register").is_dir():
        pytest.skip("CDK synth requires node and npm ci in infra")

    workspace = tmp_path_factory.mktemp("provider-synth")
    staged = workspace / "infra"
    shutil.copytree(
        INFRA,
        staged,
        ignore=shutil.ignore_patterns(
            "node_modules",
            "cdk.out",
            "dist",
            "cdk-outputs.json",
            "set-env.sh",
        ),
    )
    (staged / "node_modules").symlink_to(dependencies, target_is_directory=True)
    (workspace / "terraform").symlink_to(ROOT / "terraform", target_is_directory=True)
    context = {
        f"availability-zones:account={ACCOUNT}:region={REGION}": [
            f"{REGION}a",
            f"{REGION}b",
            f"{REGION}c",
        ],
        f"hosted-zone:account={ACCOUNT}:domainName={DOMAIN}:region={REGION}": {
            "Id": "/hostedzone/ZTEST123",
            "Name": f"{DOMAIN}.",
        },
    }
    cache = {}

    def _synth(flags, cloudfront, dns, admin_cidrs=None, missing_secret=None):
        key = (flags, cloudfront, dns, tuple(admin_cidrs or []), missing_secret)
        if key in cache:
            return cache[key]
        config = {
            "awsAccountId": ACCOUNT,
            "awsRegion": REGION,
            "baseDomain": DOMAIN,
            "enableRoute53Dns": dns,
            "cloudfront": {"enabled": cloudfront},
            "keycloakAdminCidrBlocks": admin_cidrs,
            "enableObservability": False,
            "monitoring": {"enabled": False},
            "images": {
                "registry": "example.invalid/registry:test",
                "authServer": "example.invalid/auth-server:test",
            },
            "auth0": {"enabled": "auth0" in flags},
            "okta": {"enabled": "okta" in flags},
            "entra": {"enabled": "entra" in flags},
            "cognito": {
                "enabled": "cognito" in flags,
                "userPoolId": f"{REGION}_Example123",
                "clientId": "example-web-client",
                "domain": "example-login",
                "clientSecretArn": (
                    f"arn:aws:secretsmanager:{REGION}:{ACCOUNT}:secret:example-cognito-AbCdEf"
                ),
            },
            "pingfederate": {
                "enabled": "pingfederate" in flags,
                "baseUrl": "https://idp.example.test",
                "clientId": "example-web-client",
                "clientSecretArn": (
                    f"arn:aws:secretsmanager:{REGION}:{ACCOUNT}:secret:example-ping-client-AbCdEf"
                ),
                "adminUrl": "https://admin.idp.example.test",
                "adminUser": "example-admin",
                "adminPasswordArn": (
                    f"arn:aws:secretsmanager:{REGION}:{ACCOUNT}:secret:example-ping-admin-AbCdEf"
                ),
            },
        }
        (staged / "config.yaml").write_text(json.dumps({"registry": config}))
        outdir = workspace / f"assembly-{len(cache)}"
        env = {
            k: v for k, v in os.environ.items() if not k.startswith(("CDK_", "AWS_", "TS_NODE_"))
        }
        env.update(
            CDK_OUTDIR=str(outdir),
            CDK_CONTEXT_JSON=json.dumps(context),
            CDK_KEYCLOAK_ADMIN_PASSWORD="EXAMPLE_AdminPassword123",
            CDK_KEYCLOAK_DATABASE_PASSWORD="EXAMPLE_DatabasePassword123",
            CDK_DOCUMENTDB_ADMIN_PASSWORD="EXAMPLE_DocumentPassword123",
            AWS_EC2_METADATA_DISABLED="true",
        )
        if missing_secret:
            env.pop(missing_secret)
        result = subprocess.run(
            [node, "-r", "ts-node/register/transpile-only", "bin/main.ts"],
            cwd=staged,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        templates = {
            path.name.removesuffix(".template.json"): json.loads(path.read_text())
            for path in outdir.glob("*.template.json")
        }
        cache[key] = (result, templates)
        return result, templates

    return _synth


def _resources(template, resource_type):
    return {
        key: value
        for key, value in template.get("Resources", {}).items()
        if value["Type"] == resource_type
    }


def _containers(template):
    return {
        container["Name"]: container
        for task in _resources(template, "AWS::ECS::TaskDefinition").values()
        for container in task["Properties"]["ContainerDefinitions"]
    }


@pytest.mark.parametrize(
    "cloudfront,dns", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize(
    "flags,provider",
    [
        (("auth0",), "auth0"),
        (("okta",), "okta"),
        (("entra",), "entra"),
        (("cognito",), "cognito"),
        (("pingfederate",), "pingfederate"),
        (("entra", "cognito"), "entra"),
        (("pingfederate", "auth0", "okta", "entra", "cognito"), "pingfederate"),
        (("okta", "entra"), "okta"),
        (("auth0", "okta", "entra"), "auth0"),
    ],
)
def test_inactive_keycloak_has_no_runtime_or_public_routes(
    synth_app,
    flags,
    provider,
    cloudfront,
    dns,
):
    result, templates = synth_app(flags, cloudfront, dns, ["203.0.113.10/32"])
    assert result.returncode == 0, result.stderr
    auth = templates["Registry-Auth"]
    service = templates["Registry-Service"]
    forbidden = {
        "AWS::ECS::Cluster",
        "AWS::ECS::Service",
        "AWS::ECS::TaskDefinition",
        "AWS::ElasticLoadBalancingV2::LoadBalancer",
        "AWS::ElasticLoadBalancingV2::TargetGroup",
        "AWS::ElasticLoadBalancingV2::Listener",
        "AWS::ElasticLoadBalancingV2::ListenerRule",
        "AWS::CloudFront::Distribution",
        "AWS::Route53::RecordSet",
        "AWS::CertificateManager::Certificate",
        "AWS::WAFv2::WebACL",
    }
    assert not forbidden.intersection(r["Type"] for r in auth.get("Resources", {}).values())
    assert not auth.get("Outputs")
    assert not any(key.startswith("Keycloak") for key in service.get("Outputs", {}))
    assert service["Outputs"]["AuthProvider"]["Value"] == provider
    assert len(_resources(service, "AWS::ElasticLoadBalancingV2::LoadBalancer")) == 1
    assert len(_resources(service, "AWS::CloudFront::Distribution")) == int(cloudfront)
    assert not any("KeycloakAlb" in key for key in service["Resources"])
    assert "Registry-Auth:" not in json.dumps(service)
    selected = []
    for container in _containers(service).values():
        env = {entry["Name"]: entry["Value"] for entry in container.get("Environment", [])}
        if "AUTH_PROVIDER" not in env:
            continue
        selected.append(container["Name"])
        assert env["AUTH_PROVIDER"] == provider
        assert env["KEYCLOAK_ENABLED"] == "false"
        assert env["KEYCLOAK_URL"] == env["KEYCLOAK_EXTERNAL_URL"] == ""
        assert not any(s["Name"].startswith("KEYCLOAK_") for s in container.get("Secrets", []))
    assert len(selected) == 2
    for policy in _resources(service, "AWS::IAM::Policy").values():
        statements = policy["Properties"]["PolicyDocument"]["Statement"]
        for statement in statements:
            if "secretsmanager:GetSecretValue" in statement.get("Action", []):
                assert "Keycloak" not in json.dumps(statement.get("Resource"))


@pytest.mark.parametrize("cloudfront,dns", [(False, True), (True, False), (True, True)])
def test_active_keycloak_keeps_public_routes_and_container_credentials(synth_app, cloudfront, dns):
    result, templates = synth_app((), cloudfront, dns)
    assert result.returncode == 0, result.stderr
    auth = templates["Registry-Auth"]
    assert len(_resources(auth, "AWS::ElasticLoadBalancingV2::LoadBalancer")) == 1
    assert len(_resources(auth, "AWS::ElasticLoadBalancingV2::TargetGroup")) == 1
    assert len(_resources(auth, "AWS::ECS::Service")) == 1
    assert len(_resources(auth, "AWS::CloudFront::Distribution")) == int(cloudfront)
    assert len(_resources(auth, "AWS::Route53::RecordSet")) == int(dns)
    paths = {
        path
        for rule in _resources(auth, "AWS::ElasticLoadBalancingV2::ListenerRule").values()
        for condition in rule["Properties"]["Conditions"]
        if condition["Field"] == "path-pattern"
        for path in condition.get(
            "Values", condition.get("PathPatternConfig", {}).get("Values", [])
        )
    }
    assert {"/realms/mcp-gateway/*", "/resources/*", "/admin/*", "/realms/master/*"} <= paths
    service = templates["Registry-Service"]
    assert "KeycloakUrl" in auth["Outputs"]
    assert "KeycloakUrl" in service["Outputs"]
    selected = []
    for container in _containers(service).values():
        env = {entry["Name"]: entry["Value"] for entry in container.get("Environment", [])}
        if "AUTH_PROVIDER" not in env:
            continue
        selected.append(container["Name"])
        assert env["AUTH_PROVIDER"] == "keycloak"
        assert env["KEYCLOAK_ENABLED"] == "true"
        assert env["KEYCLOAK_URL"] == env["KEYCLOAK_EXTERNAL_URL"]
        assert env["KEYCLOAK_URL"] != ""
        assert {"KEYCLOAK_CLIENT_SECRET", "KEYCLOAK_M2M_CLIENT_SECRET"} <= {
            secret["Name"] for secret in container["Secrets"]
        }
    assert len(selected) == 2


def test_provider_switch_preserves_database_and_credential_resource_addresses(synth_app):
    active_result, active = synth_app((), True, False)
    inactive_result, inactive = synth_app(("auth0",), True, False)
    assert active_result.returncode == inactive_result.returncode == 0
    assert active["Registry-Data"] == inactive["Registry-Data"]
    assert active["Registry-Ops"] == inactive["Registry-Ops"]
    assert _resources(active["Registry-Service"], "AWS::SecretsManager::Secret") == {
        key: value
        for key, value in _resources(
            inactive["Registry-Service"], "AWS::SecretsManager::Secret"
        ).items()
        if "Auth0" not in key
    }
    bootstrap = {
        key: resource
        for key, resource in active["Registry-Auth"]["Resources"].items()
        if resource["Type"] == "Custom::AWS" and "SsmKeycloakAdmin" in key
    }
    assert len(bootstrap) == 2
    assert bootstrap == {key: inactive["Registry-Auth"]["Resources"][key] for key in bootstrap}


@pytest.mark.parametrize(
    "cloudfront,dns,cidrs,error",
    [
        (False, False, None, "Keycloak needs a public HTTPS front"),
        (True, False, ["203.0.113.10/32"], "keycloakAdminCidrBlocks cannot be enforced"),
    ],
)
def test_active_keycloak_still_rejects_unreachable_or_unenforceable_config(
    synth_app,
    cloudfront,
    dns,
    cidrs,
    error,
):
    result, _ = synth_app((), cloudfront, dns, cidrs)
    assert result.returncode != 0
    assert error in result.stderr


@pytest.mark.parametrize(
    "secret", ["CDK_KEYCLOAK_ADMIN_PASSWORD", "CDK_KEYCLOAK_DATABASE_PASSWORD"]
)
def test_retained_resources_still_require_their_credentials(synth_app, secret):
    result, _ = synth_app(("auth0",), False, False, missing_secret=secret)
    assert result.returncode != 0
    assert secret in result.stderr
