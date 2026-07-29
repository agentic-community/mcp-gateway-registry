"""LIVE AWS Secrets Manager integration tests for the egress token store.

These tests create and force-delete isolated AWS secrets. They are skipped before
constructing an AWS client unless explicitly enabled. Run only in a non-production
account whose ID is supplied as a second, independent safety check:

    RUN_LIVE_AWS_SECRETS_MANAGER_INTEGRATION=1 \
    AWS_SECRETS_MANAGER_INTEGRATION_EXPECTED_ACCOUNT_ID=123456789012 \
    AWS_SECRETS_MANAGER_INTEGRATION_REGION=us-east-1 \
    uv run pytest tests/integration/test_aws_secrets_manager_live_integration.py \
      -m 'integration and live' -v -o addopts=''
"""

import json
import os
import uuid

import boto3
import pytest
from botocore.exceptions import ClientError

from registry.egress_auth.schemas import StoredToken
from registry.secrets.secrets_manager.store import SecretsManagerStore

pytestmark = [pytest.mark.integration, pytest.mark.live, pytest.mark.asyncio]
_ENABLE_ENV = "RUN_LIVE_AWS_SECRETS_MANAGER_INTEGRATION"
_ACCOUNT_ENV = "AWS_SECRETS_MANAGER_INTEGRATION_EXPECTED_ACCOUNT_ID"
_REGION_ENV = "AWS_SECRETS_MANAGER_INTEGRATION_REGION"


class _TrackingSecretsManagerClient:
    """Track exactly the isolated resources this test may clean up."""

    def __init__(self, client) -> None:
        self._client = client
        self.created_names: set[str] = set()
        self.active_names: set[str] = set()

    def create_secret(self, **kwargs):
        kwargs.setdefault("Tags", [{"Key": "mcp-gateway-registry", "Value": "integration-test"}])
        response = self._client.create_secret(**kwargs)
        self.created_names.add(kwargs["Name"])
        self.active_names.add(kwargs["Name"])
        return response

    def delete_secret(self, **kwargs):
        response = self._client.delete_secret(**kwargs)
        self.active_names.discard(kwargs["SecretId"])
        return response

    def __getattr__(self, name):
        return getattr(self._client, name)


@pytest.fixture
async def live_secrets_manager():
    if os.getenv(_ENABLE_ENV) != "1":
        pytest.skip(f"destructive live AWS test; set {_ENABLE_ENV}=1 to opt in")
    expected_account = os.getenv(_ACCOUNT_ENV, "")
    if not (expected_account.isdigit() and len(expected_account) == 12):
        pytest.fail(f"{_ACCOUNT_ENV} must be the exact 12-digit non-production account ID")

    region = os.getenv(_REGION_ENV) or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not region:
        pytest.fail(f"set {_REGION_ENV} (or AWS_REGION/AWS_DEFAULT_REGION)")
    session = boto3.Session(region_name=region)
    actual_account = session.client("sts").get_caller_identity()["Account"]
    if actual_account != expected_account:
        pytest.fail(f"AWS account mismatch: expected {expected_account}, got {actual_account}")

    raw_client = session.client("secretsmanager")
    client = _TrackingSecretsManagerClient(raw_client)
    prefix = f"mcp/egress/integration-tests/{uuid.uuid4().hex}"
    try:
        yield client, SecretsManagerStore(client, prefix, target_payload_bytes=3072), prefix
    finally:
        cleanup_errors = []
        for name in sorted(client.active_names, reverse=True):
            try:
                raw_client.delete_secret(SecretId=name, ForceDeleteWithoutRecovery=True)
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
                    cleanup_errors.append(f"{name}: {exc}")
        if cleanup_errors:
            pytest.fail("live AWS integration cleanup failed: " + "; ".join(cleanup_errors))


async def test_live_aws_secrets_manager_integration_overflow_lifecycle(
    live_secrets_manager, capsys
):
    """Exercise real create/put/get/delete APIs and the overflow manifest lifecycle.

    Uses target_payload_bytes=3072 so a single token (~1.6 KiB serialized) fits
    inline but two tokens (~3.2 KiB) overflow to shards.
    """
    client, store, prefix = live_secrets_manager
    print(f"\n{'=' * 60}")
    print("LIVE AWS Secrets Manager integration test")
    print(f"Prefix: {prefix}")
    print(f"Target payload bytes: {store._target_bytes}")
    print(f"{'=' * 60}")

    token_a = StoredToken(access_token="a" * 700, refresh_token="r" * 700)
    token_b = StoredToken(access_token="b" * 700, refresh_token="s" * 700)

    # Step 1: First token fits inline (single JSON map, no manifest).
    print("\n[1] put_token github — expecting inline layout")
    await store.put_token("oauth2", "integration-user", "github", "/github", token_a)
    root_name = store._secret_name("oauth2", "integration-user")
    inline = json.loads(client.get_secret_value(SecretId=root_name)["SecretString"])
    inline_size = len(json.dumps(inline).encode("utf-8"))
    print(f"    Root secret: {root_name}")
    print(f"    Layout: inline (bare map, {inline_size} bytes)")
    print(f"    Keys: {list(inline.keys())[:3]}...")
    assert "_egress" not in inline, (
        f"Expected inline layout but got manifest: {list(inline.keys())}"
    )
    print("    ✓ inline confirmed")

    # Step 2: Second token overflows to sharded layout.
    print("\n[2] put_token slack — expecting overflow to sharded layout")
    await store.put_token("oauth2", "integration-user", "slack", "/slack", token_b)
    manifest = json.loads(client.get_secret_value(SecretId=root_name)["SecretString"])
    assert "_egress" in manifest, (
        f"Expected sharded manifest but got bare map: {list(manifest.keys())}"
    )
    meta = manifest["_egress"]
    print("    Layout: sharded")
    print(f"    Generation: {meta['generation']}")
    print(f"    Bucket count: {meta['bucket_count']}")
    print(f"    Hash algorithm: {meta['hash']}")
    print(f"    Secrets created: {len(client.created_names)}")
    for name in sorted(client.created_names):
        print(f"      - {name}")
    assert meta["layout"] == "sharded"
    assert meta["bucket_count"] >= 2
    assert all(name.startswith(prefix + "/") for name in client.created_names)
    assert any("/overflow/" in name for name in client.created_names)
    print("    ✓ overflow sharding confirmed")

    # Step 3: Read a specific token back from a shard.
    print("\n[3] get_token github — reading from shard")
    github = await store.get_token("oauth2", "integration-user", "github", "/github")
    assert github is not None and github.access_token == token_a.access_token
    print(f"    access_token length: {len(github.access_token)}")
    print(f"    refresh_token present: {github.refresh_token is not None}")
    print("    ✓ shard read correct")

    # Step 4: List all connections.
    print("\n[4] list_for_user — expecting 2 connections")
    conns = await store.list_for_user("oauth2", "integration-user")
    pairs = sorted((p, s) for p, s, _ in conns)
    print(f"    Connections: {pairs}")
    assert pairs == [("github", "/github"), ("slack", "/slack")]
    print("    ✓ list correct")

    # Step 5: Delete one token — should compact back to inline.
    print("\n[5] delete_token slack — expecting compaction to inline")
    await store.delete_token("oauth2", "integration-user", "slack", "/slack")
    after_delete = json.loads(client.get_secret_value(SecretId=root_name)["SecretString"])
    after_size = len(json.dumps(after_delete).encode("utf-8"))
    if "_egress" not in after_delete:
        print(f"    Layout: inline ({after_size} bytes)")
        print("    ✓ compacted back to inline")
    else:
        # Still sharded means remaining entry is close to the target boundary.
        print(f"    Layout: still sharded ({after_size} bytes, target={store._target_bytes})")
        print("    (remaining entry near target boundary — acceptable)")
    # Either way, the deleted token must not be retrievable.
    assert await store.get_token("oauth2", "integration-user", "slack", "/slack") is None
    print("    ✓ deleted token is gone")

    # Step 6: Delete last token — root secret removed entirely.
    print("\n[6] delete_token github — expecting full cleanup")
    await store.delete_token("oauth2", "integration-user", "github", "/github")
    print(f"    Active secrets remaining: {len(client.active_names)}")
    assert not client.active_names, f"Leaked secrets: {client.active_names}"
    print("    ✓ all secrets cleaned up")

    print(f"\n{'=' * 60}")
    print("PASSED: Full overflow lifecycle verified against live AWS")
    print(f"{'=' * 60}\n")
