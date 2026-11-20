"""
Behavior of the realm security baseline shared by the four Keycloak provisioners.

Each provisioner GETs a realm, runs REALM_SECURITY_BASELINE_JQ on it and PUTs
the result, so a stronger setting an operator chose is kept and a missing or
weaker one is raised to the baseline. These tests run each script's copy of
the program with jq on fixed realm representations.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent

PROVISIONERS = [
    "keycloak/setup/init-keycloak.sh",
    "keycloak/setup/upgrade-realm-for-dcr.sh",
    "charts/keycloak-configure/templates/configmap.yaml",
    "terraform/aws-ecs/scripts/init-keycloak.sh",
]

BASELINE_POLICY = "length(8) and notUsername and notEmail"

STRONGER_REALM = {
    "bruteForceProtected": True,
    "failureFactor": 3,
    "waitIncrementSeconds": 120,
    "maxFailureWaitSeconds": 3600,
    "permanentLockout": True,
    "passwordPolicy": "length(14) and passwordHistory(5)",
    "eventsExpiration": 7776000,
    "offlineSessionMaxLifespanEnabled": True,
    "offlineSessionMaxLifespan": 604800,
    "adminEventsDetailsEnabled": True,
}

WEAKER_REALM = {
    "bruteForceProtected": False,
    "failureFactor": 3,
    "waitIncrementSeconds": 5,
    "maxFailureWaitSeconds": 10,
    "permanentLockout": False,
    "passwordPolicy": "",
    "eventsExpiration": 3600,
    "offlineSessionMaxLifespanEnabled": False,
    "offlineSessionMaxLifespan": 5184000,
    "eventsEnabled": False,
    "adminEventsEnabled": False,
}


def _apply_baseline(
    path: str,
    realm: dict,
    gateway: bool,
    admin_expiry: bool = False,
) -> dict:
    source = (REPO_ROOT / path).read_text()
    program = re.search(r"REALM_SECURITY_BASELINE_JQ='(.*?)'", source, re.DOTALL)
    assert program, f"{path}: REALM_SECURITY_BASELINE_JQ not found"
    result = subprocess.run(  # nosec B603 B607 - hardcoded jq invocation on fixture data
        [
            "jq",
            "-c",
            "--argjson",
            "gateway",
            json.dumps(gateway),
            "--argjson",
            "adminExpiry",
            json.dumps(admin_expiry),
            program.group(1),
        ],
        input=json.dumps(realm),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(result.stdout)


@pytest.fixture(autouse=True)
def _require_jq() -> None:
    if shutil.which("jq") is None:
        pytest.skip("jq is not installed")


@pytest.mark.parametrize("path", PROVISIONERS)
def test_stronger_gateway_settings_are_kept(path: str) -> None:
    """Operator settings stricter than the baseline survive a setup run."""
    settings = _apply_baseline(path, STRONGER_REALM, gateway=True)
    assert settings["passwordPolicy"] == STRONGER_REALM["passwordPolicy"]
    assert settings["offlineSessionMaxLifespan"] == 604800
    assert settings["eventsExpiration"] == 7776000
    assert settings["permanentLockout"] is True
    assert settings["failureFactor"] == 3
    assert settings["waitIncrementSeconds"] == 120
    assert settings["maxFailureWaitSeconds"] == 3600
    # Stored request bodies can carry secrets, so this one is always forced off.
    assert settings["adminEventsDetailsEnabled"] is False


@pytest.mark.parametrize("path", PROVISIONERS)
def test_weaker_gateway_settings_are_raised(path: str) -> None:
    """Missing or weaker settings are replaced by the baseline."""
    settings = _apply_baseline(path, WEAKER_REALM, gateway=True)
    assert settings == {
        "bruteForceProtected": True,
        "permanentLockout": False,
        # Not protected before, so the existing (lower) number is not trusted.
        "failureFactor": 5,
        "waitIncrementSeconds": 60,
        "maxFailureWaitSeconds": 900,
        "eventsEnabled": True,
        "eventsExpiration": 2592000,
        "adminEventsEnabled": True,
        "adminEventsDetailsEnabled": False,
        "passwordPolicy": BASELINE_POLICY,
        "offlineSessionMaxLifespanEnabled": True,
        "offlineSessionMaxLifespan": 2592000,
    }


@pytest.mark.parametrize("path", PROVISIONERS)
def test_events_that_never_expire_are_left_alone(path: str) -> None:
    """Unset retention receives 30 days; explicit unlimited retention is preserved."""
    assert _apply_baseline(path, {}, gateway=True)["eventsExpiration"] == 2592000
    assert _apply_baseline(path, {"eventsExpiration": 0}, gateway=True)["eventsExpiration"] == 0


@pytest.mark.parametrize("path", PROVISIONERS)
def test_master_realm_gets_no_password_policy(path: str) -> None:
    """The master realm gets brute-force and quick-login settings only."""
    settings = _apply_baseline(path, {"maxDeltaTimeSeconds": 86400}, gateway=False)
    assert "passwordPolicy" not in settings
    assert "offlineSessionMaxLifespan" not in settings
    assert settings["maxDeltaTimeSeconds"] == 86400
    assert settings["quickLoginCheckMilliSeconds"] == 1000


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize(
    "expiry, expected", [(None, "2592000"), ("0", "0"), ("3600", "2592000"), ("7776000", "7776000")]
)
def test_admin_event_retention_preserves_explicit_unlimited_and_longer(path, expiry, expected):
    attributes = {"unrelated": "preserved"}
    if expiry is not None:
        attributes["adminEventsExpiration"] = expiry
    settings = _apply_baseline(path, {"attributes": attributes}, True, admin_expiry=True)
    assert settings["attributes"] == dict(attributes, adminEventsExpiration=expected)
    assert "attributes" not in _apply_baseline(
        path, {"attributes": attributes}, True, admin_expiry=False
    )
