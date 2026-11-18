"""
Behavior of the DCR-off keep-list shared by the three Keycloak provisioners.

With DCR off, each provisioner sorts every enabled client into: a Keycloak
built-in (left alone), a first-party web client kept by name, a public PKCE
IDE kept by validated shape, a client kept by the service-account-only shape rule,
or a client to disable. M2M names (mcp-gateway-m2m, agent-<id>-m2m, the bots)
are not trusted on their own: a registrant can choose any clientId. The rule
lives in one jq program per script; these tests run that program with jq on a
fixed client list so the three copies cannot drift apart.
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
]

SA_ONLY = {
    "enabled": True,
    "publicClient": False,
    "serviceAccountsEnabled": True,
    "standardFlowEnabled": False,
    "implicitFlowEnabled": False,
    "directAccessGrantsEnabled": False,
    "redirectUris": [],
}

BROWSER = {
    "enabled": True,
    "publicClient": True,
    "standardFlowEnabled": True,
    "redirectUris": ["http://localhost:3000/cb"],
}

# (clientId, representation, expected action)
CLIENTS = [
    ("account", dict(BROWSER), "builtin"),
    ("admin-cli", dict(BROWSER), "builtin"),
    ("mcp-gateway-web", dict(BROWSER, publicClient=False), "keep"),
    ("mcp-gateway-ide", dict(BROWSER), "disable"),
    ("agent-demo-m2m", dict(BROWSER), "disable"),
    ("agent-real-m2m", dict(SA_ONLY, clientAuthenticatorType="client-secret"), "shape"),
    ("mcp-gateway-m2m", dict(SA_ONLY, clientAuthenticatorType="client-secret"), "shape"),
    ("lob1-bot", dict(BROWSER, publicClient=False), "disable"),
    ("svc-secret", dict(SA_ONLY, clientAuthenticatorType="client-secret"), "shape"),
    ("svc-jwt", dict(SA_ONLY, clientAuthenticatorType="client-jwt"), "disable"),
    ("svc-x509", dict(SA_ONLY, clientAuthenticatorType="client-x509"), "disable"),
    (
        "svc-secret-with-redirect",
        dict(SA_ONLY, clientAuthenticatorType="client-secret", redirectUris=["https://x/cb"]),
        "disable",
    ),
    ("registered-browser-client", dict(BROWSER), "disable"),
    ("https://vscode.dev/oauth/client-metadata.json", dict(BROWSER), "disable"),
]


def _read_script(path: str) -> str:
    return (REPO_ROOT / path).read_text()


def _array_words(
    source: str,
    name: str,
) -> list[str]:
    match = re.search(rf"^\s*{name}=\((.*?)\)", source, re.MULTILINE | re.DOTALL)
    assert match, f"{name} not found"
    return match.group(1).split()


def _keep_lists(source: str) -> tuple[list[str], list[str]]:
    builtin = _array_words(source, "DCR_OFF_BUILTIN_CLIENTS")
    keep = []
    for word in _array_words(source, "DCR_OFF_KEEP_CLIENTS"):
        if word == '"${DCR_OFF_BUILTIN_CLIENTS[@]}"':
            keep.extend(builtin)
        else:
            keep.append(word)
    return builtin, keep


def _classify(
    path: str, overrides: dict | None = None, explicit_keep: list[str] | None = None
) -> dict[str, str]:
    source = _read_script(path)
    program = re.search(r"DCR_OFF_CLASSIFY_JQ='(.*?)'", source, re.DOTALL)
    assert program, f"{path}: DCR_OFF_CLASSIFY_JQ not found"
    builtin, keep = _keep_lists(source)
    keep.extend(explicit_keep or [])

    clients = [dict(rep, id=f"uuid-{cid}", clientId=cid) for cid, rep, _ in CLIENTS]
    if overrides is not None:
        clients = [dict(overrides, id="ide", clientId="mcp-gateway-ide", enabled=True)]
    else:
        clients.append(dict(BROWSER, id="uuid-off", clientId="already-disabled", enabled=False))

    result = subprocess.run(  # nosec B603 B607 - hardcoded jq invocation on fixture data
        [
            "jq",
            "-r",
            "--argjson",
            "keep",
            json.dumps(keep),
            "--argjson",
            "builtin",
            json.dumps(builtin),
            program.group(1),
        ],
        input=json.dumps(clients),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    actions = {}
    for line in result.stdout.splitlines():
        action, _uuid, client_id = line.split("\t")
        actions[client_id] = action
    return actions


@pytest.fixture(autouse=True)
def _require_jq() -> None:
    if shutil.which("jq") is None:
        pytest.skip("jq is not installed")


@pytest.mark.parametrize("path", PROVISIONERS)
def test_dcr_off_classification(path: str) -> None:
    """Each client lands in the expected bucket; disabled clients are skipped."""
    expected = {cid: action for cid, _, action in CLIENTS}
    assert _classify(path) == expected


@pytest.mark.parametrize("path", PROVISIONERS)
@pytest.mark.parametrize(
    "change, expected",
    [
        ({}, "ide"),
        ({"publicClient": False, "serviceAccountsEnabled": True}, "disable"),
        ({"attributes": {}}, "disable"),
        ({"implicitFlowEnabled": True}, "disable"),
        ({"redirectUris": ["http://localhost/*"]}, "disable"),
        ({"redirectUris": ["https://attacker.invalid/callback"]}, "disable"),
    ],
)
def test_ide_preserved_only_as_public_pkce_loopback_client(path, change, expected):
    client = dict(
        BROWSER, protocol="openid-connect", attributes={"pkce.code.challenge.method": "S256"}
    )
    client.update(change)
    assert _classify(path, client) == {"mcp-gateway-ide": expected}
    assert _classify(path, client, ["mcp-gateway-ide"]) == {"mcp-gateway-ide": "keep"}


@pytest.mark.parametrize("path", PROVISIONERS)
def test_cimd_client_requires_exact_keep_entry_when_dcr_is_off(path):
    client_id = "https://vscode.dev/oauth/client-metadata.json"
    assert _classify(path, explicit_keep=[client_id])[client_id] == "keep"
    assert _classify(path, explicit_keep=["https://vscode.dev/*"])[client_id] == "disable"
