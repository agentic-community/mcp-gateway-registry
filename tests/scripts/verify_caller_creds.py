#!/usr/bin/env python3
"""Confirm YOUR OpenAI key works through the gateway, and that the gateway holds none.

Two calls against the same proxied entity:

  1. WITH your key in Authorization  -> 200, model list. Your credential reached OpenAI.
  2. WITHOUT it                      -> 401. The gateway stores no default, so there is
                                        nothing to inject and the backend refuses.

The gateway JWT always rides X-Authorization. Sending it in Authorization instead
trips the equal-token guard and returns 401 from the gateway rather than OpenAI.

Usage (from the repository root):

    uv run python tests/scripts/verify_caller_creds.py
    uv run python tests/scripts/verify_caller_creds.py --entity openai-proxy-default-auth
    uv run python tests/scripts/verify_caller_creds.py --registry-url http://localhost
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import requests

DEFAULT_REGISTRY_URL = "http://localhost"
DEFAULT_TOKEN_FILE = ".token"
DEFAULT_API_KEY_FILE = ".scratchpad/.oai"
DEFAULT_ENTITY = "openai-proxy"
TIMEOUT_SECONDS = 60


def _read_secret(path: str) -> str:
    """Read a credential from a file, or exit with an operator-facing message.

    Args:
        path: File holding the secret.

    Returns:
        The secret, stripped.
    """
    p = Path(path)
    if not p.is_file() or not p.stat().st_size:
        sys.exit(f"FATAL: missing or empty credential file {path}")
    return p.read_text().strip()


def _gateway_token(token_file: str) -> str:
    """Read the gateway JWT and refuse an expired one.

    Args:
        token_file: JSON file holding tokens.access_token.

    Returns:
        The access token.
    """
    raw = _read_secret(token_file)
    token = json.loads(raw)["tokens"]["access_token"]
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    left = json.loads(base64.urlsafe_b64decode(payload))["exp"] - int(time.time())
    if left <= 0:
        sys.exit(f"FATAL: gateway token expired {abs(left) // 60} minutes ago; regenerate it")
    print(f"gateway token : valid for {left // 60} more minutes")
    return token


def _resolve_base(registry_url: str, token: str, entity: str) -> str:
    """Look up the entity's client URL instead of hand-assembling it.

    Args:
        registry_url: Registry base URL.
        token: Gateway JWT.
        entity: Record name to match.

    Returns:
        Absolute base URL with a trailing slash.
    """
    try:
        r = requests.get(
            f"{registry_url}/api/custom/rest-endpoint",
            headers={"Authorization": f"Bearer {token}"},
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        sys.exit(
            f"FATAL: cannot reach {registry_url} ({type(exc).__name__}).\n"
            "       Check the URL, and that the stack is up."
        )

    # Explain the likely cause rather than raising. A token is scoped to the
    # deployment whose auth-server signed it, and the failure mode that costs the
    # most time is pointing a valid-looking token at the wrong registry.
    if r.status_code == 401:
        sys.exit(
            f"FATAL: {registry_url} rejected the token from {DEFAULT_TOKEN_FILE!r} (401).\n"
            "       A gateway JWT is only valid for the deployment whose auth-server\n"
            "       signed it: iss and aud are identical across deployments, but each\n"
            "       signs with its own SECRET_KEY. Generate a token from THIS registry's\n"
            "       UI (Get JWT Token) and pass it with --token-file, or point\n"
            "       --registry-url at the deployment the current token came from."
        )
    if r.status_code == 403:
        sys.exit(
            f"FATAL: {registry_url} returned 403 for the custom-record list.\n"
            "       The caller lacks list_rest-endpoint_entity (or admin) on this registry."
        )
    if r.status_code == 404:
        sys.exit(
            f"FATAL: {registry_url} has no /api/custom/rest-endpoint route (404).\n"
            "       CUSTOM_ENTITY_TYPES_ENABLED is false there, so the custom-entity\n"
            "       routers are never registered (registry/main.py), or the\n"
            "       'rest-endpoint' type does not exist yet."
        )
    if r.status_code != 200:
        sys.exit(f"FATAL: {registry_url} returned HTTP {r.status_code}: {r.text[:200]}")

    for rec in r.json().get("records", []):
        if rec.get("name") == entity:
            if not rec.get("proxy_client_url"):
                sys.exit(f"FATAL: {entity} is not proxied (no proxy_client_url)")
            print(f"entity        : {entity}")
            print(
                f"upstream auth : names={rec.get('custom_header_names')} "
                f"overridable={rec.get('custom_header_overridable_names')}"
            )
            # Trailing slash matters: the nginx location ends in one, so the bare
            # path answers 301 and a POST would be downgraded to GET.
            return registry_url + rec["proxy_client_url"].rstrip("/") + "/"

    names = sorted(x.get("name", "?") for x in r.json().get("records", []))
    sys.exit(
        f"FATAL: no proxied record named {entity!r} on {registry_url}.\n"
        f"       records present: {', '.join(names) or '(none)'}"
    )


def main() -> int:
    """Run both calls and report whether the caller's credential is the one in use."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry-url", default=DEFAULT_REGISTRY_URL)
    ap.add_argument("--token-file", default=DEFAULT_TOKEN_FILE)
    ap.add_argument("--api-key-file", default=DEFAULT_API_KEY_FILE)
    ap.add_argument("--entity", default=DEFAULT_ENTITY)
    args = ap.parse_args()

    token = _gateway_token(args.token_file)
    api_key = _read_secret(args.api_key_file)
    base = _resolve_base(args.registry_url, token, args.entity)
    print(f"client URL    : {base}\n")

    # requests sets Accept-Encoding and decompresses for us. Plain curl does not,
    # which is why the same call prints unreadable gzip bytes without --compressed.
    with_key = requests.get(
        f"{base}v1/models",
        headers={"X-Authorization": f"Bearer {token}", "Authorization": f"Bearer {api_key}"},
        timeout=TIMEOUT_SECONDS,
    )
    without_key = requests.get(
        f"{base}v1/models",
        headers={"X-Authorization": f"Bearer {token}"},
        timeout=TIMEOUT_SECONDS,
    )

    ok_with = with_key.status_code == 200 and "data" in with_key.json()
    models = len(with_key.json().get("data", [])) if ok_with else 0
    print(
        f"with your key    : HTTP {with_key.status_code}  "
        + (
            f"{models} models, first={with_key.json()['data'][0]['id']}"
            if ok_with
            else with_key.text[:120]
        )
    )

    detail = ""
    try:
        body = without_key.json()
        detail = body.get("error", {}).get("message") or body.get("detail") or ""
    except ValueError:
        detail = without_key.text[:120]
    print(f"without your key : HTTP {without_key.status_code}  {detail}")

    print()
    if ok_with and without_key.status_code == 401:
        print("PASS: your key is the credential in use, and the gateway stores none.")
        return 0
    if ok_with and without_key.status_code == 200:
        print("FAIL: the call succeeded WITHOUT your key, so a stored operator default")
        print("      is being injected. Clear it, then re-add the name only:")
        print('        PATCH .../upstream-headers  {"custom_headers": []}')
        print(
            '        PATCH .../upstream-headers  {"custom_headers": [{"name": "Authorization", "overridable": true}]}'
        )
        return 1
    print("FAIL: the authenticated call did not succeed. 401 from the GATEWAY means the")
    print("      equal-token guard fired (JWT in both headers); 401 from OpenAI means the")
    print("      key is bad; 403 means the group lacks an http verb grant on this entity.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
