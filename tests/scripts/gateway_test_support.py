#!/usr/bin/env python3
"""Shared plumbing for the gateway generic-proxy test clients.

Both ``openai_gateway_client.py`` and ``bedrock_gateway_client.py`` drive a proxied
custom-entity record through the gateway; only the upstream API shape differs.
Everything that is about the GATEWAY -- credential split, entity discovery,
per-verb scope authorization, failure interpretation -- lives here so the two
clients cannot drift apart.

The credential split is the load-bearing convention (see
docs/design/gateway-generic-proxy.md, "Caller header passthrough"):

- gateway JWT  -> ``X-Authorization``  authenticates ingress.
- backend key  -> ``Authorization``    forwarded to the upstream, admitted only
  because the entity registers ``Authorization`` as caller-OVERRIDABLE. Sending
  the gateway JWT in both trips the equal-token guard (401).
"""

import argparse
import base64
import binascii
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_TOKEN_FILE: str = ".token"
DEFAULT_REGISTRY_URL: str = "http://localhost"
DEFAULT_SCOPE_GROUP: str = "registry-admins"
REQUEST_TIMEOUT_SECONDS: int = 60


class GatewayClientError(RuntimeError):
    """A prerequisite or gateway-side failure with an operator-facing message."""


def configure_logging(
    debug: bool = False,
) -> None:
    """Install the shared log format used by both clients.

    Args:
        debug: Enable DEBUG level.
    """
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def read_access_token(
    token_file: str,
) -> str:
    """Read the gateway JWT from a raw-text or nested-JSON token file.

    Accepts the three shapes the repo's tooling produces: a bare JWT,
    ``{"access_token": ...}``, and the nested ``{"tokens": {"access_token": ...}}``
    written by the credential providers.

    Args:
        token_file: Path to the token file.

    Returns:
        The bearer token value, stripped.

    Raises:
        GatewayClientError: The file is missing, empty, or carries no token.
    """
    path = Path(token_file)
    if not path.exists():
        raise GatewayClientError(
            f"Gateway token file not found: {token_file}\n"
            "Generate one, or point --token-file at an existing token."
        )
    raw = path.read_text().strip()
    if not raw:
        raise GatewayClientError(f"Gateway token file is empty: {token_file}")

    try:
        blob = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    if isinstance(blob, dict):
        nested = blob.get("tokens")
        token = blob.get("access_token") or (
            nested.get("access_token") if isinstance(nested, dict) else None
        )
        if token:
            return str(token).strip()
    raise GatewayClientError(f"Could not find an access token in {token_file}")


def token_lifetime_seconds(
    token: str,
) -> int | None:
    """Return seconds until the JWT expires, or None if it cannot be read.

    Decodes the payload WITHOUT verifying the signature: a usability check (a
    stale token yields a confusing 401), never an authorization decision.

    Args:
        token: The bearer token.

    Returns:
        Remaining lifetime in seconds, or None when undecodable / no ``exp``.
    """
    parts = token.split(".")
    if len(parts) != 3:
        return None
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
    exp = claims.get("exp")
    return int(exp - time.time()) if isinstance(exp, int | float) else None


def read_api_key(
    api_key_file: str | None,
    env_var: str,
) -> str:
    """Resolve the upstream API key from the environment or a file.

    The environment variable wins so CI can inject the key without touching disk.

    Args:
        api_key_file: Path to a file containing the key (raw text), or None.
        env_var: Environment variable consulted first (e.g. ``OPENAI_API_KEY``).

    Returns:
        The API key, stripped.

    Raises:
        GatewayClientError: Neither source yields a key.
    """
    env_key = (os.environ.get(env_var) or "").strip()
    if env_key:
        logger.info("Backend key : loaded from environment (sent as Authorization)")
        return env_key

    if not api_key_file:
        raise GatewayClientError(
            f"No backend key: set {env_var} or pass --api-key-file.\n"
            "It is forwarded as the caller-overridable Authorization header."
        )
    path = Path(api_key_file)
    if not path.exists():
        raise GatewayClientError(f"Backend key file not found: {api_key_file} (or set {env_var})")
    key = path.read_text().strip()
    if not key:
        raise GatewayClientError(f"Backend key file is empty: {api_key_file}")
    logger.info("Backend key : loaded from file (sent as Authorization)")
    return key


def api_session(
    token: str,
) -> requests.Session:
    """Build a session that authenticates to the registry management API.

    The registry API itself takes the gateway JWT in ``Authorization``; only the
    proxied hop needs the ``X-Authorization`` split.

    Args:
        token: The gateway JWT.

    Returns:
        A session with the bearer header applied.
    """
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    return session


def get_json(
    session: requests.Session,
    url: str,
) -> Any:
    """GET a registry endpoint and return the decoded JSON body.

    Args:
        session: An authenticated session.
        url: Absolute URL to fetch.

    Returns:
        The decoded JSON body.

    Raises:
        GatewayClientError: Non-2xx status or an undecodable body.
    """
    response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    if response.status_code >= 400:
        raise GatewayClientError(f"GET {url} -> HTTP {response.status_code}: {response.text[:300]}")
    try:
        return response.json()
    except ValueError as exc:
        raise GatewayClientError(f"GET {url} returned non-JSON: {response.text[:200]}") from exc


def proxied_custom_records(
    session: requests.Session,
    registry_url: str,
) -> list[dict[str, Any]]:
    """List every proxied custom-entity record the caller can see.

    Walks the custom-type descriptors, then each type's records, keeping rows that
    carry ``is_proxied`` plus the server-derived ``proxy_client_url``. Same read
    model ``registry_management.py custom-record-list`` prints: header NAMES are
    present, values never are.

    Args:
        session: An authenticated session.
        registry_url: Registry base URL, no trailing slash.

    Returns:
        Matching record dicts, each with ``entity_type`` filled in.
    """
    types_body = get_json(session, f"{registry_url}/api/custom-types")
    type_names = [t["name"] for t in types_body.get("custom_types", []) if t.get("name")]

    found: list[dict[str, Any]] = []
    for type_name in type_names:
        body = get_json(session, f"{registry_url}/api/custom/{type_name}?limit=200")
        records = body.get("records") if isinstance(body, dict) else body
        for record in records or []:
            if record.get("is_proxied") and record.get("proxy_client_url"):
                record.setdefault("entity_type", type_name)
                found.append(record)
    return found


def select_record(
    records: list[dict[str, Any]],
    selector: str | None,
    target_hint: str | None = None,
) -> dict[str, Any]:
    """Pick the record to test, by selector, backend hint, or sole candidate.

    Args:
        records: Proxied custom records.
        selector: Case-insensitive substring matched against name, path, and
            client URL. Takes precedence over ``target_hint``.
        target_hint: Substring of ``proxy_target_url`` identifying this client's
            backend (e.g. ``api.openai.com``), used when no selector is given so a
            registry holding several proxied entities still resolves cleanly.

    Returns:
        The selected record.

    Raises:
        GatewayClientError: Nothing matched, or the choice is ambiguous.
    """
    if not records:
        raise GatewayClientError(
            "No proxied custom records found. Register one with is_proxied=true, a "
            "proxy_target_url, and an overridable Authorization upstream header "
            "(UI, or 'api/registry_management.py custom-proxy-create')."
        )

    if selector:
        needle = selector.lower()
        matches = [
            r
            for r in records
            if needle in str(r.get("name", "")).lower()
            or needle in str(r.get("path", "")).lower()
            or needle in str(r.get("proxy_client_url", "")).lower()
        ]
        criterion = f"--entity {selector!r}"
    elif target_hint and len(records) > 1:
        matches = [r for r in records if target_hint in str(r.get("proxy_target_url", "")).lower()]
        criterion = f"a backend containing {target_hint!r}"
    else:
        matches = records
        criterion = "the only proxied record"

    if not matches:
        known = ", ".join(f"{r.get('entity_type')}/{r.get('name')}" for r in records)
        raise GatewayClientError(f"No proxied record matches {criterion}. Known: {known}")
    if len(matches) > 1:
        known = ", ".join(f"{r.get('entity_type')}/{r.get('name')}" for r in matches)
        raise GatewayClientError(f"Ambiguous target ({known}); narrow it with --entity.")
    return matches[0]


def authz_key(
    record: dict[str, Any],
) -> str:
    """Return the scope key the generic hop authorizes against.

    ``/validate`` composes ``{entity_type}/{registered_path}`` from the server-set
    nginx markers, so a custom record whose path already carries its type token
    yields a doubled-looking key (``rest-endpoint/rest-endpoint/<uuid>``). That is
    the exact string a scope rule must name -- the bare path 403s.

    Args:
        record: A proxied record carrying ``entity_type`` and ``path``.

    Returns:
        The canonical authz key.
    """
    return f"{record['entity_type']}/{str(record.get('path', '')).strip('/')}"


def report_record(
    record: dict[str, Any],
    expect_streaming: bool = False,
) -> None:
    """Log the proxy read model and warn about setups that cannot work.

    Args:
        record: The selected proxied record.
        expect_streaming: True when the caller intends to run a streaming check,
            so a buffered entity is worth warning about.
    """
    overridable = record.get("custom_header_overridable_names") or []
    logger.info("Entity      : %s/%s", record.get("entity_type"), record.get("name"))
    logger.info("Client URL  : %s", record.get("proxy_client_url"))
    logger.info("Backend     : %s", record.get("proxy_target_url") or "(redacted for non-admin)")
    logger.info("Streaming   : %s", bool(record.get("proxy_streaming")))
    logger.info(
        "Header names: %s (overridable: %s)",
        record.get("custom_header_names") or [],
        overridable,
    )
    logger.info("Authz key   : %s", authz_key(record))
    if record.get("proxy_connect_notes"):
        logger.info("Notes       : %s", record["proxy_connect_notes"])

    if "authorization" not in {str(n).lower() for n in overridable}:
        logger.warning(
            "Authorization is NOT registered as an overridable upstream header: the "
            "backend key will be dropped on egress and the upstream will answer 401/403."
        )
    if expect_streaming and not record.get("proxy_streaming"):
        logger.warning(
            "proxy_streaming is false: a streaming check still returns a body, but the "
            "hop buffers it, so the incremental-delivery assertion will fail."
        )


def _rule_grants(
    rule: dict[str, Any],
    key: str,
    verbs: tuple[str, ...],
) -> bool:
    """Report whether one ``server_access`` rule authorizes the verbs on the key.

    Mirrors ``validate_server_tool_access``: an HTTP verb matches the methods list
    case-insensitively, or the distinct ``http:*`` token. The legacy ``all``/``*``
    methods wildcard does NOT grant a verb (the no-escalation split), and
    ``server:"*"`` matches any entity.

    Args:
        rule: One rule from a group's ``server_access``.
        key: The entity's authz key.
        verbs: Verbs that must all be authorized.

    Returns:
        True when the rule grants every verb.
    """
    server = str(rule.get("server", "")).strip("/")
    if server not in {"*", key}:
        return False
    methods = {str(m).upper() for m in rule.get("methods", [])}
    if "HTTP:*" in methods:
        return True
    return all(v in methods for v in verbs)


def ensure_scope_grant(
    session: requests.Session,
    registry_url: str,
    group: str,
    key: str,
    verbs: tuple[str, ...],
    apply_changes: bool,
) -> bool:
    """Check -- and optionally add -- the per-verb scope rule for an entity.

    Args:
        session: An authenticated admin session.
        registry_url: Registry base URL, no trailing slash.
        group: Group whose scope config carries the rule.
        key: The entity's authz key.
        verbs: Verbs the client needs.
        apply_changes: When True, PATCH the group to append a missing rule.

    Returns:
        True when the grant is in place (already present, or just written).

    Raises:
        GatewayClientError: The group cannot be read, or the PATCH failed.
    """
    detail = get_json(session, f"{registry_url}/api/management/iam/groups/{group}")
    rules = detail.get("server_access") or []

    if any(_rule_grants(r, key, verbs) for r in rules):
        logger.info("Scope grant : present in group '%s' for %s", group, ", ".join(verbs))
        return True

    remediation = (
        f"Group '{group}' does not authorize {', '.join(verbs)} on '{key}'.\n"
        "Add this rule to its server_access (IAM scope editor, or re-run with "
        "--ensure-scope):\n"
        f'  {{"server": "{key}", "methods": {json.dumps(list(verbs))}}}\n'
        'Note: an existing methods:["all"] rule does NOT authorize HTTP verbs -- '
        "that split is the deliberate no-escalation guard."
    )
    if not apply_changes:
        logger.warning(remediation)
        return False

    new_rules = [*rules, {"server": key, "methods": list(verbs), "tools": []}]
    response = session.patch(
        f"{registry_url}/api/management/iam/groups/{group}",
        json={"scope_config": {"server_access": new_rules}},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code >= 400:
        raise GatewayClientError(
            f"Adding the scope rule failed: HTTP {response.status_code}: "
            f"{response.text[:300]}\n{remediation}"
        )
    logger.info("Scope grant : added %s on '%s' to group '%s'", ", ".join(verbs), key, group)
    return True


# Gateway-layer meaning of each status a client may see. The upstream can also
# emit these (a 401 from the BACKEND means the passthrough worked and the key is
# bad), so both readings are given where they are ambiguous.
STATUS_HINTS: dict[int, str] = {
    401: (
        "401 from the GATEWAY means the equal-token guard fired (the outbound "
        "Authorization equals the gateway credential) or the JWT is stale. 401 from "
        "the BACKEND means the key is bad -- the passthrough itself worked."
    ),
    403: (
        "Scope denial or the CSRF gate at the gateway (re-run with --ensure-scope, and "
        "keep the gateway JWT in X-Authorization). From the BACKEND: the key lacks "
        "permission for this model/action."
    ),
    404: (
        "No route: the feature is off, nginx has not reloaded, or the client path is "
        "wrong. Check GATEWAY_GENERIC_PROXY_ENABLED and restart the registry. From the "
        "BACKEND: wrong upstream sub-path or model id."
    ),
    413: "Streaming byte cap (GATEWAY_GENERIC_STREAM_MAX_BYTES) exceeded.",
    502: (
        "The hop refused to forward: upstream-header vend failure (fail-closed) or an "
        "egress-guard block on the pinned target."
    ),
    503: (
        "Feature latch off (startup egress self-check failed -- see "
        "GATEWAY_EGRESS_SELFCHECK_ENABLED) or no stream slot within "
        "GATEWAY_GENERIC_ACQUIRE_TIMEOUT_SECONDS."
    ),
    504: "Upstream timeout, or the absolute streaming duration ceiling fired.",
}


def explain_status(
    status: int | None,
    body: str = "",
) -> None:
    """Log the gateway-layer meaning of an HTTP status.

    Args:
        status: The HTTP status code, if known.
        body: Response body excerpt to include.
    """
    logger.error("Upstream/gateway reported HTTP %s: %s", status, body[:300])
    if status in STATUS_HINTS:
        logger.error("Hint: %s", STATUS_HINTS[status])


def add_common_arguments(
    parser: argparse.ArgumentParser,
    api_key_env: str,
    default_api_key_file: str | None = None,
) -> None:
    """Register the CLI arguments shared by every gateway test client.

    Args:
        parser: The parser to extend.
        api_key_env: Environment variable naming the backend key.
        default_api_key_file: Default path for ``--api-key-file``, if any.
    """
    parser.add_argument(
        "--registry-url",
        default=DEFAULT_REGISTRY_URL,
        help=f"Gateway/registry base URL (default: {DEFAULT_REGISTRY_URL})",
    )
    parser.add_argument(
        "--token-file",
        default=DEFAULT_TOKEN_FILE,
        help=(
            "File holding the gateway JWT, sent as X-Authorization "
            f"(default: {DEFAULT_TOKEN_FILE})"
        ),
    )
    parser.add_argument(
        "--api-key-file",
        default=default_api_key_file,
        help=(
            f"File holding the backend key, sent as Authorization; {api_key_env} wins"
            + (f" (default: {default_api_key_file})" if default_api_key_file else "")
        ),
    )
    parser.add_argument(
        "--entity",
        help="Substring selecting the proxied custom record (name, path, or client URL)",
    )
    parser.add_argument(
        "--client-path",
        help=(
            "Explicit gateway client path (e.g. /gateway/rest-endpoint/<uuid>); skips "
            "custom-record discovery and the scope check"
        ),
    )
    parser.add_argument(
        "--ensure-scope",
        action="store_true",
        help="Write the missing per-verb scope rule to --scope-group instead of only reporting it",
    )
    parser.add_argument(
        "--scope-group",
        default=DEFAULT_SCOPE_GROUP,
        help=f"Group whose scope config carries the rule (default: {DEFAULT_SCOPE_GROUP})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=REQUEST_TIMEOUT_SECONDS,
        help=f"Per-request timeout in seconds (default: {REQUEST_TIMEOUT_SECONDS})",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")


def preflight(
    args: argparse.Namespace,
    api_key_env: str,
    verbs: tuple[str, ...],
    target_hint: str | None = None,
    expect_streaming: bool = False,
) -> tuple[str, str, str]:
    """Resolve credentials and the client path, reporting every prerequisite.

    Args:
        args: Parsed arguments carrying the common options.
        api_key_env: Environment variable naming the backend key.
        verbs: HTTP verbs the client will use (checked against the scope config).
        target_hint: Backend substring used to disambiguate discovery.
        expect_streaming: Whether a streaming check will run.

    Returns:
        ``(gateway_token, api_key, client_path)``.

    Raises:
        GatewayClientError: Any prerequisite is missing or unusable.
    """
    registry_url = args.registry_url.rstrip("/")

    token = read_access_token(args.token_file)
    lifetime = token_lifetime_seconds(token)
    if lifetime is not None:
        if lifetime <= 0:
            raise GatewayClientError(
                f"The gateway token in {args.token_file} expired {-lifetime}s ago; "
                "refresh it before testing."
            )
        logger.info(
            "Gateway JWT : %s (expires in %ds, sent as X-Authorization)",
            args.token_file,
            lifetime,
        )

    api_key = read_api_key(args.api_key_file, api_key_env)
    session = api_session(token)

    if args.client_path:
        client_path = args.client_path
        logger.info("Client URL  : %s (explicit; skipping record + scope checks)", client_path)
        return token, api_key, client_path

    record = select_record(proxied_custom_records(session, registry_url), args.entity, target_hint)
    report_record(record, expect_streaming=expect_streaming)
    ensure_scope_grant(
        session=session,
        registry_url=registry_url,
        group=args.scope_group,
        key=authz_key(record),
        verbs=verbs,
        apply_changes=args.ensure_scope,
    )
    return token, api_key, str(record["proxy_client_url"])


def run_checks(
    checks: dict[str, Any],
    selected: list[str],
) -> int:
    """Run the selected checks, isolating failures, and log a summary.

    Args:
        checks: Mapping of check name -> zero-arg callable returning a bool.
        selected: Names to run, in order.

    Returns:
        Process exit code: 0 when all passed, 1 otherwise.
    """
    failures = 0
    for name in selected:
        try:
            if not checks[name]():
                failures += 1
        except GatewayClientError as exc:
            logger.error("%s: %s", name, exc)
            failures += 1
        except Exception as exc:  # noqa: BLE001 - report any transport/SDK failure per check
            logger.error("%s: FAILED - %s: %s", name, type(exc).__name__, exc)
            status = getattr(exc, "status_code", None)
            if status is not None:
                response = getattr(exc, "response", None)
                explain_status(status, getattr(response, "text", "") or "")
            failures += 1

    logger.info(
        "Summary     : %d/%d checks passed (%s)",
        len(selected) - failures,
        len(selected),
        ", ".join(selected),
    )
    return 1 if failures else 0
