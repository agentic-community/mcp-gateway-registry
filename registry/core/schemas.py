import re
from datetime import datetime
from typing import Any, Literal, cast
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from registry.constants import DeploymentType, LocalRuntimeType, TransportType
from registry.schemas.agent_models import AgentProvider
from registry.schemas.registry_card import LifecycleStatus

_IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


_RFC7230_TOKEN_RE = re.compile(r"[A-Za-z0-9!#$%&'*+\-.^_`|~]+")


# A registered server path is interpolated into generated nginx config (location
# directives, `set` directives, regex maps) and into per-server discovery URLs.
# Restrict it to a safe slug so a hostile value can never carry nginx
# metacharacters (quotes, semicolons, braces, whitespace, newlines, backslashes)
# that could break out of a directive/string context. Allowed: alphanumerics and
# `-` `_` `.` `/` (multi-segment paths, optional leading/trailing slash -- the
# registration write path normalizes the leading slash, and some callers build
# ServerInfo with the bare segment). This is the systemic guard behind the
# per-site nginx sanitization in NginxConfigService (defense-in-depth: reject the
# dangerous character class at the model, escape at every render site).
_SERVER_PATH_RE = re.compile(r"^[A-Za-z0-9._\-/]+$")


# obo_exchange target-audience GUID handling. A GUID could be an internal server's
# app id OR a Microsoft first-party resource, so form matters:
#   - A BARE GUID is how first-party resources are DIRECTLY addressable (Graph
#     00000003-..., ARM 797f4846-..., Key Vault cfa8b339-..., Storage, SQL, ...).
#     The set cannot be enumerated safely, so ANY bare GUID is rejected by shape
#     and only permitted via the operator allowlist (EGRESS_OBO_ALLOWED_AUDIENCES).
#   - The ``api://<guid>`` form is Entra's standard auto-generated App ID URI for a
#     CUSTOM (tenant-local) app, so it is accepted by shape -- with one caveat: we
#     do NOT rely on the assumption that Entra never scheme-normalizes
#     ``api://<appId>`` to a bare-GUID first-party service-principal-name match.
#     As defense-in-depth we still reject the ``api://`` form of the KNOWN
#     first-party app-id GUIDs below, so the documented confused-deputy targets
#     (Graph/ARM/Key Vault) are blocked in both bare and api:// spellings even if
#     that normalization exists. A genuinely-GUID first-party resource outside
#     this set would still require the operator to have granted the gateway app
#     delegated permissions on it AND to register the obo server for it.
_GUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# Known Microsoft first-party resource app-id GUIDs -- rejected as an obo target in
# BOTH bare and ``api://<guid>`` forms (see _GUID_RE comment). These are also an
# ALWAYS-ON floor: EGRESS_OBO_ALLOWED_AUDIENCES cannot re-enable them (see
# _is_first_party_obo_audience / _is_disallowed_obo_audience). There is no
# legitimate reason to OBO-exchange into Graph/ARM/Key Vault and forward that
# broad delegated token to an MCP server's upstream -- that IS the confused deputy
# the control exists to stop -- so the operator override does not extend to them.
_FIRST_PARTY_APPID_GUIDS: frozenset[str] = frozenset(
    {
        "00000003-0000-0000-c000-000000000000",  # Microsoft Graph
        "00000002-0000-0000-c000-000000000000",  # Azure AD Graph (legacy)
        "797f4846-ba00-4fd7-ba43-dac1f8f63013",  # Azure Resource Manager
        "cfa8b339-82a2-471a-a3c9-0fc0be7a4093",  # Azure Key Vault
    }
)

# Host portions of the same first-party resources' canonical https:// audiences,
# for the always-on floor (an allowlisted 'https://graph.microsoft.com' must still
# be blocked). Matched against the target's host regardless of scheme/port/path.
_FIRST_PARTY_HOSTS: frozenset[str] = frozenset(
    {
        "graph.microsoft.com",
        "graph.microsoft.us",  # GCC-High / DoD
        "dod-graph.microsoft.us",
        "graph.microsoft.de",  # legacy Germany
        "microsoftgraph.chinacloudapi.cn",  # China
        "management.azure.com",
        "management.core.windows.net",
        "management.usgovcloudapi.net",
        "management.chinacloudapi.cn",
        "vault.azure.net",
        "vault.usgovcloudapi.net",
        "vault.azure.cn",
        "vault.microsoftazure.de",
    }
)


def _is_first_party_obo_audience(target: str) -> bool:
    """True if ``target`` names a blocked first-party resource in ANY form.

    Detects Microsoft Graph / Azure AD Graph / ARM / Key Vault whether expressed
    as a bare app-id GUID, an ``api://<guid>`` URI, or an ``http(s)://<host>``
    URL (any port/path). This is the ALWAYS-ON floor: it is checked before the
    operator allowlist, so ``EGRESS_OBO_ALLOWED_AUDIENCES`` cannot re-enable these.
    ``target`` must already be lowercased/stripped.
    """
    # api://<guid> or bare guid
    bare = target[len("api://") :] if target.startswith("api://") else target
    if bare in _FIRST_PARTY_APPID_GUIDS:
        return True
    # http(s)://<host>[:port][/...] -- extract the host and compare.
    if target.startswith("https://") or target.startswith("http://"):
        rest = target.split("://", 1)[1]
        host = rest.split("/", 1)[0].split(":", 1)[0]
        if host in _FIRST_PARTY_HOSTS:
            return True
    return False


# The authority of an accepted ``api://<authority>`` target, and the accepted bare
# (schemeless) client-id form. Deliberately narrow so the shape rule fails CLOSED:
# only these two shapes are ever accepted; every other/unknown form is dropped.
# Allowed chars are those Entra App ID URIs and Keycloak client-ids actually use
# (alphanumerics and ``. _ - :`` for host:port-style App ID URIs). No slashes,
# quotes, whitespace, or scheme markers -- so ``spiffe://x``, ``urn:uuid:...``,
# ``{guid}``, ``http(s)://...`` etc. never match.
_OBO_API_AUTHORITY_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
_OBO_BARE_CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _gateway_own_client_id() -> str:
    """The gateway's own IdP client id, resolved by the configured auth provider.

    Used to reject an obo_exchange target_audience that points back at the
    gateway's own app (a same-app OBO is not a valid exchange). Returns "" when
    the provider's client id is not configured (then no same-app check applies).
    """
    from registry.core.config import settings

    provider = (settings.auth_provider or "").lower()
    if provider == "entra":
        return settings.entra_client_id or ""
    if provider == "keycloak":
        return settings.keycloak_client_id or ""
    return ""


def _gateway_own_audiences() -> set[str]:
    """All audience forms that refer to the gateway's own IdP app.

    Includes the bare client id, the Entra ``api://<client_id>`` App ID URI, and
    the gateway's own public resource URL (``registry_url``, which is an App ID
    URI the gateway app owns and advertises in its PRM). Covering these means a
    same-app OBO cannot be smuggled past the check by using a non-client-id
    audience form the gateway app also owns. Returned lower-cased for
    case-insensitive comparison (Entra audiences are GUIDs/URIs, not
    case-sensitive).
    """
    from registry.core.config import settings

    own = _gateway_own_client_id().strip().lower()
    auds: set[str] = set()
    if own:
        auds.add(own)
        auds.add(f"api://{own}")
    # The gateway's own public resource URL is an audience of the gateway app
    # (advertised in its PRM as the gateway-wide resource), so a target_audience
    # equal to it is still a same-app OBO (Entra rejects it at runtime).
    registry_url = (getattr(settings, "registry_url", "") or "").strip().lower()
    if registry_url:
        auds.add(registry_url)
        auds.add(registry_url.rstrip("/"))
    return auds


def _is_gateway_own_audience(target_audience: str) -> bool:
    """True if target_audience refers to the gateway's own IdP app.

    Matches the bare client id, the Entra ``api://<client_id>`` App ID URI form,
    and any configured gateway App ID URI, case-insensitively.
    """
    own = _gateway_own_audiences()
    if not own:
        return False
    target = target_audience.strip().lower()
    return target in own or target.rstrip("/") in own


def _obo_audience_allowlist() -> set[str]:
    """Operator allowlist of permitted obo_exchange target_audience values.

    From ``EGRESS_OBO_ALLOWED_AUDIENCES`` (whitespace-separated). When non-empty
    it is the authoritative positive control: only these exact audiences (compared
    case-insensitively, trailing slash ignored) may be registered. Empty => the
    shape heuristic in :func:`_is_disallowed_obo_audience` applies instead.
    """
    from registry.core.config import settings

    raw = getattr(settings, "egress_obo_allowed_audiences", "") or ""
    return {a.strip().lower().rstrip("/") for a in raw.split() if a.strip()}


def _is_disallowed_obo_audience(target_audience: str) -> bool:
    """True if target_audience is not an acceptable obo_exchange target.

    An obo target must be an internal MCP server's OWN IdP audience, never a
    shared first-party resource (Microsoft Graph / ARM / Key Vault / any sovereign
    cloud), or the exchange would mint a broadly-scoped delegated token and forward
    it to the server's upstream (confused-deputy token exfiltration).

    A denylist of hosts/GUIDs cannot express this (new first-party hosts, ports,
    path suffixes, sovereign clouds, and the full first-party app-id GUID set all
    evade it), so this is a positive control that fails closed:

    0. A fixed set of first-party resources (Graph/ARM/Key Vault, in any form) is
       ALWAYS disallowed, checked before the operator allowlist -- the allowlist
       cannot re-enable them.
    1. If the operator set ``EGRESS_OBO_ALLOWED_AUDIENCES``, the target must be in
       that exact set (authoritative for everything not blocked by (0)). Anything
       else is disallowed.
    2. Otherwise a shape allowlist: ONLY two shapes are accepted, everything else
       (unknown schemes, host URLs, bare GUIDs, braced/urn forms, garbage) is
       disallowed:
         a. ``api://<authority>`` where the authority is a well-formed token --
            an Entra App ID URI for an internal server. This is the standard form,
            INCLUDING the auto-generated ``api://<app-guid>`` (e.g.
            ``api://aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee``): an ``api://`` string
            names a custom (tenant-local) app registration, so no per-server
            allowlist entry is required for the normal case. As defense-in-depth we
            still reject the ``api://`` form of the KNOWN first-party app-id GUIDs
            (Graph/ARM/Key Vault -- see _FIRST_PARTY_APPID_GUIDS), so the documented
            confused-deputy targets are blocked in both bare and api:// spellings
            regardless of Entra's internal SPN-matching semantics.
         b. a bare, schemeless non-GUID client-id token (Keycloak -- e.g.
            ``outlook-mcp-client``).
       A BARE GUID is NOT accepted by shape -- that IS how first-party resources
       are directly addressable (Graph 00000003-..., ARM 797f4846-..., ...), and
       the set is not safely enumerable -- so a bare-GUID target must be pinned via
       the operator allowlist in (1). ``http(s)://`` hosts (all shared first-party
       APIs) and every unrecognized form fail the allowlist and are dropped.
    """
    target = target_audience.strip().lower().rstrip("/")
    if not target:
        # Empty is handled as "missing" by the caller; not this function's job.
        return False

    # Always-on floor (checked BEFORE the operator allowlist): the fixed set of
    # first-party resources (Graph/ARM/Key Vault, any spelling) are never a valid
    # obo target, and EGRESS_OBO_ALLOWED_AUDIENCES cannot re-enable them. There is
    # no legitimate reason to OBO-exchange into them and forward that broad
    # delegated token to an MCP server's upstream.
    if _is_first_party_obo_audience(target):
        return True

    allowlist = _obo_audience_allowlist()
    if allowlist:
        return target not in allowlist

    # Shape allowlist (no operator allowlist). Fail closed: disallowed UNLESS the
    # target matches one of exactly two accepted shapes.
    if target.startswith("api://"):
        authority = target[len("api://") :]
        # Reject a malformed/empty authority. (The api:// form of the blocked
        # first-party GUIDs is already handled by the always-on floor above.)
        if not _OBO_API_AUTHORITY_RE.match(authority):
            return True
        # Any well-formed api:// authority (a named App ID URI or the auto-generated
        # api://<guid> for a custom app) is accepted.
        return False
    # No scheme: accept only a bare non-GUID client-id token; reject a BARE GUID
    # (directly addresses a first-party resource) or any value carrying a scheme
    # marker / disallowed characters.
    if "://" in target or ":" in target:
        return True
    if _GUID_RE.match(target):
        return True
    return not bool(_OBO_BARE_CLIENT_ID_RE.match(target))


def _obo_scope_resource(scope: str) -> str:
    """The resource prefix a requested OBO scope grants against.

    Entra scopes are ``<resource>/<permission>`` (e.g.
    ``api://outlook-mcp-server/.default`` or ``https://graph.microsoft.com/User.Read``).
    The resource is the scope minus the final permission segment; a scope with no
    permission segment is its own resource. The ``scheme://`` prefix is split off
    first so the authority's ``//`` is never mistaken for the permission separator
    (otherwise ``api://app`` would wrongly yield ``api:/``).
    """
    s = scope.strip().rstrip("/")
    if "://" in s:
        scheme, _, rest = s.partition("://")
        # Strip only a trailing permission segment AFTER the authority.
        if "/" in rest:
            return f"{scheme}://{rest.rsplit('/', 1)[0]}"
        return s  # bare scheme://authority (no permission segment)
    # No scheme (bare client-id / GUID): strip a trailing permission if present.
    if "/" in s:
        return s.rsplit("/", 1)[0]
    return s


def _obo_scope_mismatches_target(scope: str, target_audience: str) -> bool:
    """True if a requested OBO scope grants against a resource other than the target.

    The exchange engine (auth_server.egress_obo) sends ``egress_oauth.scopes``
    verbatim when present, IGNORING target_audience. So an unvalidated scope like
    ``https://graph.microsoft.com/.default`` would exchange for a Graph token even
    when target_audience is a benign internal app. We bind every scope to the
    validated target: the scope's resource prefix must equal target_audience (both
    normalized). This is what keeps the target check meaningful.
    """
    resource = _obo_scope_resource(scope).lower().rstrip("/")
    target = target_audience.strip().lower().rstrip("/")
    return bool(resource) and resource != target


class CustomHeader(BaseModel):
    """A single user-defined HTTP header attached to an MCP server."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="HTTP header name. Must be a valid RFC 7230 token.",
    )
    value: str = Field(
        ...,
        max_length=4096,
        description="HTTP header value. Stored encrypted; never returned in list responses.",
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not _RFC7230_TOKEN_RE.fullmatch(v):
            raise ValueError("Header name contains invalid characters (must be RFC 7230 token)")
        return v

    @field_validator("value")
    @classmethod
    def _validate_value(cls, v: str) -> str:
        if "\r" in v or "\n" in v:
            raise ValueError("Header value cannot contain CR or LF")
        return v


class CustomHeaderEncrypted(BaseModel):
    """Stored form: header name + Fernet-encrypted value."""

    name: str
    value_encrypted: str


class EgressOAuthConfig(BaseModel):
    """Per-server egress OAuth config for the egress credential paths.

    None on ServerInfo == no egress auth. Two modes share this container:

    - ``oauth_user`` (3LO vault): the operator supplies
      ``provider``/``client_id``/``client_secret``/``scopes`` at registration;
      ``custom_*`` fields apply only when ``provider == 'custom'``.
    - ``obo_exchange`` (same-IdP OBO): the gateway re-audiences the user's ingress
      token to the internal MCP server's app via the gateway's OWN IdP
      credentials. No per-server provider/client_id/secret is needed; only
      ``target_audience`` (and audience-scoped ``scopes``) are required.

    ``provider`` is therefore optional at the model level and required only for
    ``oauth_user`` (enforced by ServerInfo's egress validator, since the mode
    lives on ServerInfo, not here).
    """

    provider: str | None = Field(
        default=None,
        description="Provider key ('github', 'google', 'custom', ...). Required for "
        "oauth_user; unused for obo_exchange (same-IdP exchange uses the gateway's own IdP).",
    )
    client_id: str = Field(default="", description="Operator-supplied OAuth app client_id.")
    client_secret_encrypted: str | None = Field(
        default=None,
        description="Fernet-encrypted client_secret. Never returned in API responses.",
    )
    scopes: list[str] = Field(default_factory=list)
    # OBO exchange (obo_exchange mode only): the internal MCP server's audience.
    target_audience: str | None = Field(
        default=None,
        description="obo_exchange only: the internal MCP server's App ID URI (Entra, "
        "e.g. 'api://outlook-mcp-server') or client id (Keycloak). The 'aud' the gateway "
        "requests in the OBO exchange. IdP-shaped; the exchange engine formats the request "
        "per IdP.",
    )
    # Custom-OIDC overrides (only when provider == 'custom')
    custom_authorize_url: str | None = None
    custom_token_url: str | None = None
    custom_scope_separator: str | None = None
    custom_token_auth_style: str | None = None
    updated_at: str | None = None


class ServerVersion(BaseModel):
    """Represents a single version of an MCP server.

    Used for multi-version server support where different versions
    can run simultaneously behind a single endpoint.
    """

    version: str = Field(..., description="Version identifier (e.g., 'v2.0.0', 'v1.5.0')")
    proxy_pass_url: str = Field(..., description="Backend URL for this version")
    status: str = Field(default="stable", description="Version status: stable, deprecated, beta")
    is_default: bool = Field(
        default=False, description="Whether this is the default (latest) version"
    )
    released: str | None = Field(default=None, description="Release date (ISO format)")
    sunset_date: str | None = Field(
        default=None, description="Deprecation sunset date (ISO format)"
    )
    description: str | None = Field(
        default=None, description="Version-specific description (if different from main)"
    )


def _validate_deployment_invariants(obj: Any) -> None:
    """Enforce remote-vs-local field invariants on a server-like object.

    Used by ServerInfo's @model_validator. The object must expose:
    deployment, local_runtime, proxy_pass_url, mcp_endpoint, sse_endpoint,
    auth_scheme. `versions` (multi-version routing) is checked via getattr so
    callers that don't have such a field don't need to define one.

    For deployment='local' the helper also forces transport='stdio' and
    supported_transports=['stdio'] on the object.
    """
    if obj.deployment == DeploymentType.LOCAL:
        if obj.local_runtime is None:
            raise ValueError("deployment='local' requires local_runtime")
        if obj.proxy_pass_url is not None:
            raise ValueError("deployment='local' must not set proxy_pass_url")
        if obj.mcp_endpoint is not None:
            raise ValueError("deployment='local' must not set mcp_endpoint")
        if obj.sse_endpoint is not None:
            raise ValueError("deployment='local' must not set sse_endpoint")
        if obj.auth_scheme not in ("none", ""):
            raise ValueError(
                "deployment='local' must use auth_scheme='none' "
                "(local servers handle auth via env vars on the user's machine)"
            )
        if getattr(obj, "versions", None) is not None:
            raise ValueError("deployment='local' does not support multi-version routing")
        obj.transport = TransportType.STDIO
        obj.supported_transports = [TransportType.STDIO]
    else:
        # deployment == "remote"
        if obj.local_runtime is not None:
            raise ValueError("deployment='remote' must not set local_runtime")
        if not obj.proxy_pass_url:
            raise ValueError("deployment='remote' requires proxy_pass_url")


class LocalRuntime(BaseModel):
    """How to launch a local (stdio) MCP server on a developer's machine.

    The registry stores the recipe; it does NOT run the server. Health checks
    do not apply. The recipe is emitted as IDE config (Claude Code, Cursor, etc.)
    via the Connect modal.
    """

    type: Literal["npx", "docker", "uvx", "command"] = Field(
        ...,
        description=(
            "Launcher type. npx/uvx: package name. docker: image ref. "
            "command: raw executable path (admin-only, highest trust)."
        ),
    )
    package: str = Field(
        ...,
        min_length=1,
        description="Package name, image reference, or command path depending on `type`.",
    )
    args: list[str] = Field(
        default_factory=list,
        description="Argv-style arguments passed to the launcher (no shell interpolation).",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Environment variables. Values may be literal or ${VAR} placeholders. "
            "Literal-looking secrets are rejected at registration time."
        ),
    )
    required_env: list[str] = Field(
        default_factory=list,
        description=(
            "Env var names the user MUST provide at connect time. MUST NOT overlap with `env` keys."
        ),
    )

    # docker-only
    image_digest: str | None = Field(
        default=None,
        description="Pinned image digest, e.g. 'sha256:abc...'. Encouraged for supply-chain hardening.",
    )
    platforms: list[str] | None = Field(
        default=None,
        description="Supported platforms, e.g. ['linux/amd64', 'linux/arm64'].",
    )

    # npx/uvx-only
    version: str | None = Field(
        default=None,
        description="Package version pin, e.g. '1.2.0'. Encouraged.",
    )

    @model_validator(mode="after")
    def _validate_runtime_consistency(self) -> "LocalRuntime":
        """Validate runtime fields and required_env disjointness from env."""
        # required_env keys must not overlap with env keys (kiro round-1 feedback)
        overlap = set(self.required_env) & set(self.env.keys())
        if overlap:
            raise ValueError(f"required_env keys must not also appear in env: {sorted(overlap)}")

        # platforms only meaningful for docker
        if self.platforms is not None and self.type != LocalRuntimeType.DOCKER:
            raise ValueError("platforms is only valid for docker runtime")

        # image_digest only meaningful for docker
        if self.image_digest is not None and self.type != LocalRuntimeType.DOCKER:
            raise ValueError("image_digest is only valid for docker runtime")

        # image_digest format check (only when provided): require the full
        # 'sha256:<64 hex>' shape so malformed digests fail at registration
        # rather than silently propagating to clients.
        if self.image_digest is not None and not _IMAGE_DIGEST_RE.fullmatch(self.image_digest):
            raise ValueError(
                f"image_digest must match 'sha256:<64 hex chars>', got: {self.image_digest!r}"
            )

        return self


class ServerInfo(BaseModel):
    """Server information model."""

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        max_length=512,
        description=(
            "Unique identifier for this server. Any non-empty string "
            "(UUID, ARN, URN, ...). Auto-generated UUID if not supplied."
        ),
    )
    server_name: str
    description: str = ""
    path: str
    proxy_pass_url: str | None = None
    tags: list[str] = Field(default_factory=list)
    num_tools: int = 0
    license: str = "N/A"
    tool_list: list[dict[str, Any]] = Field(default_factory=list)
    is_enabled: bool = False
    transport: str | None = Field(
        default="auto", description="Preferred transport: sse, streamable-http, or auto"
    )
    supported_transports: list[str] = Field(
        default_factory=lambda: ["streamable-http"], description="List of supported transports"
    )
    mcp_endpoint: str | None = Field(
        default=None,
        description="Full URL for the MCP streamable-http endpoint. If set, used directly for health checks and client connections instead of appending /mcp to proxy_pass_url. Example: 'https://server.com/custom-path'",
    )
    sse_endpoint: str | None = Field(
        default=None,
        description="Full URL for the SSE endpoint. If set, used directly for health checks and client connections instead of appending /sse to proxy_pass_url. Example: 'https://server.com/events'",
    )
    oauth_client_id: str | None = Field(
        default=None,
        description=(
            "Pre-registered public OAuth client_id advertised in this server's "
            "Connect config so IDEs (Cursor, Claude Code, Codex) run the OAuth/PKCE "
            "login flow instead of embedding a static gateway token. Overrides the "
            "registry-wide IDE_OAUTH_CLIENT_ID default. Use when anonymous Dynamic "
            "Client Registration is disabled and a fixed public client is registered."
        ),
    )
    append_mcp_path: bool | None = Field(
        default=None,
        description=(
            "Override whether the gateway Connect URL gets a trailing '/mcp' "
            "transport segment. None (default) auto-detects from proxy_pass_url. "
            "Set false for root-endpoint servers (e.g. AWS Knowledge) that serve "
            "MCP at the server path itself; set true to force the suffix."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom metadata for organization, compliance, or integration purposes",
    )
    # Version routing fields
    version: str | None = Field(
        default=None,
        description="Current version identifier (e.g., 'v1.0.0'). None for legacy single-version servers.",
    )
    versions: list[ServerVersion] | None = Field(
        default=None,
        description="List of available versions. None = single-version server (backward compatible).",
    )
    default_version: str | None = Field(
        default=None, description="Default version identifier for routing (e.g., 'v2.0.0')"
    )
    is_active: bool = Field(
        default=True,
        description="Whether this is the active version. False for inactive versions in multi-version setup.",
    )
    version_group: str | None = Field(
        default=None, description="Groups related versions together (derived from path)"
    )
    other_version_ids: list[str] = Field(
        default_factory=list, description="IDs of other versions in this group (for quick lookup)"
    )

    def get_default_proxy_url(self) -> str:
        """Get the proxy URL for the default version."""
        if not self.versions:
            return self.proxy_pass_url or ""

        for v in self.versions:
            if v.is_default or v.version == self.default_version:
                return v.proxy_pass_url

        # Fallback to first version or original proxy_pass_url
        if self.versions:
            return self.versions[0].proxy_pass_url
        return self.proxy_pass_url or ""

    def has_multiple_versions(self) -> bool:
        """Check if server has multiple versions configured."""
        return self.versions is not None and len(self.versions) > 1

    # Federation and access control fields
    visibility: str = Field(
        default="public",
        description="Federation visibility: public (shared with all peers), group-restricted (shared with allowed_groups only), or private (never shared). 'internal' is accepted as an alias for 'private'.",
    )
    allowed_groups: list[str] = Field(
        default_factory=list, description="Groups with access when visibility is group-restricted"
    )
    sync_metadata: dict[str, Any] | None = Field(
        default=None, description="Metadata for items synced from peer registries"
    )

    # ANS Integration
    ans_metadata: dict[str, Any] | None = Field(
        default=None,
        alias="ansMetadata",
        description="ANS (Agent Name Service) verification metadata",
    )

    # Backend authentication (replaces legacy auth_type)
    auth_scheme: str = Field(
        default="none",
        description="Authentication scheme for backend server: none, bearer, api_key",
    )
    auth_credential_encrypted: str | None = Field(
        default=None,
        description="Encrypted auth credential (Fernet). Never returned in API responses.",
    )
    auth_header_name: str | None = Field(
        default=None,
        description="Custom header name. Default: 'Authorization' for bearer, 'X-API-Key' for api_key.",
    )
    credential_updated_at: str | None = Field(
        default=None, description="ISO timestamp of last credential update."
    )

    # Custom HTTP headers (encrypted values, names public)
    custom_header_names: list[str] = Field(
        default_factory=list,
        description="Names of custom HTTP headers defined for this server.",
    )
    custom_headers_encrypted: list[CustomHeaderEncrypted] | None = Field(
        default=None,
        description="List of {name, value_encrypted} pairs. Never serialized to API consumers.",
    )
    custom_headers_updated_at: str | None = Field(
        default=None,
        description="ISO timestamp of last custom-headers update.",
    )

    # Per-user egress credential vault (third-party OBO). Default 'none' keeps
    # today's behavior; the registration write path is not yet implemented.
    egress_auth_mode: str = Field(
        default="none",
        description="Egress auth to the upstream: 'none', 'oauth_user' (3LO vault), "
        "or 'obo_exchange' (same-IdP OBO).",
    )
    egress_oauth: EgressOAuthConfig | None = Field(
        default=None,
        description="Egress OAuth config. Required when egress_auth_mode is 'oauth_user' "
        "(provider/client_id/secret) or 'obo_exchange' (target_audience).",
    )

    # Lifecycle and federation metadata fields
    status: LifecycleStatus = Field(
        default=LifecycleStatus.ACTIVE,
        description="Lifecycle status",
    )
    provider: AgentProvider | None = Field(
        default=None,
        description="Provider organization and URL",
    )
    source_created_at: datetime | None = Field(
        default=None,
        description="Original creation timestamp from source system",
    )
    source_updated_at: datetime | None = Field(
        default=None,
        description="Last update timestamp from source system",
    )
    external_tags: list[str] = Field(
        default_factory=list,
        description="Tags from external/source system (separate from local tags)",
    )
    deployment: Literal["remote", "local"] = Field(
        default=cast(Literal["remote", "local"], DeploymentType.REMOTE),
        description=(
            "Deployment model: 'remote' (HTTP-reachable, registry proxies) or "
            "'local' (stdio, runs on developer's machine via launch recipe)."
        ),
    )
    local_runtime: LocalRuntime | None = Field(
        default=None,
        description="Launch recipe. Required when deployment='local', forbidden otherwise.",
    )
    registered_by: str | None = Field(
        default=None,
        description=(
            "Username of the user who registered this server. Audit trail; "
            "load-bearing for local servers (executable recipe approval). "
            "Records the ORIGINAL registrant only — edits do not update this "
            "field. The general audit log captures who last touched the entry."
        ),
    )

    @field_validator("visibility")
    @classmethod
    def _validate_visibility(
        cls,
        v: str,
    ) -> str:
        """Validate and normalize visibility value.

        Accepts "internal" as alias for "private" and "group" as alias
        for "group-restricted" for backward compatibility.
        """
        from registry.utils.visibility import validate_visibility

        return validate_visibility(v)

    @field_validator("path")
    @classmethod
    def _validate_path(
        cls,
        v: str,
    ) -> str:
        """Reject server paths carrying nginx/URL-unsafe characters.

        The path is interpolated into generated nginx config (location and `set`
        directives, regex maps) and into per-server discovery URLs. A value with
        a double-quote, semicolon, brace, whitespace, or newline could break out
        of a directive/string context and inject nginx config. Restrict to a safe
        slug (alphanumerics and ``. _ - /``) so the class of injection is
        impossible at the source, not only escaped at each render site
        (defense-in-depth with NginxConfigService._sanitize_for_nginx_set).
        """
        if not v or not _SERVER_PATH_RE.fullmatch(v):
            raise ValueError(
                f"invalid server path {v!r}: must be non-empty and contain only "
                "letters, digits, '.', '_', '-', and '/'"
            )
        return v

    @model_validator(mode="after")
    def _populate_provider_default(self) -> "ServerInfo":
        """Populate default provider from config if not set."""
        if self.provider is None:
            from registry.core.config import settings

            self.provider = AgentProvider(
                organization=settings.registry_organization_name,
                url=settings.registry_url,
            )
        return self

    @model_validator(mode="after")
    def _validate_deployment_consistency(self) -> "ServerInfo":
        """Enforce remote/local field invariants. See _validate_deployment_invariants."""
        _validate_deployment_invariants(self)
        return self

    @model_validator(mode="after")
    def _validate_egress_auth(self) -> "ServerInfo":
        """Enforce per-mode egress config invariants.

        - oauth_user: requires egress_oauth with a provider (3LO needs a provider).
        - obo_exchange: requires egress_oauth.target_audience, and that audience
          MUST (a) differ from the gateway's own IdP client id / app ID URI (Entra
          rejects a same-app OBO -- it is a passthrough, not an exchange), and
          (b) not be a shared first-party IdP resource (Microsoft Graph / ARM /
          Key Vault / ...): an obo target must be an internal MCP server's own
          app, never a broad first-party API whose delegated token would be
          exfiltrated to the server's upstream. Both are rejected at registration
          rather than at the first live request.
        """
        mode = self.egress_auth_mode
        if mode not in ("none", "oauth_user", "obo_exchange"):
            raise ValueError(
                f"invalid egress_auth_mode {mode!r}; expected 'none', 'oauth_user', "
                "or 'obo_exchange'"
            )
        if mode == "none":
            return self
        if self.egress_oauth is None:
            raise ValueError(f"egress_auth_mode={mode!r} requires egress_oauth config")
        if mode == "oauth_user":
            if not self.egress_oauth.provider:
                raise ValueError("egress_auth_mode='oauth_user' requires egress_oauth.provider")
            return self
        # mode == "obo_exchange"
        target = (self.egress_oauth.target_audience or "").strip()
        if not target:
            raise ValueError(
                "egress_auth_mode='obo_exchange' requires egress_oauth.target_audience"
            )
        # Reject a malformed scheme audience (e.g. 'api://', 'api:/', 'api:'): a
        # value carrying a ':' scheme separator must be a well-formed
        # 'scheme://<non-empty-authority>'. A degenerate scheme would collapse to a
        # bare scheme in scope-resource extraction, letting a scope for another
        # resource appear to "match" it.
        if ":" in target:
            scheme, sep, rest = target.partition("://")
            if not sep or not rest.strip() or not scheme.strip():
                raise ValueError(
                    f"egress_oauth.target_audience {target!r} is malformed: a scheme "
                    "audience must be 'scheme://<authority>' with a non-empty authority "
                    "(e.g. 'api://<app-id>')"
                )
        if _is_gateway_own_audience(target):
            raise ValueError(
                "egress_oauth.target_audience must differ from the gateway's own IdP "
                "client id / app ID URI; same-app OBO is not a valid exchange"
            )
        if _is_disallowed_obo_audience(target):
            raise ValueError(
                f"egress_oauth.target_audience {target!r} is not an allowed obo_exchange "
                "target. It must be an internal MCP server's own IdP audience: an "
                "'api://...' Entra App ID URI (including the 'api://<app-guid>' form) or "
                "a bare non-GUID client-id. It must NEVER be an 'https://' host URL or a "
                "bare GUID, which is how a shared first-party API (Microsoft Graph, ARM, "
                "Key Vault) is directly addressable -- a delegated token for such a "
                "resource would be exfiltrated to the server's upstream (confused deputy). "
                "Pin a bare-GUID audience explicitly via EGRESS_OBO_ALLOWED_AUDIENCES."
            )
        # Bind scopes to the target. The exchange engine sends egress_oauth.scopes
        # verbatim (ignoring target_audience) when present, so an unvalidated scope
        # for a different resource (e.g. https://graph.microsoft.com/.default) would
        # defeat the target check entirely. Require every scope to grant against the
        # validated target.
        for scope in self.egress_oauth.scopes or []:
            if not scope or not scope.strip():
                raise ValueError("egress_oauth.scope entries must be non-empty")
            if _obo_scope_mismatches_target(scope, target):
                raise ValueError(
                    f"egress_oauth.scope {scope!r} grants against a resource other than "
                    f"target_audience {target!r}. obo_exchange scopes must be audience-"
                    "scoped to the target (e.g. '<target_audience>/.default'); a scope "
                    "for a different resource would exchange the user's token for THAT "
                    "resource (confused deputy)."
                )
        return self


class ToolDescription(BaseModel):
    """Parsed tool description sections."""

    main: str = "No description available."
    args: str | None = None
    returns: str | None = None
    raises: str | None = None


class ToolInfo(BaseModel):
    """Tool information model."""

    name: str
    parsed_description: ToolDescription
    tool_schema: dict[str, Any] = Field(default_factory=dict, alias="schema")
    server_path: str | None = None
    server_name: str | None = None

    class Config:
        populate_by_name = True


class HealthStatus(BaseModel):
    """Health check status model."""

    status: str
    last_checked_iso: str | None = None
    num_tools: int = 0


class SessionData(BaseModel):
    """Session data model."""

    username: str
    auth_method: str = "oauth2"
    provider: str = "local"


class ServiceRegistrationRequest(BaseModel):
    """Service registration request model."""

    name: str = Field(..., min_length=1)
    description: str = ""
    path: str = Field(..., min_length=1)
    proxy_pass_url: str = Field(..., min_length=1)
    tags: str = ""
    num_tools: int = Field(0, ge=0)
    license: str = "N/A"
    transport: str | None = Field(
        default="auto", description="Preferred transport: sse, streamable-http, or auto"
    )
    supported_transports: str = Field(
        default="streamable-http", description="Comma-separated list of supported transports"
    )
    mcp_endpoint: str | None = Field(
        default=None,
        description="Full URL for the MCP streamable-http endpoint. If set, used directly for health checks and client connections instead of appending /mcp to proxy_pass_url. Example: 'https://server.com/custom-path'",
    )
    sse_endpoint: str | None = Field(
        default=None,
        description="Full URL for the SSE endpoint. If set, used directly for health checks and client connections instead of appending /sse to proxy_pass_url. Example: 'https://server.com/events'",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom metadata for organization, compliance, or integration purposes",
    )
    visibility: str = Field(
        default="public",
        description="Federation visibility: public (shared with all peers), group-restricted (shared with allowed_groups only), or private (never shared). 'internal' is accepted as an alias for 'private'.",
    )
    allowed_groups: list[str] = Field(
        default_factory=list, description="Groups with access when visibility is group-restricted"
    )
    auth_scheme: str = Field(
        default="none", description="Authentication scheme: none, bearer, api_key"
    )
    auth_credential: str | None = Field(
        default=None,
        description="Plaintext credential (encrypted before storage, never stored as-is)",
    )
    auth_header_name: str | None = Field(
        default=None, description="Custom header name for API key auth. Default: X-API-Key"
    )
    status: LifecycleStatus = Field(
        default=LifecycleStatus.ACTIVE,
        description="Lifecycle status: active, deprecated, draft, or beta",
    )


class AuthCredentialUpdateRequest(BaseModel):
    """Request model for updating server auth credentials via PATCH."""

    auth_scheme: str = Field(..., description="Authentication scheme: none, bearer, api_key")
    auth_credential: str | None = Field(
        default=None, description="New credential (required if auth_scheme is not 'none')"
    )
    auth_header_name: str | None = Field(
        default=None, description="Custom header name. Default: X-API-Key for api_key"
    )


class OAuth2Provider(BaseModel):
    """OAuth2 provider information."""

    name: str
    display_name: str
    icon: str | None = None
