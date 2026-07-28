import asyncio
import hashlib
import ipaddress
import json
import logging
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from registry.constants import REGISTRY_CONSTANTS, DeploymentType, HealthStatus

from .config import settings
from .metrics import NGINX_CONFIG_WRITES, NGINX_UPDATES_SKIPPED

logger = logging.getLogger(__name__)


# Default mode applied to a fresh nginx config file when no destination
# exists yet. Subsequent writes preserve whatever mode the destination
# currently has so an operator's chmod isn't silently reverted.
DEFAULT_NGINX_CONFIG_MODE: int = 0o644

# Route prefix under which enabled A2A agents are reverse-proxied through the
# gateway. An agent registered at path "/flight-booking-agent" is
# reachable at "{ROOT_PATH}/agent/flight-booking-agent/".
AGENT_ROUTE_PREFIX: str = "/agent"

# Agent path and backend url come from registry data and are interpolated into
# nginx directive positions, so they must be validated to prevent config
# injection (e.g. "}", ";", newlines breaking out of the location block).
_NGINX_AGENT_PATH_SAFE = re.compile(r"^[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$")
_NGINX_AGENT_URL_SAFE = re.compile(r"^https?://[A-Za-z0-9.\-]+(?::\d+)?(?:/[A-Za-z0-9._~\-/]*)?$")


# Headroom added on top of the auth-server mcp-proxy hop's own upstream timeout
# (MCP_PROXY_TIMEOUT) when deriving nginx's proxy_read_timeout for the
# /mcp-proxy/ location blocks. nginx must outlive the inner hop so a
# slow-but-progressing upstream is severed by auth-server (clean 504 "Upstream
# MCP server timed out") rather than by nginx. The default upstream timeout
# (30s) yields 60s here, matching nginx's historical implicit default for these
# blocks, so behavior is unchanged unless MCP_PROXY_TIMEOUT is raised.
# Credit: derivation approach contributed by @go-faustino (PR #1321).
MCP_PROXY_NGINX_READ_TIMEOUT_BUFFER_SECONDS: int = 30


# Minimum sane prefix length for a TRUSTED_REAL_IP_CIDRS entry. A prefix shorter
# than this (e.g. 0.0.0.0/1, 10.0.0.0/4) trusts an implausibly large peer range
# for a proxy subnet and edges toward the catch-all footgun, so we warn (but still
# honour it — unlike a /0, which is rejected outright). IPv4-scale; IPv6 prefixes
# are compared on their IPv4-equivalent host-bit width so a normal /48–/64 proxy
# subnet does not trip the warning.
MIN_TRUSTED_REAL_IP_PREFIXLEN_V4: int = 8
MIN_TRUSTED_REAL_IP_PREFIXLEN_V6: int = 32


def _resolve_mcp_proxy_read_timeout_seconds() -> int:
    """Resolve nginx's proxy_read_timeout (seconds) for MCP location blocks.

    Derived from the auth-server upstream timeout (``settings.mcp_proxy_timeout``
    / ``MCP_PROXY_TIMEOUT``) plus a fixed headroom buffer, so a single knob
    raises the whole proxy chain and the inner hop always times out first.
    Invalid values fall back to the 30s default (which becomes 60s with the
    buffer added).

    Returns:
        nginx proxy_read_timeout in whole seconds.
    """
    default_upstream = 30.0
    minimum_upstream = 1.0
    upstream = default_upstream
    try:
        value = getattr(settings, "mcp_proxy_timeout", None)
        if value is not None:
            upstream = max(float(value), minimum_upstream)
    except (TypeError, ValueError) as e:
        logger.debug(f"Invalid mcp_proxy_timeout, using default for nginx read timeout: {e}")
        upstream = default_upstream
    return int(math.ceil(upstream)) + MCP_PROXY_NGINX_READ_TIMEOUT_BUFFER_SECONDS


def _render_real_ip_config() -> str:
    """Render nginx realip directives from the ``TRUSTED_REAL_IP_CIDRS`` env var.

    ``TRUSTED_REAL_IP_CIDRS`` is a comma-separated list of CIDRs (or bare IPs)
    identifying the trusted proxy hop(s) directly in front of nginx — typically
    the VPC/subnet CIDR an ALB's ENI lives in. When set, nginx recovers the real
    client IP from ``X-Forwarded-For`` for connections whose immediate peer falls
    in one of these ranges, so the audited client IP is the end user rather than
    the load balancer's internal address.

    Fails closed on bad input: each entry is validated as a well-formed network,
    and any malformed entry is dropped with a warning rather than emitting an
    invalid directive that would break nginx config generation. When the variable
    is unset or yields no valid entries, an EMPTY string is returned — no realip
    directives are emitted, which is the correct behaviour for edge deployments
    (compose / single host) where nginx's peer already IS the client.

    ``real_ip_recursive`` walks the forwarded chain right-to-left skipping the
    trusted CIDRs, stopping at the first untrusted address (the real client). It
    is emitted as ``on`` ONLY when more than one trusted CIDR is configured (a
    stacked-proxy topology, e.g. CloudFront in front of an ALB). For the common
    single-hop case (one ALB) recursion is left off: nginx takes the single
    right-most entry, which is what the one trusted proxy appended, and a spoofed
    left-most entry can never win. Recursion + an over-broad trusted range would
    let a client whose own source falls inside that range inject a left-most
    entry, so we don't enable it unless the topology actually needs it.

    Catch-all ranges (``0.0.0.0/0`` / ``::/0``) are rejected: they would make
    nginx trust EVERY peer and take the spoofable left-most XFF entry — a
    fail-open misconfiguration — so they are dropped with a warning like any other
    invalid entry. Narrow the trust to the load balancer's specific subnet(s).

    Returns:
        The nginx directive block (``set_real_ip_from`` lines + ``real_ip_header``
        + optional ``real_ip_recursive on``), or an empty string when no valid
        CIDRs are configured.
    """
    raw = os.environ.get("TRUSTED_REAL_IP_CIDRS", "").strip()
    if not raw:
        return ""

    valid_cidrs: list[str] = []
    for entry in raw.split(","):
        candidate = entry.strip()
        if not candidate:
            continue
        try:
            # strict=False so a host address (e.g. 10.1.3.39) is accepted as a
            # /32 rather than rejected for having host bits set.
            network = ipaddress.ip_network(candidate, strict=False)
        except ValueError:
            logger.warning("Ignoring malformed entry in TRUSTED_REAL_IP_CIDRS: %r", candidate)
            continue
        # Reject a catch-all range: trusting every peer defeats the guard and
        # makes the spoofable left-most XFF entry win (fail-open). prefixlen == 0
        # is 0.0.0.0/0 or ::/0.
        if network.prefixlen == 0:
            logger.warning(
                "Refusing catch-all range %r in TRUSTED_REAL_IP_CIDRS "
                "(would trust every peer); narrow it to the proxy's subnet",
                candidate,
            )
            continue
        # Warn (but honour) an implausibly broad, non-catch-all range. A proxy
        # subnet is realistically a /16-/28 (v4) or /48-/64 (v6); anything much
        # broader trusts far more peers than a real LB tier occupies and edges
        # toward the same spoofing risk as a catch-all.
        floor = (
            MIN_TRUSTED_REAL_IP_PREFIXLEN_V6
            if network.version == 6
            else MIN_TRUSTED_REAL_IP_PREFIXLEN_V4
        )
        if network.prefixlen < floor:
            logger.warning(
                "TRUSTED_REAL_IP_CIDRS entry %r is very broad (/%d); trusting this "
                "many peers is risky — narrow it to the proxy's actual subnet",
                candidate,
                network.prefixlen,
            )
        valid_cidrs.append(str(network))

    if not valid_cidrs:
        logger.warning(
            "TRUSTED_REAL_IP_CIDRS was set but contained no valid CIDRs; "
            "not emitting realip directives"
        )
        return ""

    logger.info("Configuring nginx realip trust for CIDRs: %s", ", ".join(valid_cidrs))
    lines = [f"set_real_ip_from {cidr};" for cidr in valid_cidrs]
    lines.append("real_ip_header X-Forwarded-For;")
    # Only recurse for stacked proxies (>1 trusted hop). A single trusted CIDR
    # needs no recursion — the right-most entry is the one that proxy appended.
    if len(valid_cidrs) > 1:
        lines.append("real_ip_recursive on;")
    return "\n".join(lines)


def _atomic_write_text(
    path: Path,
    content: str,
) -> None:
    """Write content to path atomically (issue #1044).

    Writes to a temporary file in the same directory as ``path`` and uses
    ``os.replace()`` to swap it into place. ``os.replace()`` is atomic on POSIX
    when source and destination are on the same filesystem, so any reader
    (including ``nginx -t``) sees either the old config or the new one - never
    a truncated mid-write file.

    The temp file's mode is set to match the destination's existing mode, or
    ``DEFAULT_NGINX_CONFIG_MODE`` when the destination does not yet exist.
    Without this, ``tempfile.NamedTemporaryFile`` defaults to ``0o600`` and
    silently tightens permissions across atomic writes.

    On any failure the temp file is removed and the destination is left
    unchanged. ``NGINX_CONFIG_WRITES`` is incremented with
    ``status="success"`` or ``status="failure"`` accordingly.

    Args:
        path: Destination path.
        content: Text content to write.

    Raises:
        OSError: If the temp file cannot be created, written, or replaced.
    """
    dest_dir = path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    if path.exists():
        target_mode = path.stat().st_mode & 0o777
    else:
        target_mode = DEFAULT_NGINX_CONFIG_MODE

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        dir=dest_dir,
        prefix=f".{path.name}.tmp.",
        delete=False,
        encoding="utf-8",
    )
    tmp_path = Path(tmp.name)
    try:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.chmod(tmp_path, target_mode)
        os.replace(tmp_path, path)
        NGINX_CONFIG_WRITES.labels(status="success").inc()
    except Exception:
        NGINX_CONFIG_WRITES.labels(status="failure").inc()
        try:
            tmp.close()
        except Exception:  # nosec B110 - best-effort cleanup of temp file on write failure
            pass
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _cleanup_stale_temp_files(config_path: Path) -> None:
    """Remove leftover ``.{config_name}.tmp.*`` files from a crashed write.

    On a clean write, the temp file is renamed away by ``os.replace()``. If the
    process was killed mid-write (SIGKILL, OOM kill, host reboot), a temp file
    matching ``.{config_name}.tmp.*`` can remain. Container restarts on
    ECS/EKS typically clean this up naturally, but long-lived hosts (local
    dev, EC2 without container ephemerality) need explicit cleanup.

    Best-effort: logs warnings but does not raise on failure.
    """
    dest_dir = config_path.parent
    pattern = f".{config_path.name}.tmp.*"
    try:
        leftovers = list(dest_dir.glob(pattern))
    except OSError as e:
        logger.warning(f"Could not scan {dest_dir} for stale temp files: {e}")
        return

    for stale in leftovers:
        try:
            stale.unlink()
            logger.info(f"Removed stale nginx config temp file: {stale}")
        except OSError as e:
            logger.warning(f"Failed to remove stale temp file {stale}: {e}")


# Suffix for the last-known-good copy kept beside the live config while a
# candidate is being validated. On a failed validation the live config is
# restored from this copy so the broken candidate never survives on disk.
_LAST_GOOD_SUFFIX: str = ".last-good"


# Sentinel returned by the config test when the nginx binary is absent. There
# is no nginx to protect from a bad cold start on such a host (e.g. registry-
# only mode or local dev), so the caller promotes the candidate without a test
# rather than rejecting a legitimate render it cannot validate.
_NGINX_TEST_NO_BINARY: str = "__nginx_binary_missing__"


def _run_nginx_config_test() -> tuple[bool, str]:
    """Run ``nginx -t`` against the live config tree.

    ``nginx -t`` parses the full configuration tree exactly as a cold start
    (container restart) would - main ``nginx.conf`` plus every ``include``d
    fragment and the Lua modules they reference. Validating the candidate in
    place (see :func:`_write_and_validate_config`) is therefore the only
    faithful predictor of whether a subsequent cold start will boot.

    Returns:
        A ``(passed, message)`` tuple. ``passed`` is False on a non-zero exit,
        a timeout, or any other error (fail closed); ``message`` carries
        nginx's stderr or the failure reason. When the nginx binary is not
        installed, ``message`` is the ``_NGINX_TEST_NO_BINARY`` sentinel so the
        caller can distinguish "no nginx to protect" from "config is invalid".
    """
    import subprocess  # nosec B404

    try:
        result = subprocess.run(
            ["nginx", "-t"],  # nosec B603 B607 - hardcoded command
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return False, _NGINX_TEST_NO_BINARY
    except subprocess.TimeoutExpired:
        return False, "nginx -t timed out"
    except Exception as e:  # pragma: no cover - defensive
        return False, f"nginx -t raised: {e}"

    if result.returncode != 0:
        return False, result.stderr or "nginx -t returned non-zero"
    return True, result.stderr or ""


def _write_and_validate_config(
    path: Path,
    content: str,
) -> None:
    """Promote a rendered nginx config only if ``nginx -t`` accepts it.

    A malformed config (whether from a config-injection attempt that still
    parses partially, an unhealthy-backend edge case, or a template bug) must
    never persist on disk: even when the running nginx keeps serving its
    in-memory config, the broken file on disk takes down the *next* cold start
    (routine on ECS/K8s) and with it the whole gateway.

    This renders the candidate to a temporary file, atomically promotes it into
    ``path`` after backing up the current live config, then runs ``nginx -t``
    against the real config tree. If the test passes, the backup is discarded.
    If it fails, the previous last-known-good config is restored (or, when
    there was no prior config, the rejected file is removed) so the gateway can
    still cold-start on the last-good config. Fails closed: any error leaves the
    last-good config in place and raises.

    Args:
        path: The live nginx config path.
        content: The rendered candidate config content.

    Raises:
        RuntimeError: If ``nginx -t`` rejects the candidate (the live config is
            restored to the last-known-good state before raising).
        OSError: If the temp write / rename cannot be performed.
    """
    import shutil

    path = Path(path)

    # Pre-flight: decide the missing-binary policy BEFORE the candidate ever
    # touches the live path, so a split/sidecar deployment cannot pick up an
    # unvalidated config in the window between write and restore.
    #
    # When the nginx binary is absent we cannot run ``nginx -t``. In a
    # single-container / local-dev deployment that means there is no nginx to
    # cold-start, so promoting the candidate matches the pre-existing behavior.
    # In a split topology (nginx in a separate container/sidecar sharing this
    # config volume) an unvalidated config WOULD poison the sidecar's next cold
    # start, so operators set ``nginx_config_validation_required=True`` to fail
    # closed — and here we refuse before writing, leaving the live config
    # untouched entirely.
    nginx_available = shutil.which("nginx") is not None
    if not nginx_available:
        if settings.nginx_config_validation_required:
            NGINX_CONFIG_WRITES.labels(status="rejected").inc()
            raise RuntimeError(
                "nginx config rejected: nginx binary not found and "
                "nginx_config_validation_required=True (refusing to promote an "
                "unvalidated config to the shared config path)"
            )
        # No nginx on this host to protect: promote as before. In with-gateway
        # mode this is surfaced at WARNING so a misconfigured split deployment
        # is visible without enabling debug logging.
        if settings.nginx_updates_enabled:
            logger.warning(
                "nginx binary not found; promoting config without nginx -t validation. "
                "If nginx runs in a separate container sharing this path, set "
                "NGINX_CONFIG_VALIDATION_REQUIRED=true to fail closed."
            )
        else:
            logger.debug("nginx binary not found; promoting config without nginx -t validation")
        _atomic_write_text(path, content)
        return

    had_previous = path.exists()
    backup_path = path.with_name(path.name + _LAST_GOOD_SUFFIX)

    # Back up the current live config so we can restore it on rejection.
    if had_previous:
        try:
            # copy2 preserves mode/timestamps; a plain copy keeps the live file
            # intact (unlike a rename, which would briefly leave no live file).
            shutil.copy2(path, backup_path)
        except OSError as e:
            logger.error("Could not back up live nginx config before validation: %s", e)
            raise

    # Promote the candidate into the live path (atomic, preserves mode).
    _atomic_write_text(path, content)

    passed, message = _run_nginx_config_test()

    if passed:
        # Candidate accepted: drop the backup.
        if had_previous:
            try:
                backup_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.warning("Could not remove nginx config backup %s: %s", backup_path, e)
        return

    # Candidate rejected: restore the last-known-good config so a cold start
    # still boots, then fail closed.
    logger.error(
        "Rejected nginx config candidate (nginx -t failed); restoring last-good config: %s",
        message.strip(),
    )
    NGINX_CONFIG_WRITES.labels(status="rejected").inc()
    if had_previous:
        try:
            os.replace(backup_path, path)
        except OSError as e:
            logger.error("Failed to restore last-good nginx config from %s: %s", backup_path, e)
            raise RuntimeError(
                f"nginx config validation failed and last-good restore failed: {message.strip()}"
            ) from e
    else:
        # No prior config existed: remove the rejected file so nothing invalid
        # is left for a cold start to load.
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.error("Failed to remove rejected nginx config %s: %s", path, e)
            raise

    raise RuntimeError(f"nginx config rejected by nginx -t: {message.strip()}")


def _ensure_mcp_compliant_schema(input_schema: dict[str, Any]) -> dict[str, Any]:
    """Ensure inputSchema conforms to MCP spec by adding 'type': 'object' if missing.

    The MCP spec requires all tool inputSchema definitions to have "type": "object"
    at the top level. This function ensures backend tool schemas are compliant.

    Args:
        input_schema: The input schema from a backend tool

    Returns:
        MCP-compliant schema with "type": "object" at top level
    """
    if not input_schema:
        return {"type": "object", "properties": {}}

    # If schema already has "type": "object", return as-is
    if input_schema.get("type") == "object":
        return input_schema

    # If schema has "type" but it's not "object", wrap it
    if "type" in input_schema:
        logger.warning(
            f"Tool inputSchema has non-object type '{input_schema.get('type')}'. "
            "Wrapping in object schema to comply with MCP spec."
        )
        return {"type": "object", "properties": {"value": input_schema}}

    # If no "type" field but has "properties", add "type": "object"
    if "properties" in input_schema or "additionalProperties" in input_schema:
        schema_copy = input_schema.copy()
        schema_copy["type"] = "object"
        return schema_copy

    # Default: wrap unknown schema structure
    logger.warning(
        "Tool inputSchema missing 'type' field and has unexpected structure. "
        "Adding 'type': 'object' to comply with MCP spec."
    )
    schema_copy = input_schema.copy()
    schema_copy["type"] = "object"
    return schema_copy


class NginxConfigService:
    """Service for generating Nginx configuration for registered servers."""

    def __init__(self):
        # Contract: every call site that invokes generate_config_async() or
        # reload_nginx() (directly or transitively) MUST acquire this lock for
        # the duration of those calls. The lock prevents:
        #   1. Two writers racing on the nginx config path (lost-update).
        #   2. nginx -t in one writer reading a partial file written by another.
        #   3. Two `nginx -s reload` signals in flight simultaneously.
        # The lock is intentionally coarse-grained because regen + reload is
        # bounded (~150-300ms) and infrequent (tens per minute). See issue
        # #1044 and .scratchpad/issue-1044/lld.md for the full rationale.
        self.reload_lock: asyncio.Lock = asyncio.Lock()

        # Cache for get_additional_server_names (avoids hitting metadata
        # endpoints on every scheduler tick). Invalidated by mark_dirty().
        self._cached_server_names: str | None = None

        # Minimum interval between nginx reload signals. Prevents cascading
        # SIGHUP when many flush_now() calls land in rapid succession (e.g.
        # bulk toggle during stress tests). nginx needs time for worker
        # processes to shut down before accepting another reload.
        self._min_reload_interval_seconds: float = 3.0
        self._last_reload_time: float = 0.0

        # Determine which template to use based on SSL certificate availability
        ssl_cert_path = Path(REGISTRY_CONSTANTS.SSL_CERT_PATH)
        ssl_key_path = Path(REGISTRY_CONSTANTS.SSL_KEY_PATH)

        # Check if SSL certificates exist
        if ssl_cert_path.exists() and ssl_key_path.exists():
            # Use HTTP + HTTPS template
            if Path(REGISTRY_CONSTANTS.NGINX_TEMPLATE_HTTP_AND_HTTPS).exists():
                self.nginx_template_path = Path(REGISTRY_CONSTANTS.NGINX_TEMPLATE_HTTP_AND_HTTPS)
            else:
                # Fallback for local development
                self.nginx_template_path = Path(
                    REGISTRY_CONSTANTS.NGINX_TEMPLATE_HTTP_AND_HTTPS_LOCAL
                )
        else:
            # Use HTTP-only template
            if Path(REGISTRY_CONSTANTS.NGINX_TEMPLATE_HTTP_ONLY).exists():
                self.nginx_template_path = Path(REGISTRY_CONSTANTS.NGINX_TEMPLATE_HTTP_ONLY)
            else:
                # Fallback for local development
                self.nginx_template_path = Path(REGISTRY_CONSTANTS.NGINX_TEMPLATE_HTTP_ONLY_LOCAL)

    async def get_additional_server_names(self) -> str:
        """Fetch or determine additional server names for nginx gateway configuration.

        Supports multi-platform detection:
        1. User-provided GATEWAY_ADDITIONAL_SERVER_NAMES env var
        2. EC2 private IP detection via metadata service
        3. ECS metadata service detection
        4. EKS/Kubernetes pod detection
        5. Generic hostname command fallback
        6. Backward compatibility with EC2_PUBLIC_DNS env var
        """
        import os
        import subprocess  # nosec B404

        # Priority 1: Check GATEWAY_ADDITIONAL_SERVER_NAMES env var (user-provided)
        gateway_names = os.environ.get("GATEWAY_ADDITIONAL_SERVER_NAMES", "")
        if gateway_names:
            logger.info(f"Using GATEWAY_ADDITIONAL_SERVER_NAMES from environment: {gateway_names}")
            return gateway_names.strip()

        # Priority 2: Try EC2 metadata service for private IP
        try:
            async with httpx.AsyncClient() as client:
                # Get session token for IMDSv2
                token_response = await client.put(
                    "http://169.254.169.254/latest/api/token",
                    headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
                    timeout=2.0,
                )

                if token_response.status_code == 200:
                    token = token_response.text

                    # Try to get private IP from EC2 metadata
                    ip_response = await client.get(
                        "http://169.254.169.254/latest/meta-data/local-ipv4",
                        headers={"X-aws-ec2-metadata-token": token},
                        timeout=2.0,
                    )

                    if ip_response.status_code == 200:
                        private_ip = ip_response.text.strip()
                        logger.info(f"Auto-detected EC2 private IP: {private_ip}")
                        return private_ip

        except (httpx.TimeoutException, httpx.ConnectError):
            logger.debug("EC2 metadata service not available - not running on EC2")
        except Exception as e:
            logger.debug(f"EC2 metadata detection failed: {e}")

        # Priority 3: Try ECS metadata service
        ecs_uri = os.environ.get("ECS_CONTAINER_METADATA_URI") or os.environ.get(
            "ECS_CONTAINER_METADATA_URI_V4"
        )
        if ecs_uri:
            try:
                async with httpx.AsyncClient() as client:
                    metadata_response = await client.get(f"{ecs_uri}", timeout=2.0)
                    if metadata_response.status_code == 200:
                        import json

                        metadata = json.loads(metadata_response.text)
                        # Try to extract IP from ECS metadata
                        if "Networks" in metadata and metadata["Networks"]:
                            private_ip = metadata["Networks"][0].get("IPv4Addresses", [None])[0]
                            if private_ip:
                                logger.info(f"Auto-detected ECS container IP: {private_ip}")
                                return private_ip
            except Exception as e:
                logger.debug(f"ECS metadata detection failed: {e}")

        # Priority 4: Try EKS/Kubernetes detection
        pod_ip = os.environ.get("POD_IP")
        if pod_ip:
            logger.info(f"Auto-detected Kubernetes pod IP: {pod_ip}")
            return pod_ip

        # Priority 5: Try generic hostname command (works on most Linux systems)
        try:
            result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2.0)  # nosec B603 B607 - hardcoded command
            if result.returncode == 0:
                ips = result.stdout.strip().split()
                if ips:
                    # Use first IP (usually the private IP on single-interface systems)
                    private_ip = ips[0]
                    logger.info(f"Auto-detected private IP via hostname command: {private_ip}")
                    return private_ip
        except Exception as e:
            logger.debug(f"Generic hostname detection failed: {e}")

        # Priority 6: Backward compatibility with old EC2_PUBLIC_DNS env var
        fallback_dns = os.environ.get("EC2_PUBLIC_DNS", "")
        if fallback_dns:
            logger.info(f"Using EC2_PUBLIC_DNS environment variable (deprecated): {fallback_dns}")
            return fallback_dns

        # No additional server names available
        logger.info(
            "No additional server names available - will use only localhost and mcpgateway.ddns.net"
        )
        return ""

    def generate_config(self, servers: dict[str, dict[str, Any]]) -> bool:
        """Generate Nginx configuration (synchronous version for non-async contexts)."""
        if not settings.nginx_updates_enabled:
            logger.info(
                f"Skipping nginx config generation - "
                f"DEPLOYMENT_MODE={settings.deployment_mode.value}"
            )
            NGINX_UPDATES_SKIPPED.labels(operation="generate_config").inc()
            return True

        try:
            # Check if we're in an async context
            try:
                # If we're already in an event loop, we need to run this differently
                loop = asyncio.get_running_loop()
                # We're in an async context, this won't work
                logger.error(
                    "generate_config called from async context - use generate_config_async instead"
                )
                return False
            except RuntimeError:
                # No running loop, we can use asyncio.run()
                return asyncio.run(self.generate_config_async(servers))
        except Exception as e:
            logger.error(f"Failed to generate Nginx configuration: {e}", exc_info=True)
            return False

    async def render_config(
        self,
        servers: dict[str, dict[str, Any]],
    ) -> str | None:
        """Render the nginx config string without writing to disk or reloading.

        Returns the rendered config text, or None if nginx updates are disabled.
        Used by NginxReloadScheduler for hash-based change detection.
        """
        if not settings.nginx_updates_enabled:
            return None
        return await self._render_config_impl(servers)

    async def generate_config_async(
        self, servers: dict[str, dict[str, Any]], force_base_config: bool = False
    ) -> bool:
        """Generate Nginx configuration with additional server names and dynamic location blocks.

        Args:
            servers: Dictionary of server path -> server info for location blocks
            force_base_config: If True, generate base config even in registry-only mode
                              (used at startup to ensure nginx has valid config)

        In registry-only mode:
        - At startup (force_base_config=True): generates base config with empty location blocks
        - On server changes (force_base_config=False): skips regeneration (no-op)
        """
        if not settings.nginx_updates_enabled and not force_base_config:
            logger.info(
                f"Skipping nginx config generation - "
                f"DEPLOYMENT_MODE={settings.deployment_mode.value}"
            )
            NGINX_UPDATES_SKIPPED.labels(operation="generate_config").inc()
            return True

        try:
            config_content = await self._render_config_impl(servers)
            if config_content is None:
                return False

            # Write virtual server Lua mapping files (side effect, not part of render)
            await self._commit_virtual_server_mappings()

            # Validate the candidate against the real config tree BEFORE it is
            # allowed to persist as the live config. A rejected candidate is
            # never left on disk (the last-known-good config is restored), so a
            # subsequent nginx cold start cannot be poisoned by a bad render.
            await asyncio.to_thread(
                _write_and_validate_config, settings.nginx_config_path, config_content
            )

            logger.info(
                "Generated Nginx configuration with location blocks and additional server names"
            )

            await asyncio.to_thread(self.reload_nginx, force_base_config)
            return True

        except Exception as e:
            logger.error(f"Failed to generate Nginx configuration: {e}", exc_info=True)
            return False

    async def _render_config_impl(
        self,
        servers: dict[str, dict[str, Any]],
    ) -> str | None:
        """Internal: render the full nginx config content string.

        Returns the rendered string, or None if the template is missing.
        """
        try:
            # Read template
            if not self.nginx_template_path.exists():
                logger.warning(f"Nginx template not found at {self.nginx_template_path}")
                return None

            with open(self.nginx_template_path) as f:
                template_content = f.read()

            # Local-dev / Podman compatibility:
            # The default nginx templates protect `/api/` via `auth_request /validate` (JWT validation).
            # The React dashboard, however, uses cookie-based session auth for `/api/servers` and
            # `/api/tokens/generate`. When auth_request is enabled but Keycloak/Cognito isn't fully
            # configured, nginx returns 403/500 and the UI cannot load.
            #
            # Set NGINX_DISABLE_API_AUTH_REQUEST=true to bypass `auth_request` for `/api/` and rely
            # on FastAPI's own auth (session cookie or bearer token validation inside the app).
            import os

            if os.environ.get("NGINX_DISABLE_API_AUTH_REQUEST", "false").lower() in (
                "1",
                "true",
                "yes",
                "on",
            ):
                protected_api_block = """    # Protected API endpoints - require authentication
    location {{ROOT_PATH}}/api/ {
        # Mark this as a registry-API request (rewrite phase) so the shared
        # /validate subrequest forwards X-Registry-Api-Auth and /validate mints
        # the registry-UI internal token. Set inside the location (NOT a
        # server-scope default), which the auth_request subrequest would clobber.
        set $registry_api_auth "1";

        # Authenticate request via auth server (validates JWT Bearer tokens)
        auth_request /validate;

        # Capture auth server response headers
        auth_request_set $auth_user $upstream_http_x_user;
        auth_request_set $auth_username $upstream_http_x_username;
        auth_request_set $auth_client_id $upstream_http_x_client_id;
        auth_request_set $auth_scopes $upstream_http_x_scopes;
        auth_request_set $auth_method $upstream_http_x_auth_method;
        # Capture the /validate-minted registry-UI token (binds verified identity).
        # The registry verifies this instead of trusting the forgeable X-* headers.
        auth_request_set $auth_internal_token_registry $upstream_http_x_internal_token_registry;

        # Proxy to FastAPI service
        proxy_pass http://127.0.0.1:7860/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Forward validated auth context to FastAPI
        proxy_set_header X-User $auth_user;
        proxy_set_header X-Username $auth_username;
        proxy_set_header X-Client-Id $auth_client_id;
        proxy_set_header X-Scopes $auth_scopes;
        proxy_set_header X-Auth-Method $auth_method;
        # The internal token the registry verifies (it ignores the X-* headers above).
        proxy_set_header X-Internal-Token-Registry $auth_internal_token_registry;

        # Pass through original Authorization header
        proxy_set_header Authorization $http_authorization;

        # Pass all request headers
        proxy_pass_request_headers on;

        # Timeouts
        proxy_connect_timeout 10s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }"""

                unprotected_api_block = """    # API endpoints - FastAPI handles authentication (session cookie / bearer)
    location {{ROOT_PATH}}/api/ {
        # Inbound rate limits still apply even though auth_request is bypassed:
        # /api/ is the highest-volume surface and must stay bounded at the edge,
        # and the registration endpoints keep their stricter per-source cap (the
        # register zone key is empty for non-registration URIs, so it is a no-op
        # for the rest of /api/).
        limit_req zone=mcp_gateway_edge burst=100 nodelay;
        limit_req zone=mcp_gateway_register burst=10 nodelay;
        limit_conn mcp_gateway_conn 100;

        # Proxy to FastAPI service
        proxy_pass http://127.0.0.1:7860/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Pass through original Authorization header (if present)
        proxy_set_header Authorization $http_authorization;

        # Pass all request headers and cookies
        proxy_pass_request_headers on;

        # Timeouts
        proxy_connect_timeout 10s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }"""

                if protected_api_block in template_content:
                    template_content = template_content.replace(
                        protected_api_block, unprotected_api_block
                    )
                    logger.warning(
                        "NGINX_DISABLE_API_AUTH_REQUEST enabled: bypassing auth_request for /api/"
                    )
                else:
                    logger.warning(
                        "NGINX_DISABLE_API_AUTH_REQUEST enabled but could not find /api/ auth_request block in template"
                    )

            # Generate location blocks for enabled and healthy servers with transport support
            # In registry-only mode, skip MCP server location blocks (use empty list)
            location_blocks = []
            if settings.nginx_updates_enabled:
                # Get health service to check server health
                from ..health.service import health_service

                for path, server_info in servers.items():
                    # Local servers don't get nginx routes
                    if server_info.get("deployment") == DeploymentType.LOCAL:
                        logger.debug(f"Skipping local server {path} from nginx config")
                        continue
                    proxy_pass_url = server_info.get("proxy_pass_url")
                    if proxy_pass_url:
                        # Check if server is healthy (including auth-expired which is still reachable)
                        health_status = health_service.server_health_status.get(
                            path, HealthStatus.UNKNOWN
                        )

                        # Include servers that are healthy or just have expired auth (server is up)
                        if HealthStatus.is_healthy(health_status):
                            # Generate transport-aware location blocks
                            transport_blocks = self._generate_transport_location_blocks(
                                path, server_info
                            )
                            location_blocks.extend(transport_blocks)
                            logger.debug(f"Added location blocks for healthy service: {path}")
                        else:
                            # Add commented out block for unhealthy services.
                            # Sanitize both the path and the backend URL: a
                            # newline in a stored value would otherwise break out
                            # of the leading '#' and emit a live directive.
                            # Registration validation now rejects such values;
                            # this protects legacy data persisted beforehand.
                            safe_commented_path = self._sanitize_for_nginx_set(path)
                            safe_commented_url = self._sanitize_for_nginx_set(proxy_pass_url)
                            commented_block = f"""
#    location {{{{ROOT_PATH}}}}{safe_commented_path}/ {{
#        # Service currently unhealthy (status: {health_status})
#        # Proxy to MCP server
#        proxy_pass {safe_commented_url};
#        proxy_http_version 1.1;
#        proxy_set_header Host $host;
#        proxy_set_header X-Real-IP $remote_addr;
#        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
#        proxy_set_header X-Forwarded-Proto $scheme;
#    }}"""
                            location_blocks.append(commented_block)
                            logger.debug(
                                f"Added commented location block for unhealthy service {path} (status: {health_status})"
                            )
            else:
                logger.info(
                    "Registry-only mode: generating base nginx config without MCP server location blocks"
                )

            # Fetch additional server names (cached to avoid per-tick metadata calls)
            if self._cached_server_names is None:
                self._cached_server_names = await self.get_additional_server_names()
            additional_server_names = self._cached_server_names

            # Get API version from constants
            api_version = REGISTRY_CONSTANTS.ANTHROPIC_API_VERSION

            # Parse Keycloak configuration from KEYCLOAK_URL environment variable
            import os

            auth_provider = os.environ.get("AUTH_PROVIDER", "keycloak").lower()

            # Strip Keycloak location blocks from nginx config when not using Keycloak
            if auth_provider != "keycloak":
                template_content = re.sub(
                    r"    # \{\{KEYCLOAK_LOCATIONS_START\}\}.*?# \{\{KEYCLOAK_LOCATIONS_END\}\}\n?",
                    "",
                    template_content,
                    flags=re.DOTALL,
                )
                logger.info(
                    f"AUTH_PROVIDER is '{auth_provider}', removed Keycloak location blocks from nginx config"
                )

            # Strip PingFederate location blocks from nginx config when not using PingFederate
            if auth_provider != "pingfederate":
                template_content = re.sub(
                    r"    # \{\{PINGFEDERATE_LOCATIONS_START\}\}.*?# \{\{PINGFEDERATE_LOCATIONS_END\}\}\n?",
                    "",
                    template_content,
                    flags=re.DOTALL,
                )
                logger.info(
                    f"AUTH_PROVIDER is '{auth_provider}', removed PingFederate location blocks from nginx config"
                )

            # Parse Keycloak configuration from KEYCLOAK_URL environment variable.
            # This always runs so the Keycloak template placeholders are filled even
            # when another provider is active (the location blocks are stripped above).
            keycloak_url = os.environ.get("KEYCLOAK_URL", "http://keycloak:8080")
            try:
                parsed_keycloak = urlparse(keycloak_url)
                keycloak_scheme = parsed_keycloak.scheme or "http"
                keycloak_host = parsed_keycloak.hostname or "keycloak"
                # Use default port based on scheme if not specified
                if parsed_keycloak.port:
                    keycloak_port = str(parsed_keycloak.port)
                else:
                    keycloak_port = "443" if keycloak_scheme == "https" else "8080"

                # Validate that we can actually resolve the hostname
                if not keycloak_host or keycloak_host == "keycloak":
                    # If we end up with just 'keycloak', use the full URL's netloc instead
                    keycloak_host = (
                        parsed_keycloak.netloc.split(":")[0]
                        if parsed_keycloak.netloc
                        else "keycloak"
                    )
                    logger.warning(
                        f"Keycloak hostname is 'keycloak', using netloc instead: {keycloak_host}"
                    )

                logger.info(
                    f"Using Keycloak configuration from KEYCLOAK_URL '{keycloak_url}': "
                    f"{keycloak_scheme}://{keycloak_host}:{keycloak_port}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to parse KEYCLOAK_URL '{keycloak_url}': {e}. Using defaults."
                )
                keycloak_scheme = "http"
                keycloak_host = "keycloak"
                keycloak_port = "8080"

            # Generate version map for multi-version servers
            # In registry-only mode, skip version map generation (use empty string)
            if settings.nginx_updates_enabled:
                version_map = await self._generate_version_map(servers)
            else:
                version_map = ""

            # Replace placeholders in template
            config_content = template_content.replace("{{VERSION_MAP}}", version_map)
            config_content = config_content.replace(
                "{{LOCATION_BLOCKS}}", "\n".join(location_blocks)
            )
            config_content = config_content.replace(
                "{{ADDITIONAL_SERVER_NAMES}}", additional_server_names
            )
            config_content = config_content.replace("{{ANTHROPIC_API_VERSION}}", api_version)
            # egress marker: force-set on the /validate subrequest so a direct
            # :8888 caller cannot supply it. Empty default leaves the header empty
            # (marker disabled), matching auth_server's empty-secret pass-through.
            config_content = config_content.replace(
                "{{NGINX_MARKER_SECRET}}",
                os.environ.get("AUTH_SERVER_NGINX_MARKER_SECRET", ""),
            )
            config_content = config_content.replace("{{KEYCLOAK_SCHEME}}", keycloak_scheme)
            config_content = config_content.replace("{{KEYCLOAK_HOST}}", keycloak_host)
            config_content = config_content.replace("{{KEYCLOAK_PORT}}", keycloak_port)

            # Parse PingFederate configuration, falling back to defaults on any error
            # so a malformed PINGFEDERATE_BASE_URL never breaks config generation.
            pingfederate_url = os.environ.get("PINGFEDERATE_BASE_URL", "http://pingfederate:9032")
            try:
                pf_parsed = urlparse(pingfederate_url)
                pf_scheme = pf_parsed.scheme or "http"
                pf_host = pf_parsed.hostname or "pingfederate"
                pf_port = str(pf_parsed.port or ("443" if pf_scheme == "https" else "9032"))
            except Exception as e:
                logger.warning(
                    f"Failed to parse PINGFEDERATE_BASE_URL '{pingfederate_url}': {e}. Using defaults."
                )
                pf_scheme = "http"
                pf_host = "pingfederate"
                pf_port = "9032"
            config_content = config_content.replace("{{PINGFEDERATE_SCHEME}}", pf_scheme)
            config_content = config_content.replace("{{PINGFEDERATE_HOST}}", pf_host)
            config_content = config_content.replace("{{PINGFEDERATE_PORT}}", pf_port)

            # Parse AUTH_SERVER_URL so nginx templates can reference the
            # auth-server by its actual hostname/FQDN instead of the
            # hard-coded Docker-Compose service name (#553).  Follows the
            # same pattern used for Keycloak and PingFederate above.
            auth_server_url = os.environ.get("AUTH_SERVER_URL", "http://auth-server:8888")
            try:
                parsed_auth = urlparse(auth_server_url)
                auth_host = parsed_auth.hostname or "auth-server"
                if parsed_auth.port:
                    auth_port = str(parsed_auth.port)
                else:
                    auth_scheme = parsed_auth.scheme or "http"
                    auth_port = "443" if auth_scheme == "https" else "8888"

                logger.info(
                    f"Using auth-server configuration from AUTH_SERVER_URL "
                    f"'{auth_server_url}': {auth_host}:{auth_port}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to parse AUTH_SERVER_URL '{auth_server_url}': {e}. Using defaults."
                )
                auth_host = "auth-server"
                auth_port = "8888"
            config_content = config_content.replace("{{AUTH_SERVER_HOST}}", auth_host)
            config_content = config_content.replace("{{AUTH_SERVER_PORT}}", auth_port)

            # Real client-IP recovery (TRUSTED_REAL_IP_CIDRS). Empty by default so
            # edge deployments emit nothing; when trusted proxy CIDRs are set, the
            # audited client IP becomes the end user instead of the load balancer.
            config_content = config_content.replace("{{REAL_IP_CONFIG}}", _render_real_ip_config())

            # Generate registry-only block (503 response for MCP proxy requests)
            registry_only_block = self._generate_registry_only_block()
            config_content = config_content.replace("{{REGISTRY_ONLY_BLOCK}}", registry_only_block)

            # Generate virtual server blocks
            try:
                virtual_server_locations = await self._generate_virtual_server_blocks()

                # Get the virtual servers list for backend locations and mappings
                from registry.repositories.factory import get_virtual_server_repository

                virtual_repo = get_virtual_server_repository()
                virtual_servers = await virtual_repo.list_enabled()

                virtual_backend_locations = await self._generate_virtual_backend_locations(
                    virtual_servers
                )

                # Combine virtual server and backend location blocks
                virtual_blocks = virtual_server_locations
                if virtual_backend_locations:
                    virtual_blocks = (
                        virtual_blocks + "\n" + virtual_backend_locations
                        if virtual_blocks
                        else virtual_backend_locations
                    )

                config_content = config_content.replace("{{VIRTUAL_SERVER_BLOCKS}}", virtual_blocks)

                logger.info(
                    f"Generated virtual server config with {len(virtual_servers)} virtual servers"
                )
            except Exception as e:
                logger.error(f"Failed to generate virtual server config: {e}", exc_info=True)
                config_content = config_content.replace("{{VIRTUAL_SERVER_BLOCKS}}", "")

            # Generate A2A agent reverse-proxy blocks. Opt-in via
            # A2A_REVERSE_PROXY_ENABLED, and only effective in with-gateway mode
            # (a2a_reverse_proxy_effective is the shared flag-AND-with-gateway
            # gate). In registry-only mode they are skipped: the registry-only
            # 503 block already returns 503 for any /agent/* path that is not an
            # API route.
            if settings.a2a_reverse_proxy_effective:
                agent_blocks = await self._generate_agent_location_blocks()
            else:
                agent_blocks = ""
            config_content = config_content.replace("{{AGENT_LOCATION_BLOCKS}}", agent_blocks)

            root_path = os.environ.get("ROOT_PATH", "").rstrip("/")
            config_content = config_content.replace("{{ROOT_PATH}}", root_path)

            # MCP 2025-06-18 / RFC 9728 §5.1: WWW-Authenticate on auth-failure 401s
            # must point at the gateway's PRM endpoint. The URL must match the
            # `resource` field returned by /.well-known/oauth-protected-resource
            # byte-for-byte.
            try:
                from registry.auth.oauth_metadata import (
                    build_canonical_resource_url,
                    build_resource_metadata_url,
                )

                resource_metadata_url = build_resource_metadata_url(
                    build_canonical_resource_url(settings.registry_url)
                )
            except ValueError as exc:
                logger.warning(
                    f"Could not derive MCP_RESOURCE_METADATA_URL "
                    f"(registry_url={settings.registry_url!r}): {exc}. "
                    "Substituting empty value; clients will not see WWW-Authenticate."
                )
                resource_metadata_url = ""
            config_content = config_content.replace(
                "{{MCP_RESOURCE_METADATA_URL}}", resource_metadata_url
            )

            return config_content

        except Exception as e:
            logger.error(f"Failed to render Nginx configuration: {e}", exc_info=True)
            return None

    async def _commit_virtual_server_mappings(self) -> None:
        """Write Lua mapping JSON files for virtual servers.

        Separated from _render_config_impl so that rendering is pure (no disk
        side effects) and mappings are only written when config actually changes.
        """
        try:
            from registry.repositories.factory import get_virtual_server_repository

            virtual_repo = get_virtual_server_repository()
            virtual_servers = await virtual_repo.list_enabled()
            await self._write_virtual_server_mappings(virtual_servers)
        except Exception as e:
            logger.error(f"Failed to write virtual server mappings: {e}")

    def reload_nginx(self, force: bool = False) -> bool:
        """Reload Nginx configuration (if running in appropriate environment).

        Args:
            force: If True, reload even in registry-only mode (used after base config generation)

        In registry-only mode, skip reload unless force=True.
        """
        if not settings.nginx_updates_enabled and not force:
            logger.info(f"Skipping nginx reload - DEPLOYMENT_MODE={settings.deployment_mode.value}")
            NGINX_UPDATES_SKIPPED.labels(operation="reload").inc()
            return True

        # Rate-limit reload signals. nginx needs time for worker processes to
        # gracefully shut down before accepting another SIGHUP. Without this
        # guard, rapid-fire flush_now() calls (e.g. bulk toggle) can spawn
        # multiple master processes and leave workers in "shutting down" limbo.
        import time as _time

        now = _time.monotonic()
        elapsed = now - self._last_reload_time
        if elapsed < self._min_reload_interval_seconds and not force:
            logger.debug(
                "Skipping nginx reload (%.1fs since last, min interval %.1fs)",
                elapsed,
                self._min_reload_interval_seconds,
            )
            return False

        try:
            import subprocess  # nosec B404

            # Test the configuration first before reloading
            test_result = subprocess.run(["nginx", "-t"], capture_output=True, text=True, timeout=5)  # nosec B603 B607 - hardcoded command
            if test_result.returncode != 0:
                logger.error(f"Nginx configuration test failed: {test_result.stderr}")
                logger.info("Skipping Nginx reload due to configuration errors")
                return False

            result = subprocess.run(
                ["nginx", "-s", "reload"], capture_output=True, text=True, timeout=5
            )  # nosec B603 B607 - hardcoded command
            if result.returncode == 0:
                self._last_reload_time = _time.monotonic()
                logger.info("Nginx configuration reloaded successfully")
                return True
            # On Fargate the registry container starts uvicorn before nginx
            # (the entrypoint waits for the runtime nginx config to be
            # generated by uvicorn before starting nginx). When the demo
            # servers register during uvicorn startup and call reload_nginx(),
            # nginx is not running yet — the pid file is empty and nginx -s
            # reload exits non-zero with "invalid PID number". The reload is
            # idempotent, so retry briefly to give the entrypoint time to
            # start nginx. Without this, server location blocks are written
            # to disk but never made active until the next reload (which may
            # never come for auto-registered demo servers).
            stderr = result.stderr or ""
            if "invalid PID number" in stderr or ("open()" in stderr and "nginx.pid" in stderr):
                logger.warning("Nginx not yet started (pid file empty); will retry reload")
                for attempt in range(10):
                    _time.sleep(1.0)
                    retry = subprocess.run(
                        ["nginx", "-s", "reload"], capture_output=True, text=True, timeout=5
                    )  # nosec B603 B607 - hardcoded command
                    if retry.returncode == 0:
                        self._last_reload_time = _time.monotonic()
                        logger.info(
                            "Nginx configuration reloaded successfully after %d retry attempts",
                            attempt + 1,
                        )
                        return True
                logger.error("Nginx still not running after 10 retries; reload abandoned")
                return False
            logger.error(f"Failed to reload Nginx: {stderr}")
            return False
        except FileNotFoundError:
            logger.warning("Nginx not found - skipping reload")
            return False
        except Exception as e:
            logger.error(f"Error reloading Nginx: {e}")
            return False

    def _generate_registry_only_block(self) -> str:
        """
        Generate nginx location block for registry-only mode.

        In registry-only mode, this block returns 503 for paths that look like
        MCP server requests (paths not matching known API prefixes).
        In with-gateway mode, this returns an empty string.

        Returns:
            Nginx location block string or empty string
        """
        if settings.nginx_updates_enabled:
            # with-gateway mode: no blocking needed, MCP servers are proxied
            return ""

        # registry-only mode: block MCP proxy requests with 503
        # This regex matches paths that don't start with known API prefixes
        block = """
    # Registry-only mode: block MCP proxy requests with 503
    # Matches paths that don't start with known API/auth prefixes
    location ~ ^{{ROOT_PATH}}/(?!api/|oauth2/|keycloak/|realms/|resources/|v0\\.1/|health|static/|assets/|_next/|validate).+ {
        default_type application/json;
        return 503 '{"error":"gateway_proxy_disabled","message":"Gateway proxy is disabled in registry-only mode. Connect directly to the MCP server using the proxy_pass_url from server registration.","deployment_mode":"registry-only","hint":"Use GET /api/servers/{path} to retrieve the proxy_pass_url for direct connection."}';
    }"""
        logger.info("Generated registry-only 503 block for MCP proxy requests")
        return block

    async def _generate_version_map(self, servers: dict[str, dict[str, Any]]) -> str:
        """
        Generate nginx map directive for version routing.

        Args:
            servers: Dictionary of server path -> server info

        Returns:
            Nginx map block as string, or empty string if no multi-version servers
        """
        from ..services.server_service import server_service

        map_entries = []

        for path, server_info in servers.items():
            # Check if this server has other versions via other_version_ids
            other_version_ids = server_info.get("other_version_ids", [])

            if not other_version_ids:
                # Single-version server - no map entry needed
                continue

            # Build versions list from active server and other versions
            versions = []

            # Add the current (active) version
            current_version = server_info.get("version", "v1.0.0")
            current_proxy_url = server_info.get("proxy_pass_url", "")
            if current_proxy_url:
                versions.append(
                    {
                        "version": current_version,
                        "proxy_pass_url": current_proxy_url,
                        "is_default": True,
                    }
                )

            # Add other versions by fetching their info
            for version_id in other_version_ids:
                version_info = await server_service.get_server_info(version_id)
                if version_info:
                    versions.append(
                        {
                            "version": version_info.get("version", "unknown"),
                            "proxy_pass_url": version_info.get("proxy_pass_url", ""),
                            "is_default": False,
                        }
                    )

            if len(versions) <= 1:
                # Only one version found, skip
                continue

            # Default backend is the active version's URL
            default_backend = current_proxy_url

            if not default_backend:
                logger.warning(f"No default backend found for {path}, skipping version map")
                continue

            # Escape path for nginx regex
            # Handle paths like /context7, /currenttime/, /ai.smithery-xxx
            escaped_path = re.escape(path.rstrip("/"))

            # Defense-in-depth: escape backend URLs before interpolating them
            # into the quoted nginx map values (belt-and-suspenders with the
            # registration-time metacharacter rejection).
            safe_default_backend = self._sanitize_for_nginx_set(default_backend)

            # Add map entries for this server
            # Entry for no header (empty string after colon)
            map_entries.append(
                f'    "~^{escaped_path}(/.*)?:$"            "{safe_default_backend}";'
            )
            # Entry for explicit "latest"
            map_entries.append(
                f'    "~^{escaped_path}(/.*)?:latest$"      "{safe_default_backend}";'
            )

            # Entry for each version
            for v in versions:
                version_str = v.get("version", "")
                backend_url = v.get("proxy_pass_url", "")
                if version_str and backend_url:
                    safe_backend_url = self._sanitize_for_nginx_set(backend_url)
                    map_entries.append(
                        f'    "~^{escaped_path}(/.*)?:{re.escape(version_str)}$"  "{safe_backend_url}";'
                    )

            logger.info(f"Generated version map entries for {path} with {len(versions)} versions")

        if not map_entries:
            return ""  # No multi-version servers configured

        return f"""# Version routing map (auto-generated)
# Routes requests based on X-MCP-Server-Version header
map "$uri:$http_x_mcp_server_version" $versioned_backend {{
    default "";

{chr(10).join(map_entries)}
}}

"""

    def _sanitize_path_for_location(
        self,
        path: str,
    ) -> str:
        """Sanitize a server path for use as an nginx internal location name.

        Replaces /, -, and . with underscores.

        Args:
            path: Server path (e.g., '/github')

        Returns:
            Sanitized string (e.g., '_github')
        """
        return re.sub(r"[/\-.]", "_", path)

    @staticmethod
    def _is_host_resolvable_at_startup(
        hostname: str,
    ) -> bool:
        """Decide whether an upstream host is safe to resolve at nginx config load.

        Nginx resolves literal proxy_pass hosts when it loads the config and
        fails to start ("host not found in upstream") if any cannot be resolved.
        A fully-qualified domain name (contains a dot, e.g. "api.github.com") or
        an IP address is expected to resolve in any environment, so it is safe to
        emit as a literal proxy_pass (resolved once, no per-request DNS cost).

        A bare hostname with no dot (e.g. a docker-compose service name like
        "currenttime-server") only resolves inside the environment that defines
        it. Treat it as NOT safe so the caller defers resolution to request time.

        Args:
            hostname: The upstream hostname (no scheme or port), may be empty.

        Returns:
            True if the host can be safely resolved at config load, else False.
        """
        if not hostname:
            return False
        # A dot indicates an FQDN or an IPv4 literal; both resolve everywhere.
        # IPv6 literals contain colons and are also always resolvable.
        return "." in hostname or ":" in hostname

    @staticmethod
    def _sanitize_for_nginx_comment(
        value: str,
    ) -> str:
        """Sanitize a string for safe interpolation into an nginx comment.

        Strips newlines and carriage returns to prevent header injection
        via multi-line nginx directives.

        Args:
            value: Raw string (e.g., server_name from user input)

        Returns:
            Sanitized single-line string
        """
        return re.sub(r"[\r\n]+", " ", value)

    @staticmethod
    def _sanitize_for_nginx_set(
        value: str,
    ) -> str:
        """Sanitize a string for safe use inside an nginx set directive's double quotes.

        Escapes double quotes and backslashes, strips newlines, and neutralizes
        the ``$`` sigil. nginx has no backslash escape for ``$`` inside a quoted
        string -- it always begins a variable reference -- so a stray ``$`` cannot
        be safely represented and is stripped (like newlines). None of this
        helper's legitimate inputs (backend URLs, hosts, paths, ids) contain a
        live nginx variable, and a ``$`` reaching here is already malformed; a
        rendered ``$undefined`` would otherwise fail ``nginx -t`` and block the
        reload. It cannot break out to a new directive (that needs ``"`` + ``;``,
        both handled), so this is defense-in-depth, not the primary guard.

        Args:
            value: Raw string (e.g., server_id from URL path)

        Returns:
            Escaped string safe for use in: set $var "value";
        """
        sanitized = re.sub(r"[\r\n]+", " ", value)
        sanitized = sanitized.replace("\\", "\\\\")
        sanitized = sanitized.replace('"', '\\"')
        sanitized = sanitized.replace("$", "")
        return sanitized

    async def _generate_virtual_server_blocks(self) -> str:
        """Generate nginx location blocks for enabled virtual servers.

        Returns:
            Nginx configuration string with virtual server location blocks
        """
        try:
            from registry.repositories.factory import get_virtual_server_repository

            virtual_repo = get_virtual_server_repository()
            virtual_servers = await virtual_repo.list_enabled()

            if not virtual_servers:
                logger.info("No enabled virtual servers found")
                return ""

            location_blocks = []
            for vs in virtual_servers:
                # Extract server_id from path (e.g., '/virtual/dev-essentials' -> 'dev-essentials')
                server_id = vs.path.replace("/virtual/", "", 1)

                # Sanitize values for safe interpolation into nginx config
                safe_name = self._sanitize_for_nginx_comment(vs.server_name)
                safe_id = self._sanitize_for_nginx_set(server_id)
                # The path is interpolated into a location directive; escape any
                # nginx-special characters (defense-in-depth over Pydantic path
                # validation) so it cannot break out of the directive.
                # Normalise to a trailing slash for the same reason as real servers
                # (issue #1501): a bare `location /virtual/dev` prefix-matches
                # unrelated routes like `/virtual/devtools`, hijacking them into the
                # /validate auth subrequest. `location /virtual/dev/` only matches the
                # subtree. The virtual_router.lua content handler receives the full
                # URI, so the added slash does not affect routing.
                safe_vs_path = self._sanitize_for_nginx_set(vs.path).rstrip("/") + "/"

                block = f"""
    # Virtual MCP Server: {safe_name}
    location {{{{ROOT_PATH}}}}{safe_vs_path} {{
        # Inbound rate limiting: this path fans out to the shared /validate auth
        # subrequest, so bound it at the edge (zones declared at http scope in
        # docker/nginx_rev_proxy_*.conf) to keep a flood from exhausting /validate.
        limit_req zone=mcp_gateway_edge burst=100 nodelay;
        limit_conn mcp_gateway_conn 100;

        set $virtual_server_id "{safe_id}";
        auth_request /validate;
        auth_request_set $auth_scopes $upstream_http_x_scopes;
        auth_request_set $auth_user $upstream_http_x_user;
        auth_request_set $auth_username $upstream_http_x_username;
        auth_request_set $auth_method $upstream_http_x_auth_method;
        rewrite_by_lua_file /etc/nginx/lua/capture_body.lua;
        content_by_lua_file /etc/nginx/lua/virtual_router.lua;

        # Route 401s through @auth_error so the WWW-Authenticate header
        # mandated by RFC 9728 §5.1 is emitted (issue #989).
        error_page 401 = @auth_error;
        error_page 403 = @forbidden_error;
    }}"""
                location_blocks.append(block)
                logger.debug(f"Generated virtual server location block for {vs.path}")

            logger.info(f"Generated {len(location_blocks)} virtual server location blocks")
            return "\n".join(location_blocks)

        except Exception as e:
            logger.error(f"Failed to generate virtual server blocks: {e}", exc_info=True)
            return ""

    async def _generate_virtual_backend_locations(
        self,
        virtual_servers: list,
    ) -> str:
        """Generate internal nginx location blocks for virtual server backends.

        Args:
            virtual_servers: List of VirtualServerConfig objects

        Returns:
            Nginx configuration string with internal backend location blocks
        """
        try:
            from registry.repositories.factory import get_server_repository

            server_repo = get_server_repository()

            # Collect unique backend server paths
            backend_paths = set()
            for vs in virtual_servers:
                for tm in vs.tool_mappings:
                    backend_paths.add(tm.backend_server_path)

            if not backend_paths:
                return ""

            location_blocks = []
            for backend_path in sorted(backend_paths):
                sanitized = self._sanitize_path_for_location(backend_path)
                server_info = await server_repo.get(backend_path)

                if not server_info:
                    logger.warning(
                        f"Backend server not found for virtual server mapping: {backend_path}"
                    )
                    continue

                proxy_pass_url = server_info.get("proxy_pass_url", "")
                if not proxy_pass_url:
                    logger.warning(f"No proxy_pass_url for backend server: {backend_path}")
                    continue

                # Determine upstream host from proxy_pass_url
                parsed_url = urlparse(proxy_pass_url)
                upstream_host = parsed_url.netloc

                # Build MCP endpoint URL from the server's mcp_endpoint or proxy_pass_url.
                # mcp_endpoint is defended by two layers: (1) registration-time
                # validation rejects an mcp_endpoint containing nginx
                # metacharacters (including "$") via the same canonical guard as
                # proxy_pass_url, so a "$" can never legitimately be stored; and
                # (2) the render-time _sanitize_for_nginx_set below strips "$"
                # and escapes quotes/backslashes as defense in depth for any
                # legacy value persisted before validation existed.
                mcp_endpoint = server_info.get("mcp_endpoint", "")
                if mcp_endpoint:
                    mcp_parsed = urlparse(mcp_endpoint)
                    mcp_path = mcp_parsed.path.rstrip("/")
                    # Construct full MCP URL from proxy_pass host + mcp path
                    mcp_proxy_url = f"{parsed_url.scheme}://{parsed_url.netloc}{mcp_path}"
                else:
                    # Fallback: use proxy_pass_url, appending /mcp only if needed
                    bare_url = proxy_pass_url.rstrip("/")
                    # Check if URL already ends with common MCP endpoint paths
                    if bare_url.endswith("/mcp") or bare_url.endswith("/sse"):
                        mcp_proxy_url = bare_url
                    else:
                        mcp_proxy_url = f"{bare_url}/mcp"

                # Use regular internal location (not named @) so proxy_pass
                # can include a URI path for the MCP endpoint
                location_path = f"/_vs_backend{sanitized}"

                # Decide how to emit proxy_pass based on whether the backend host
                # is safe to resolve at config-load time.
                #
                # Normal external hosts (with a dot, like api.github.com, or an IP)
                # use a literal proxy_pass: nginx resolves them once at startup and
                # caches for the worker's life, so there is no per-request DNS cost.
                #
                # Bare hostnames (no dot, e.g. a docker-compose service name like
                # "currenttime-server") are NOT resolvable in every environment.
                # A literal proxy_pass to such a host makes nginx fail to start with
                # "host not found in upstream", crashing the whole registry
                # container. For those we pass the URL through a variable plus a
                # resolver so nginx resolves at request time instead; an
                # unresolvable backend then degrades to a per-request 502 for only
                # that backend rather than taking the gateway down at boot.
                backend_hostname = parsed_url.hostname or ""
                host_is_resolvable_at_startup = self._is_host_resolvable_at_startup(
                    backend_hostname
                )

                # Defense-in-depth: escape the backend URL and Host header before
                # interpolating them into proxy_pass / set / proxy_set_header
                # directives, matching the escaping applied to the versioned
                # backend map. Registration-time validation already rejects URLs
                # with nginx metacharacters; escaping here means legacy data
                # persisted before validation still cannot break out of the
                # directive context.
                safe_mcp_proxy_url = self._sanitize_for_nginx_set(mcp_proxy_url)
                safe_upstream_host = self._sanitize_for_nginx_set(upstream_host)

                if host_is_resolvable_at_startup:
                    proxy_directive = f"proxy_pass {safe_mcp_proxy_url};"
                else:
                    # sanitized is already underscore-safe (valid nginx var name).
                    backend_var = f"$vs_backend{sanitized}"
                    dns_resolver = os.environ.get("NGINX_DNS_RESOLVER", "8.8.8.8 8.8.4.4")
                    dns_resolver_timeout = os.environ.get("NGINX_DNS_RESOLVER_TIMEOUT", "5")
                    proxy_directive = (
                        f"resolver {dns_resolver} valid=10s;\n"
                        f"        resolver_timeout {dns_resolver_timeout}s;\n"
                        f'        set {backend_var} "{safe_mcp_proxy_url}";\n'
                        f"        proxy_pass {backend_var};"
                    )

                block = f"""
    location {location_path} {{
        internal;
        {proxy_directive}
        proxy_http_version 1.1;
        proxy_ssl_server_name on;
        proxy_set_header Host {safe_upstream_host};
        # SECURITY: this location proxies directly to a registrant-controlled
        # (not fully trusted) MCP backend. Never relay the caller's registry
        # credential here -- clearing Authorization AND Cookie prevents a
        # malicious registered upstream from capturing and replaying the
        # caller's gateway bearer token or registry session cookie against the
        # registry API. This location is reached via a Lua subrequest that
        # inherits the parent request's headers, so Cookie must be explicitly
        # cleared or the user's session cookie would be forwarded verbatim.
        # The gateway authenticates the upstream via its own mechanism, not by
        # forwarding the caller's credential.
        proxy_set_header Authorization "";
        proxy_set_header Cookie "";
        proxy_buffering off;
        proxy_set_header Accept "application/json, text/event-stream";
        proxy_set_header Content-Type $content_type;
    }}"""
                location_blocks.append(block)
                logger.debug(
                    f"Generated virtual backend location for {backend_path} -> {location_path}"
                )

            logger.info(f"Generated {len(location_blocks)} virtual backend location blocks")
            return "\n".join(location_blocks)

        except Exception as e:
            logger.error(f"Failed to generate virtual backend locations: {e}", exc_info=True)
            return ""

    async def _write_virtual_server_mappings(
        self,
        virtual_servers: list,
    ) -> None:
        """Write pre-computed mapping JSON files for each virtual server.

        These files are consumed by virtual_router.lua at request time.

        Args:
            virtual_servers: List of VirtualServerConfig objects
        """
        try:
            from registry.repositories.factory import get_server_repository

            server_repo = get_server_repository()

            mappings_dir = Path("/etc/nginx/lua/virtual_mappings")
            mappings_dir.mkdir(parents=True, exist_ok=True)

            for vs in virtual_servers:
                server_id = vs.path.replace("/virtual/", "", 1)

                # Build scope override lookup
                scope_overrides = {}
                for override in vs.tool_scope_overrides:
                    scope_overrides[override.tool_alias] = override.required_scopes

                tools = []
                tool_backend_map = {}

                for tm in vs.tool_mappings:
                    sanitized_backend = self._sanitize_path_for_location(tm.backend_server_path)
                    backend_location = f"/_vs_backend{sanitized_backend}"
                    tool_display_name = tm.alias if tm.alias else tm.tool_name

                    # Get tool metadata from the backend server
                    server_info = await server_repo.get(tm.backend_server_path)
                    description = tm.description_override or ""
                    input_schema: dict[str, Any] = {}

                    if server_info:
                        server_tools = server_info.get("tool_list", [])
                        for st in server_tools:
                            if st.get("name") == tm.tool_name:
                                description = tm.description_override or st.get("description", "")
                                input_schema = st.get("inputSchema", st.get("input_schema", {}))
                                break

                    input_schema = _ensure_mcp_compliant_schema(input_schema)

                    # Per-tool scopes
                    required_scopes = scope_overrides.get(tool_display_name, [])

                    tools.append(
                        {
                            "name": tool_display_name,
                            "original_name": tm.tool_name,
                            "description": description,
                            "inputSchema": input_schema,
                            "backend_location": backend_location,
                            "backend_version": tm.backend_version,
                            "required_scopes": required_scopes,
                        }
                    )

                    tool_backend_map[tool_display_name] = {
                        "backend_location": backend_location,
                        "original_name": tm.tool_name,
                        "backend_version": tm.backend_version,
                    }

                mapping_data = {
                    "server_name": vs.server_name,
                    "required_scopes": vs.required_scopes,
                    "tools": tools,
                    "tool_backend_map": tool_backend_map,
                }

                mapping_path = mappings_dir / f"{server_id}.json"
                with open(mapping_path, "w") as f:
                    json.dump(mapping_data, f, indent=2, default=str)

                logger.debug(f"Wrote virtual server mapping: {mapping_path}")

            logger.info(f"Wrote {len(virtual_servers)} virtual server mapping files")

        except Exception as e:
            logger.error(f"Failed to write virtual server mappings: {e}", exc_info=True)

    @staticmethod
    async def _agent_backend_resolves(
        hostname: str,
    ) -> bool:
        """Return True if the agent backend hostname resolves right now.

        Used to fail safe before emitting a literal ``proxy_pass`` in an agent
        block: an unresolvable host would make the whole nginx reload fail. Runs
        the blocking ``getaddrinfo`` in a thread so it does not block the event
        loop. An IP literal or a resolvable name (including a bare docker/service
        name valid on this host's network) returns True; a dead name returns
        False. Fails safe to False on lookup error so a bad host is skipped, not
        emitted.

        Args:
            hostname: Upstream host (no scheme or port); may be empty.

        Returns:
            True if the host resolves, else False.
        """
        if not hostname:
            return False
        import socket

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, socket.getaddrinfo, hostname, None)
            return True
        except (OSError, UnicodeError) as exc:
            logger.debug(f"Agent backend host {hostname!r} did not resolve: {exc}")
            return False

    async def _generate_agent_location_blocks(self) -> str:
        """Generate nginx reverse-proxy location blocks for enabled A2A agents.

        Mirrors the MCP-server and virtual-server generators: each enabled
        agent gets location blocks that proxy A2A traffic through the gateway
        (centralized auth, metrics, network isolation) instead of clients
        connecting directly to the agent backend.

        Returns:
            Nginx configuration string, or an empty string when there are no
            enabled agents.
        """
        try:
            from registry.services.agent_service import agent_service

            enabled_paths = await agent_service.get_enabled_agents()
            if not enabled_paths:
                logger.debug("No enabled A2A agents found")
                return ""

            location_blocks = []
            for path in enabled_paths:
                agent = await agent_service.get_agent_info(path)
                if agent is None:
                    logger.warning(f"Enabled agent '{path}' has no card; skipping nginx block")
                    continue

                # Proxy to the real backend. In reverse-proxy mode the advertised
                # url is the gateway-facing address, so the registrant's backend
                # lives in proxy_pass_url; fall back to url for agents registered
                # before the flag was on (proxy_pass_url unset).
                backend_url = (getattr(agent, "proxy_pass_url", None) or agent.url or "").rstrip(
                    "/"
                )
                if not backend_url:
                    logger.warning(f"Agent '{path}' has no backend url; skipping nginx block")
                    continue

                # Only proxy true A2A agents. A non-A2A agent that happens to
                # carry a URL must not get a JSON-RPC proxy route.
                if (agent.supported_protocol or "").lower() != "a2a":
                    logger.debug(
                        f"Agent '{path}' protocol is {agent.supported_protocol!r} "
                        "(not 'a2a'); skipping nginx block"
                    )
                    continue

                # Never advertise a proxy path to a backend that is not
                # known-healthy, so the gateway does not route to a dead agent.
                if not HealthStatus.is_healthy(agent.health_status):
                    logger.debug(
                        f"Agent '{path}' health is {agent.health_status!r} "
                        "(not healthy); skipping nginx block"
                    )
                    continue

                # The agent block emits a LITERAL proxy_pass, which nginx resolves
                # at config-load time; a backend host that does not resolve then
                # makes the WHOLE nginx reload fail ("host not found in upstream"),
                # taking every route down, not just this agent. Fail safe: verify
                # the host resolves now and skip the block with a warning if it
                # does not (same posture as the health and no-url skips above), so
                # one dead backend host can never crash the reload. This is a real
                # DNS check (not the dot heuristic) so a legitimately-resolvable
                # bare docker/service name is kept and a dead FQDN is caught.
                backend_host = urlparse(backend_url).hostname or ""
                if not await self._agent_backend_resolves(backend_host):
                    logger.warning(
                        f"Agent '{path}' backend host {backend_host!r} does not "
                        "resolve; skipping nginx block so the reload cannot fail"
                    )
                    continue

                agent_path = (agent.path or path).strip("/")
                try:
                    block = self._create_agent_location_block(
                        agent_path,
                        backend_url,
                        agent.name,
                    )
                except ValueError as exc:
                    logger.warning(f"Skipping agent '{path}' with unsafe nginx input: {exc}")
                    continue
                location_blocks.append(block)
                logger.debug(f"Generated A2A agent location block for {agent_path}")

            logger.info(f"Generated {len(location_blocks)} A2A agent location blocks")
            return "\n".join(location_blocks)

        except Exception as e:
            logger.error(f"Failed to generate agent location blocks: {e}", exc_info=True)
            return ""

    def _create_agent_location_block(
        self,
        agent_path: str,
        backend_url: str,
        agent_name: str,
    ) -> str:
        """Create nginx location blocks for a single A2A agent.

        Emits two prefix locations under ``{ROOT_PATH}/agent/{agent_path}/``:
          1. The agent card (``.well-known/agent-card.json``) for discovery.
          2. The JSON-RPC endpoint for ``message/send`` / ``message/stream``.

        Both are protected by the ``/validate`` auth subrequest (the same hop
        used for MCP servers) and proxy straight to the agent backend. A2A is
        JSON-RPC over HTTP, so no protocol translation is required.

        Args:
            agent_path: Agent path without surrounding slashes
                (e.g. ``flight-booking-agent``).
            backend_url: Agent backend base URL without a trailing slash.
            agent_name: Human-readable agent name (used in a comment only).

        Returns:
            Nginx configuration string with the two location blocks.

        Raises:
            ValueError: If ``agent_path`` or ``backend_url`` contains characters
                that are unsafe to interpolate into an nginx config.
        """
        if not _NGINX_AGENT_PATH_SAFE.match(agent_path):
            raise ValueError(f"unsafe agent path: {agent_path!r}")
        if not _NGINX_AGENT_URL_SAFE.match(backend_url):
            raise ValueError(f"unsafe agent url: {backend_url!r}")

        parsed_url = urlparse(backend_url)
        upstream_host = parsed_url.netloc
        # External services (https or FQDN) use the upstream hostname; bare
        # internal hostnames preserve the original Host header.
        if parsed_url.scheme == "https" or "." in upstream_host:
            host_header = upstream_host
        else:
            host_header = "$host"

        dns_resolver = os.environ.get("NGINX_DNS_RESOLVER", "8.8.8.8 8.8.4.4")
        dns_resolver_timeout = os.environ.get("NGINX_DNS_RESOLVER_TIMEOUT", "5")
        safe_name = self._sanitize_for_nginx_comment(agent_name)
        route = f"{AGENT_ROUTE_PREFIX}/{agent_path}"

        return f"""
    # A2A agent card (discovery): {safe_name}
    # Exact match so suffixes (e.g. /.well-known/agent-card.json/../secret)
    # cannot be smuggled through this proxy.
    location = {{{{ROOT_PATH}}}}{route}/.well-known/agent-card.json {{
        resolver {dns_resolver} valid=10s;
        resolver_timeout {dns_resolver_timeout}s;
        auth_request /validate;
        auth_request_set $auth_scopes $upstream_http_x_scopes;
        proxy_pass {backend_url}/.well-known/agent-card.json;
        proxy_http_version 1.1;
        proxy_ssl_server_name on;
        proxy_set_header Host {host_header};
        # SECURITY (A2A egress trust model): X-Authorization carries the caller's
        # gateway credential -- it is validated at /validate and MUST NOT reach
        # this registrant-controlled backend, or a malicious agent could replay
        # it against the registry (the B1 / #1391 class of bug). Strip it and the
        # session Cookie. The standard Authorization header is left intact: per
        # the A2A spec, credentials are obtained out-of-band by the calling agent
        # and passed end-to-end in Authorization for the target agent to
        # authenticate -- the gateway is not a credential broker here.
        proxy_set_header X-Authorization "";
        proxy_set_header Cookie "";
        # Rewrite the card's endpoint URLs from the backend to this gateway so
        # clients send JSON-RPC back through the proxy. Body size changes, so
        # Content-Length must be cleared before the rewrite runs.
        header_filter_by_lua_block {{ ngx.header.content_length = nil }}
        body_filter_by_lua_file /etc/nginx/lua/agent_card_rewrite.lua;
        error_page 401 = @auth_error;
        error_page 403 = @forbidden_error;
    }}

    # A2A agent JSON-RPC endpoint: {safe_name}
    location {{{{ROOT_PATH}}}}{route}/ {{
        resolver {dns_resolver} valid=10s;
        resolver_timeout {dns_resolver_timeout}s;
        auth_request /validate;
        auth_request_set $auth_user $upstream_http_x_user;
        auth_request_set $auth_username $upstream_http_x_username;
        auth_request_set $auth_scopes $upstream_http_x_scopes;
        auth_request_set $auth_method $upstream_http_x_auth_method;
        # Rate-limit passthrough (issue #295): capture the throttle marker + headers
        # so @forbidden_error can turn a throttle-403 into a real 429 + Retry-After.
        auth_request_set $rl_throttled $upstream_http_x_ratelimit_throttled;
        auth_request_set $rl_limit $upstream_http_x_ratelimit_limit;
        auth_request_set $rl_reset $upstream_http_x_ratelimit_reset;
        auth_request_set $rl_retry $upstream_http_retry_after;

        # Attribute metrics to this specific agent. Without this, emit_metrics
        # derives the name from the first URI segment ("agent"), bucketing every
        # agent together. agent_path is validated by _NGINX_AGENT_PATH_SAFE.
        set $metrics_server_name "agent/{agent_path}";
        # Capture the JSON-RPC body (rewrite phase) so emit_metrics can record
        # the A2A method; per-agent invoke is enforced by the auth server via
        # the /validate subrequest above.
        rewrite_by_lua_file /etc/nginx/lua/capture_body.lua;
        log_by_lua_file /etc/nginx/lua/emit_metrics.lua;

        proxy_pass {backend_url}/;
        proxy_http_version 1.1;
        proxy_ssl_server_name on;
        # message/stream (SSE) can stay open for minutes; override nginx's 60s
        # default so streaming responses are not killed mid-flight.
        proxy_connect_timeout 10s;
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
        proxy_set_header Host {host_header};
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Original-URL $scheme://$host$request_uri;
        # SECURITY (A2A egress trust model): X-Authorization carries the caller's
        # gateway credential -- it is validated at /validate and MUST NOT reach
        # this registrant-controlled backend, or a malicious agent could replay
        # it against the registry (the B1 / #1391 class of bug). Strip it and the
        # session Cookie (nginx forwards client request headers by default, so
        # Cookie must be cleared explicitly). The standard Authorization header is
        # left intact and forwarded end-to-end: per the A2A spec, the calling
        # agent obtains the target agent's credential out-of-band and presents it
        # in Authorization for the target to authenticate. The gateway is a policy
        # gate, not a credential broker -- it never mints or vends the agent's
        # credential. /validate authenticates the caller on X-Authorization only
        # (no Authorization fallback for agent paths) and refuses to forward an
        # Authorization equal to the validated X-Authorization, so a caller that
        # duplicates its gateway token into both headers cannot leak it here.
        proxy_set_header X-Authorization "";
        proxy_set_header Cookie "";

        # Forward validated auth context to the agent backend
        proxy_set_header X-User $auth_user;
        proxy_set_header X-Username $auth_username;
        proxy_set_header X-Scopes $auth_scopes;
        proxy_set_header X-Auth-Method $auth_method;

        # message/stream uses SSE; disable buffering for incremental delivery
        proxy_buffering off;
        proxy_set_header Accept $http_accept;

        error_page 401 = @auth_error;
        error_page 403 = @forbidden_error;
    }}"""

    def _generate_transport_location_blocks(self, path: str, server_info: dict[str, Any]) -> list:
        """Generate nginx location blocks for different transport types."""
        blocks: list[str] = []

        # Render-time mirror of validate_server_path (issue #1501). A path that
        # normalizes to empty (a slashes-only value that bypassed registration
        # validation -- e.g. legacy data persisted before the guard existed)
        # would render as a gateway-wide `location /` block, subjecting every URL
        # to the /validate auth subrequest and shadowing the static catch-all.
        # Skip it and log, rather than emit a config that hijacks the whole
        # gateway: fail closed for this one server, keep the rest of the config.
        if not path or not path.strip("/"):
            logger.error(
                "Skipping nginx location block for server with empty/slashes-only "
                "path %r: it would render as a gateway-wide 'location /' block. "
                "This path should have been rejected at registration.",
                path,
            )
            return blocks

        proxy_pass_url = server_info.get("proxy_pass_url", "")
        supported_transports = server_info.get("supported_transports", ["streamable-http"])

        # Use the proxy_pass_url exactly as specified in the JSON file
        # Users are responsible for including /mcp, /sse, or any other path in the URL
        proxy_url = proxy_pass_url

        # Determine transport type based on supported_transports
        if not supported_transports:
            # Default to streamable-http if no transports specified
            transport_type = "streamable-http"
            logger.info(
                f"Server {path}: No supported_transports specified, defaulting to streamable-http"
            )
        elif "streamable-http" in supported_transports and "sse" in supported_transports:
            # If both are supported, prefer streamable-http
            transport_type = "streamable-http"
            logger.info(
                f"Server {path}: Both streamable-http and sse supported, preferring streamable-http"
            )
        elif "sse" in supported_transports:
            # SSE only
            transport_type = "sse"
            logger.info(f"Server {path}: Only sse transport supported, using sse")
        elif "streamable-http" in supported_transports:
            # Streamable-http only
            transport_type = "streamable-http"
            logger.info(
                f"Server {path}: Only streamable-http transport supported, using streamable-http"
            )
        else:
            # Default to streamable-http if unknown transport
            transport_type = "streamable-http"
            logger.info(
                f"Server {path}: Unknown transport types {supported_transports}, defaulting to streamable-http"
            )

        # Create a single location block for this server
        # The proxy_pass URL is used exactly as provided in the server configuration
        logger.info(f"Server {path}: Using proxy_pass URL as configured: {proxy_url}")

        block = self._create_location_block(path, proxy_url, transport_type, server_info)
        blocks.append(block)

        return blocks

    def _create_location_block(
        self,
        path: str,
        proxy_pass_url: str,
        transport_type: str,
        server_info: dict[str, Any] | None = None,
    ) -> str:
        """Create a single nginx location block with transport-specific configuration.

        Args:
            path: Server location path
            proxy_pass_url: Default backend URL
            transport_type: Transport type (streamable-http, sse, direct)
            server_info: Full server info dict (for version support)

        Returns:
            Nginx location block as string
        """
        # Check if this server has multiple versions
        # The MongoDB document stores linked version IDs in "other_version_ids"
        has_versions = False
        if server_info:
            other_version_ids = server_info.get("other_version_ids", [])
            has_versions = len(other_version_ids) > 0

        # Defense-in-depth: escape the backend URL before it is interpolated
        # into any nginx directive/string. Registration-time validation already
        # rejects URLs containing nginx metacharacters, but escaping here means a
        # value that somehow reaches this point (e.g. legacy data persisted
        # before validation existed) still cannot break out of the quoted
        # `set $backend_url "..."` context.
        safe_proxy_pass_url = self._sanitize_for_nginx_set(proxy_pass_url)

        # For servers that need a per-server PRM (obo_exchange on any provider,
        # oauth_user on Entra only -- see server_needs_per_server_prm), the 401
        # WWW-Authenticate must point MCP clients at the PER-SERVER PRM (so the
        # client discovers the per-server resource that matches its connection URL
        # and the Entra App ID URI, not the bare-origin gateway PRM Entra rejects).
        # Override the default global $mcp_resource_metadata in this location only.
        # Keycloak/Cognito 3LO keeps the global PRM (unchanged working behavior).
        # RFC 9728 clients follow the resource_metadata from the 401 header in
        # preference to guessing.
        from registry.api.wellknown_routes import server_needs_per_server_prm

        obo_resource_metadata = ""
        if server_info and server_needs_per_server_prm(server_info.get("egress_auth_mode")):
            from registry.auth.oauth_metadata import build_per_server_prm_url

            try:
                append_mcp = server_info.get("append_mcp_path") is not False
                per_server_prm = build_per_server_prm_url(
                    settings.registry_url, path, append_mcp=append_mcp
                )
                # Defense-in-depth: the PRM URL embeds the registrant-supplied
                # server path (build_per_server_prm_url does a raw concat, no
                # escaping), so sanitize before it lands in the quoted
                # `set $mcp_resource_metadata "..."` directive -- matching the
                # escaping applied to every other interpolated value in this file
                # (e.g. $backend_url above). Without it, a path containing a
                # double-quote + semicolon could break out of the string and
                # inject nginx directives into the obo location block.
                safe_per_server_prm = self._sanitize_for_nginx_set(per_server_prm)
                obo_resource_metadata = (
                    f'\n        set $mcp_resource_metadata "{safe_per_server_prm}";'
                )
            except ValueError:
                obo_resource_metadata = ""

        # Extract hostname from proxy_pass_url for external services
        parsed_url = urlparse(proxy_pass_url)
        upstream_host = parsed_url.netloc

        # Determine whether to use upstream hostname or preserve original host
        # For external services (https), use the upstream hostname
        # For internal services (http without dots in hostname), preserve original host
        if parsed_url.scheme == "https" or "." in upstream_host:
            # External service - use upstream hostname. Sanitize before it is
            # interpolated into `proxy_set_header Host ...` (defense-in-depth: the
            # netloc comes from a proxy_pass_url whose metacharacters are rejected
            # at registration, but legacy data must not break out of the directive).
            host_header = self._sanitize_for_nginx_set(upstream_host)
            logger.info(f"Using upstream hostname for Host header: {host_header}")
        else:
            # Internal service - preserve original host
            host_header = "$host"
            logger.info("Using original host for Host header: $host")

        # Issue #1026 - route MCP traffic through auth_server proxy for tools/list filtering.
        # All MCP POSTs go to auth-server:8888/mcp-proxy/{server_name} instead of the upstream
        # directly. auth_server forwards the request to the original upstream (passed via
        # X-Upstream-Url) and filters `tools/list` responses when MCP_TOOLS_LIST_FILTER_ENABLED
        # is set. All other JSON-RPC methods are passed through unchanged, so the only latency
        # impact is the extra hop. Nginx never inspects the body or flag; auth_server decides.
        # We use the header strategy (X-Upstream-Url) so auth_server does not need a separate
        # MongoDB lookup per request, and version-aware upstream selection stays in nginx.
        mcp_proxy_target = f"{settings.auth_server_url}/mcp-proxy/" + path.strip("/") + "/"
        if has_versions:
            # Multi-version server: use map variable with fallback, then proxy the selected
            # upstream URL to auth_server via X-Upstream-Url so it knows where to forward.
            proxy_directive = f"""
        # Version routing - use header-based backend selection
        # If X-MCP-Server-Version header matches a version, use that backend
        # Otherwise, use the default backend
        set $backend_url "{safe_proxy_pass_url}";
        if ($versioned_backend != "") {{
            set $backend_url $versioned_backend;
        }}

        # Tell auth_server which upstream to forward to after filtering
        proxy_set_header X-Upstream-Url $backend_url;

        # Proxy to auth_server mcp-proxy hop (Issue #1026)
        proxy_pass {mcp_proxy_target};"""
            version_headers = """

        # Add version info to response
        add_header X-MCP-Version-Routing "enabled" always;"""
        else:
            # Single-version server: forward the fixed upstream via X-Upstream-Url header.
            # Set $backend_url (in the rewrite phase) so the /validate subrequest can
            # bind it into the internal token via X-Resolved-Upstream, matching what
            # is forwarded here. Quote the URL so nginx does not interpret braces.
            proxy_directive = f"""
        set $backend_url "{safe_proxy_pass_url}";

        # Tell auth_server which upstream to forward to after filtering
        proxy_set_header X-Upstream-Url $backend_url;

        # Proxy to auth_server mcp-proxy hop (Issue #1026)
        proxy_pass {mcp_proxy_target};"""
            version_headers = ""

        # Resolve nginx read/send timeout from MCP_PROXY_TIMEOUT (+ buffer) so
        # the inner auth-server hop always times out first.
        mcp_proxy_read_timeout = _resolve_mcp_proxy_read_timeout_seconds()

        # Common proxy settings
        common_settings = f"""
        # Inbound rate limiting: every MCP request fans out to the shared
        # /validate auth subrequest, so bound it at the edge (zones + rationale
        # are declared at http scope in docker/nginx_rev_proxy_*.conf). Keeps a
        # flood on one server's /mcp-proxy/ path from exhausting /validate for
        # all servers. burst+nodelay give bursty MCP clients headroom.
        limit_req zone=mcp_gateway_edge burst=100 nodelay;
        limit_conn mcp_gateway_conn 100;

        # DNS resolver for dynamic proxy_pass upstreams.
        # Default: 8.8.8.8 8.8.4.4 (public DNS).
        # Override with NGINX_DNS_RESOLVER env var for environments where
        # backend servers use internal hostnames (e.g., Kubernetes
        # cluster-local names like *.svc.cluster.local need kube-dns).
        resolver {os.environ.get("NGINX_DNS_RESOLVER", "8.8.8.8 8.8.4.4")} valid=10s;
        resolver_timeout {os.environ.get("NGINX_DNS_RESOLVER_TIMEOUT", "5")}s;

        # Upstream timeouts for the browser -> nginx -> auth-server mcp_proxy
        # hop. read/send are derived from MCP_PROXY_TIMEOUT (+ a fixed buffer)
        # so nginx outlives the inner auth-server -> upstream hop: a
        # slow-but-progressing upstream is severed by auth-server (clean 504)
        # rather than cut short by nginx. Default (30s) yields 60s here,
        # matching nginx's historical implicit default, so behavior is
        # unchanged unless MCP_PROXY_TIMEOUT is raised. connect stays short
        # (the hop is in-cluster).
        proxy_connect_timeout 10s;
        proxy_send_timeout {mcp_proxy_read_timeout}s;
        proxy_read_timeout {mcp_proxy_read_timeout}s;

        # Authenticate request - pass entire request to auth server
        auth_request /validate;

        # Capture auth server response headers for forwarding
        auth_request_set $auth_user $upstream_http_x_user;
        auth_request_set $auth_username $upstream_http_x_username;
        auth_request_set $auth_client_id $upstream_http_x_client_id;
        auth_request_set $auth_scopes $upstream_http_x_scopes;
        auth_request_set $auth_method $upstream_http_x_auth_method;
        auth_request_set $auth_server_name $upstream_http_x_server_name;
        auth_request_set $auth_tool_name $upstream_http_x_tool_name;
        # Capture the /validate-minted internal JWT (binds identity/scopes/upstream).
        # mcp_proxy verifies this instead of trusting the forgeable X-* headers below.
        auth_request_set $auth_internal_token $upstream_http_x_internal_token;
        # Rate-limit passthrough (issue #295): capture the throttle marker + headers
        # so @forbidden_error can turn a throttle-403 into a real 429 + Retry-After.
        # auth_request only forwards 401/403, so /validate signals throttles as 403.
        auth_request_set $rl_throttled $upstream_http_x_ratelimit_throttled;
        auth_request_set $rl_limit $upstream_http_x_ratelimit_limit;
        auth_request_set $rl_reset $upstream_http_x_ratelimit_reset;
        auth_request_set $rl_retry $upstream_http_retry_after;
{proxy_directive}
        proxy_http_version 1.1;
        proxy_ssl_server_name on;
        proxy_set_header Host {host_header};
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Add original URL for auth server scope validation
        proxy_set_header X-Original-URL $scheme://$host$request_uri;

        # Pass through the original authentication headers
        proxy_set_header Authorization $http_authorization;
        proxy_set_header X-Authorization $http_x_authorization;
        proxy_set_header X-User-Pool-Id $http_x_user_pool_id;
        proxy_set_header X-Client-Id $http_x_client_id;
        proxy_set_header X-Region $http_x_region;

        # Forward MCP session ID for streamable-http transport
        proxy_set_header Mcp-Session-Id $http_mcp_session_id;

        # Forward auth server response headers to backend
        proxy_set_header X-User $auth_user;
        proxy_set_header X-Username $auth_username;
        proxy_set_header X-Client-Id-Auth $auth_client_id;
        proxy_set_header X-Scopes $auth_scopes;
        proxy_set_header X-Auth-Method $auth_method;
        proxy_set_header X-Server-Name $auth_server_name;
        proxy_set_header X-Tool-Name $auth_tool_name;
        # The internal JWT mcp_proxy verifies (it ignores the X-* headers above).
        proxy_set_header X-Internal-Token $auth_internal_token;

        # Pass all original client headers
        proxy_pass_request_headers on;

        # Handle auth errors
        error_page 401 = @auth_error;
        error_page 403 = @forbidden_error;{obo_resource_metadata}{version_headers}"""

        # Transport-specific settings
        if transport_type == "sse":
            transport_settings = """
        # Capture request body for auth validation using Lua
        rewrite_by_lua_file /etc/nginx/lua/capture_body.lua;
        log_by_lua_file /etc/nginx/lua/emit_metrics.lua;

        # For SSE connections and WebSocket upgrades
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection $http_connection;
        proxy_set_header Upgrade $http_upgrade;
        # Explicitly preserve Accept header for MCP protocol requirements
        proxy_set_header Accept $http_accept;
        chunked_transfer_encoding off;"""

        elif transport_type == "streamable-http":
            transport_settings = """
        # Capture request body for auth validation using Lua
        rewrite_by_lua_file /etc/nginx/lua/capture_body.lua;
        log_by_lua_file /etc/nginx/lua/emit_metrics.lua;

        # HTTP transport configuration
        proxy_buffering off;
        proxy_set_header Connection "";
        # Explicitly preserve Accept header for MCP protocol requirements
        proxy_set_header Accept $http_accept;"""

        else:  # direct
            transport_settings = """
        # Capture request body for auth validation using Lua
        rewrite_by_lua_file /etc/nginx/lua/capture_body.lua;
        log_by_lua_file /etc/nginx/lua/emit_metrics.lua;

        # Generic transport configuration
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection $http_connection;
        proxy_set_header Upgrade $http_upgrade;
        chunked_transfer_encoding off;"""

        # Always normalise the location path to end with a trailing slash (issue #1501).
        # nginx treats `location /a` as a prefix match against ANY URL starting with
        # `/a` (e.g. /api/, /auth, /about), so a server registered at a short path
        # would silently hijack unrelated browser/API routes and subject them to
        # auth_request /validate. `location /a/` only matches /a/ and paths that
        # literally continue past the slash, so `/api/...` no longer matches. MCP
        # clients already call `/server-name/mcp` (or /sse), which continue to match.
        # A bare `GET /a` (no trailing slash) no longer matches this block; nginx does
        # NOT auto-redirect it to `/a/` for a proxy_pass location, so it falls through
        # to the catch-all. This is fine because the discovery/connect URLs always
        # include the `/mcp` (or `/sse`) suffix -- no client connects at the bare path.
        location_path = path.rstrip("/") + "/"
        logger.info(f"Creating location block for {location_path} with {transport_type} transport")

        return f"""
    location {{{{ROOT_PATH}}}}{location_path} {{{transport_settings}{common_settings}
    }}"""


# Global nginx service instance
nginx_service = NginxConfigService()


class NginxReloadScheduler:
    """Coalesces multiple nginx reload requests into periodic batched reloads.

    Instead of reloading nginx on every server registration, callers invoke
    mark_dirty() which sets a boolean flag. A background task wakes every
    debounce_seconds, checks if the flag is set (or polls the DB for external
    changes in multi-replica deployments), regenerates the config if the
    rendered output differs from the last-applied version, and reloads nginx
    once.

    See issue #1087 and .scratchpad/lld-nginx-debounced-reload.md.
    """

    def __init__(
        self,
        debounce_seconds: float = 2.0,
        poll_external: bool = True,
    ) -> None:
        self._dirty: bool = False
        self._debounce_seconds = debounce_seconds
        self._poll_external = poll_external
        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._last_config_hash: str = ""
        self._flush_lock: asyncio.Lock = asyncio.Lock()

    def mark_dirty(self) -> None:
        """Signal that nginx config needs regeneration. Non-blocking."""
        self._dirty = True
        nginx_service._cached_server_names = None

    def seed_hash(self, config_text: str) -> None:
        """Set the initial config hash after startup generation.

        Prevents a redundant reload on the first scheduler tick.
        """
        self._last_config_hash = hashlib.sha256(config_text.encode()).hexdigest()

    async def start(self) -> None:
        """Start the background flush loop. Call once at app startup."""
        self._task = asyncio.create_task(self._flush_loop())
        logger.info(
            "NginxReloadScheduler started (debounce=%.1fs, poll_external=%s)",
            self._debounce_seconds,
            self._poll_external,
        )

    async def stop(self) -> None:
        """Gracefully stop the flush loop. Performs one final flush if dirty."""
        self._stop_event.set()
        if self._task:
            await self._task

    async def flush_now(self) -> None:
        """Force an immediate regen+reload. Used for toggle/delete where the
        change must be reflected before the HTTP response returns."""
        await self._do_reload_if_changed()

    async def _flush_loop(self) -> None:
        while not self._stop_event.is_set():
            await asyncio.sleep(self._debounce_seconds)
            if self._dirty or self._poll_external:
                await self._do_reload_if_changed()

        if self._dirty:
            await self._do_reload_if_changed()

    async def _do_reload_if_changed(self) -> None:
        async with self._flush_lock:
            self._dirty = False
            try:
                enabled_servers = await _fetch_all_enabled_servers()
                config_text = await nginx_service.render_config(enabled_servers)
                if config_text is None:
                    return

                new_hash = hashlib.sha256(config_text.encode()).hexdigest()
                if new_hash == self._last_config_hash:
                    return

                # Config changed: write virtual server Lua mappings, then nginx config
                await nginx_service._commit_virtual_server_mappings()

                async with nginx_service.reload_lock:
                    # Validate the candidate in-tree before it can persist as
                    # the live config; a rejected render restores last-good and
                    # raises, so the cold-start path can never load a bad file.
                    await asyncio.to_thread(
                        _write_and_validate_config, settings.nginx_config_path, config_text
                    )
                    reloaded = await asyncio.to_thread(nginx_service.reload_nginx)
                if reloaded:
                    self._last_config_hash = new_hash
                    logger.info(
                        "Debounced nginx reload completed (hash=%s)",
                        new_hash[:12],
                    )
                else:
                    self._dirty = True
            except Exception as e:
                logger.error("Debounced nginx reload failed: %s", e)
                self._dirty = True


async def _fetch_all_enabled_servers() -> dict[str, Any]:
    """Fetch all enabled servers from the DB for nginx config generation."""
    from registry.services.server_service import server_service

    enabled_servers: dict[str, Any] = {}
    for path in await server_service.get_enabled_services():
        info = await server_service.get_server_info(path)
        if info:
            enabled_servers[path] = info
    return enabled_servers


# Module-level singleton
nginx_reload_scheduler = NginxReloadScheduler(
    debounce_seconds=float(os.getenv("NGINX_RELOAD_DEBOUNCE_SECONDS", "5.0")),
    poll_external=os.getenv("NGINX_RELOAD_POLL_EXTERNAL", "true").lower()
    in ("1", "true", "yes", "on"),
)
