"""Shared validation for application signing secrets.

The ``SECRET_KEY`` signs internal JWTs and session cookies and is used to derive
credential-encryption keys. A missing, short, or well-known value lets an
attacker forge tokens and decrypt stored credentials, so validation must be
consistent across every process that reads the key (the auth server and the
registry) and must fail closed at startup.

Other secrets share the same threat profile. The ``AUTH_SERVER_NGINX_MARKER_SECRET``
(nginx force-sets it as ``X-Validate-Source-Secret`` on the ``/validate``
subrequest so the auth server only mints mcp-proxy tokens for requests that
actually came through nginx) is a signing/marker secret: a short or well-known
value is brute-forceable and lets a direct :8888 ``/validate`` call forge the
marker. To avoid the drift that appears when each secret grows its own bespoke
check, the min-length and weak-value logic lives in one core validator
(``validate_signing_secret``) that every named signing secret runs through.
"""

MIN_SECRET_KEY_LENGTH: int = 32

# Well-known placeholder values that have historically shipped as defaults or
# been suggested in documentation. Matched case-insensitively (after stripping
# surrounding whitespace) and must never be accepted.
_WEAK_SECRET_KEYS: frozenset[str] = frozenset(
    {
        "development-secret-key",
        "changeme",
        "change-me",
        "change-this",
        "secret",
        "secret-key",
        "your-secret-key",
        "your-secret-key-here",
        "test-secret-key",
        # The literal value historically shipped in .env.example. It is long
        # enough to pass the length check, so it must be rejected explicitly.
        "change-this-immediately-use-a-strong-random-key-in-production",
    }
)

# Substrings that indicate a placeholder even when the operator appended or
# edited the surrounding text (e.g. copied the .env.example value). Matched
# case-insensitively against the whole key so a key that merely *contains* a
# "change this" style marker is still rejected. These are deliberately narrow
# to avoid false-positives on genuinely random keys.
_WEAK_SECRET_KEY_MARKERS: tuple[str, ...] = (
    "change-this-immediately",
    "change-me-immediately",
    "replace-me",
    "replace_me",
    "changemeimmediately",
)


def validate_signing_secret(
    secret_value: str | None,
    secret_name: str,
    required: bool = True,
) -> str:
    """Validate a signing/marker secret, failing closed on weak values.

    This is the single hardened check behind every signing secret in the
    codebase. A non-empty value must be at least ``MIN_SECRET_KEY_LENGTH``
    characters long and not one of the known-weak placeholder literals. The
    weak-value check runs before the length check so a known placeholder
    produces the correct error even when it is long enough to pass the length
    check.

    Args:
        secret_value: The candidate secret, typically read from an environment
            variable. ``None`` or whitespace-only means unset.
        secret_name: The environment-variable name used in error messages so an
            operator can tell which secret failed (e.g. ``"SECRET_KEY"`` or
            ``"AUTH_SERVER_NGINX_MARKER_SECRET"``).
        required: When ``True`` (default), an unset secret is rejected. When
            ``False`` (the secret is optional in this deployment mode), an unset
            secret returns an empty string, but a *present* value is still held
            to the full min-length and weak-value bar. This lets a caller wire
            in an optional secret without ever silently accepting a short or
            well-known value for it.

    Returns:
        The validated secret with surrounding whitespace stripped, or an empty
        string when the secret is unset and ``required`` is ``False``. Stripping
        is applied consistently so that two replicas whose secret differs only
        by accidental leading/trailing whitespace still derive the same value
        instead of failing every cross-replica check.

    Raises:
        RuntimeError: If the secret is required but missing, or (whether
            required or optional) is present but shorter than
            ``MIN_SECRET_KEY_LENGTH`` characters or a known-weak literal.
    """
    remediation = (
        f"Set {secret_name} to a random value at least "
        f"{MIN_SECRET_KEY_LENGTH} characters long, identical across all "
        "auth_server and registry replicas (see your deployment's secret "
        "configuration, e.g. chart values.yaml global.secretKey)."
    )

    if not secret_value or not secret_value.strip():
        if not required:
            # Optional and unset: return empty so the caller can treat the
            # feature as disabled. A short/weak *present* value is never
            # silently accepted -- that path falls through to the checks below.
            return ""
        raise RuntimeError(f"{secret_name} environment variable is required. {remediation}")

    # Reject known placeholders before the length check: some placeholders are
    # shorter than the minimum and some are longer, so checking length first
    # would give a misleading "too short" message for a known-weak literal.
    stripped = secret_value.strip()
    normalized = stripped.lower()
    if normalized in _WEAK_SECRET_KEYS or any(
        marker in normalized for marker in _WEAK_SECRET_KEY_MARKERS
    ):
        raise RuntimeError(
            f"{secret_name} is set to a well-known placeholder value and cannot be used. "
            f"{remediation}"
        )

    # Measure the stripped length so a whitespace-padded short value (e.g.
    # "   short   ") cannot pass the length check on padding alone.
    if len(stripped) < MIN_SECRET_KEY_LENGTH:
        raise RuntimeError(
            f"{secret_name} is too short "
            f"({len(stripped)} characters); it must be at least "
            f"{MIN_SECRET_KEY_LENGTH} characters. {remediation}"
        )

    return stripped


def validate_secret_key(
    secret_key: str | None,
) -> str:
    """Validate the application SECRET_KEY, failing closed on weak values.

    A valid key must be present, at least ``MIN_SECRET_KEY_LENGTH`` characters
    long, and not one of the known-weak placeholder literals. This is enforced
    identically in the auth server and the registry so that neither process can
    start with a forgeable signing key. The actual checks live in
    :func:`validate_signing_secret`; this wrapper pins the ``SECRET_KEY`` name
    and its always-required semantics.

    Args:
        secret_key: The candidate key, typically read from the ``SECRET_KEY``
            environment variable. ``None`` or empty means unset.

    Returns:
        The validated key with surrounding whitespace stripped. Stripping is
        applied consistently so that two replicas whose ``SECRET_KEY`` differs
        only by accidental leading/trailing whitespace still derive the same
        signing key instead of failing every cross-replica signature check.

    Raises:
        RuntimeError: If the key is missing, shorter than
            ``MIN_SECRET_KEY_LENGTH`` characters, or a known-weak literal.
    """
    return validate_signing_secret(secret_key, "SECRET_KEY", required=True)
