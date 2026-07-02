"""Shared validation for the application SECRET_KEY.

The SECRET_KEY signs internal JWTs and session cookies and is used to derive
credential-encryption keys. A missing, short, or well-known value lets an
attacker forge tokens and decrypt stored credentials, so validation must be
consistent across every process that reads the key (the auth server and the
registry) and must fail closed at startup.
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


def validate_secret_key(
    secret_key: str | None,
) -> str:
    """Validate the application SECRET_KEY, failing closed on weak values.

    A valid key must be present, at least ``MIN_SECRET_KEY_LENGTH`` characters
    long, and not one of the known-weak placeholder literals. This is enforced
    identically in the auth server and the registry so that neither process can
    start with a forgeable signing key.

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
    remediation = (
        "Set it to a random value at least "
        f"{MIN_SECRET_KEY_LENGTH} characters long, identical across all "
        "auth_server and registry replicas (see chart values.yaml: "
        "global.secretKey)."
    )

    if not secret_key or not secret_key.strip():
        raise RuntimeError(f"SECRET_KEY environment variable is required. {remediation}")

    # Reject known placeholders before the length check: some placeholders are
    # shorter than the minimum and some are longer, so checking length first
    # would give a misleading "too short" message for a known-weak literal.
    stripped = secret_key.strip()
    normalized = stripped.lower()
    if normalized in _WEAK_SECRET_KEYS or any(
        marker in normalized for marker in _WEAK_SECRET_KEY_MARKERS
    ):
        raise RuntimeError(
            f"SECRET_KEY is set to a well-known placeholder value and cannot be used. {remediation}"
        )

    # Measure the stripped length so a whitespace-padded short key (e.g.
    # "   short   ") cannot pass the length check on padding alone.
    if len(stripped) < MIN_SECRET_KEY_LENGTH:
        raise RuntimeError(
            "SECRET_KEY is too short "
            f"({len(stripped)} characters); it must be at least "
            f"{MIN_SECRET_KEY_LENGTH} characters. {remediation}"
        )

    return stripped
