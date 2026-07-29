import os
from typing import Optional

# Minimum accepted length (characters, on the stripped value) for the API-key
# HMAC pepper. A short pepper is brute-forceable and defeats the point of
# per-deployment domain separation.
_MIN_PEPPER_LENGTH: int = 32

# Known-weak / placeholder pepper values that must never be accepted even if an
# operator explicitly sets them (e.g. a compose file passing ${VAR:-default}).
# The historical hard-coded constant is included so an accidental copy is
# rejected rather than silently reused.
_WEAK_PEPPER_VALUES: frozenset[str] = frozenset(
    {
        "mcp-gateway-metrics-api-key-v1",
        "changeme",
        "change-me",
        "changethis",
        "change-this",
        "development-secret-key",
        "secret",
        "password",
        "placeholder",
        "example",
        "test",
    }
)

# Case-insensitive prefixes that mark an obvious placeholder the operator was
# meant to replace (e.g. the .env.example default). Rejected even though they
# may satisfy the length check, so a copied-but-unedited example fails closed.
_WEAK_PEPPER_PREFIXES: tuple[str, ...] = (
    "change-me",
    "changeme",
    "change-this",
    "changethis",
    "your-",
    "example-",
    "placeholder",
)

# Case-insensitive markers that indicate a placeholder even when it does NOT sit
# at the start of the value -- an operator who prepended or embedded the example
# text (e.g. "internal-change-me-generate-...") would slip past the prefix check
# above. Matched as substrings anywhere in the normalized value. Kept narrow to
# avoid false-positives on genuinely random high-entropy keys.
_WEAK_PEPPER_MARKERS: tuple[str, ...] = (
    "change-me",
    "changeme",
    "change-this",
    "changethis",
    "replace-me",
    "replace_me",
    "replaceme",
    "generate-with-openssl",
)


# Minimum accepted length (characters, on the stripped value) for the admin API
# key. The admin key guards destructive/privileged operations (retention policy
# changes, cleanup, database stats), so a short/guessable value is unacceptable.
# It uses the SAME floor as the pepper: both are secrets that gate privileged
# behavior and must not accept a 16-char placeholder like "changemechangeme".
_MIN_ADMIN_KEY_LENGTH: int = 32


def _reject_weak_secret(
    raw_value: str | None,
    name: str,
    min_len: int,
) -> str:
    """Validate a required secret, failing closed on missing/weak/short values.

    Shared secret hardening for every high-entropy secret this service loads
    (the API-key HMAC pepper and the admin API key). A single helper keeps the
    known-weak literal / placeholder-prefix / embedded-marker denylist in ONE
    place, so both callers reject the same set of guessable values and a copied
    ``.env.example`` cannot slip past one validator but not the other.

    The weak-value check runs BEFORE the length check so a known placeholder
    produces the most actionable error message, matching the invariant
    "weak-check before length" from the security guidelines.

    Args:
        raw_value: The raw environment value, or None if unset.
        name: The environment variable name, used in error messages.
        min_len: Minimum accepted length of the stripped value.

    Returns:
        The normalized (stripped) secret string.

    Raises:
        ValueError: If the value is unset, empty/whitespace, a known-weak
            literal/placeholder, or shorter than ``min_len``. Denies by default.
    """
    if raw_value is None:
        raise ValueError(
            f"{name} is required but not set. Set it to a unique, high-entropy "
            "secret (e.g. `openssl rand -hex 32`). Denies by default until it "
            "is configured."
        )

    value = raw_value.strip()

    if not value:
        raise ValueError(
            f"{name} is set but empty/whitespace. Set it to a unique, "
            "high-entropy secret (e.g. `openssl rand -hex 32`)."
        )

    # Reject exact known-weak literals, known placeholder prefixes, and
    # placeholder markers appearing anywhere in the value (so editing the middle
    # of the example does not slip past a start-only check).
    normalized = value.lower()
    if (
        normalized in _WEAK_PEPPER_VALUES
        or normalized.startswith(_WEAK_PEPPER_PREFIXES)
        or any(marker in normalized for marker in _WEAK_PEPPER_MARKERS)
    ):
        raise ValueError(
            f"{name} is set to a known-weak/placeholder value. Set it to a "
            "unique, high-entropy secret (e.g. `openssl rand -hex 32`)."
        )

    if len(value) < min_len:
        raise ValueError(
            f"{name} must be at least {min_len} characters (got {len(value)}). "
            "Use a high-entropy value such as `openssl rand -hex 32`."
        )

    return value


def _validate_admin_api_key(
    raw_value: str | None,
) -> str:
    """Validate the admin API key, failing closed on missing/weak values.

    Admin operations under ``/admin/*`` (retention policy changes, cleanup,
    database stats/size) must NOT be reachable with an ordinary ingest key. A
    distinct, explicitly-configured admin key provides that privilege
    separation. Because it is the only credential standing between any valid
    ingest client and destructive operations, it must be present and strong.

    Applies the SAME weak-value / placeholder / length rejection as the pepper
    (via :func:`_reject_weak_secret`), so an unset, empty, whitespace-only,
    known-weak, or too-short value raises and ``/admin/*`` denies by default
    rather than silently accepting an ingest key or a placeholder such as
    ``changemechangeme``.

    Args:
        raw_value: The raw ``METRICS_ADMIN_API_KEY`` environment value, or None.

    Returns:
        The normalized (stripped) admin key string.

    Raises:
        ValueError: If the admin key is unset, empty/whitespace, known-weak, or
            shorter than the minimum length. Denies by default.
    """
    return _reject_weak_secret(
        raw_value,
        "METRICS_ADMIN_API_KEY",
        _MIN_ADMIN_KEY_LENGTH,
    )


def _validate_pepper(
    raw_value: str | None,
) -> str:
    """Validate the API-key HMAC pepper, failing closed on missing/weak values.

    The pepper provides per-deployment domain separation for stored API-key
    hashes: it keeps hashes deterministic (so the UNIQUE ``key_hash`` lookup
    still works) while defeating offline / cross-deployment brute force against
    a leaked hash. Because it is a secret, it must be present and strong.

    Args:
        raw_value: The raw ``METRICS_KEY_PEPPER`` environment value, or None.

    Returns:
        The normalized (stripped) pepper string.

    Raises:
        ValueError: If the pepper is unset, empty/whitespace, a known-weak
            literal, or shorter than the minimum length. Denies by default.
    """
    return _reject_weak_secret(
        raw_value,
        "METRICS_KEY_PEPPER",
        _MIN_PEPPER_LENGTH,
    )


class Settings:
    # Database settings
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "/var/lib/sqlite/metrics.db")
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{SQLITE_DB_PATH}")
    METRICS_RETENTION_DAYS: int = int(os.getenv("METRICS_RETENTION_DAYS", "90"))
    DB_CONNECTION_TIMEOUT: int = int(os.getenv("DB_CONNECTION_TIMEOUT", "30"))
    DB_MAX_RETRIES: int = int(os.getenv("DB_MAX_RETRIES", "5"))

    # Service settings
    METRICS_SERVICE_PORT: int = int(os.getenv("METRICS_SERVICE_PORT", "8890"))
    # Service binds to 0.0.0.0 for container/K8s deployment where network isolation
    # is provided by container runtime and ingress controllers.
    METRICS_SERVICE_HOST: str = os.getenv("METRICS_SERVICE_HOST", "0.0.0.0")  # nosec B104 - intentional for containerized deployment

    # OpenTelemetry settings
    OTEL_SERVICE_NAME: str = os.getenv("OTEL_SERVICE_NAME", "mcp-metrics-service")
    OTEL_PROMETHEUS_ENABLED: bool = os.getenv("OTEL_PROMETHEUS_ENABLED", "true").lower() == "true"
    OTEL_PROMETHEUS_PORT: int = int(os.getenv("OTEL_PROMETHEUS_PORT", "9465"))
    OTEL_OTLP_ENDPOINT: str | None = os.getenv("OTEL_OTLP_ENDPOINT")
    OTEL_OTLP_EXPORT_INTERVAL_MS: int = int(os.getenv("OTEL_OTLP_EXPORT_INTERVAL_MS", "30000"))

    # API Security
    METRICS_RATE_LIMIT: int = int(os.getenv("METRICS_RATE_LIMIT", "1000"))
    API_KEY_HASH_ALGORITHM: str = os.getenv("API_KEY_HASH_ALGORITHM", "sha256")

    # Per-caller-IP throttle for the unauthenticated /rate-limit lookup, which
    # would otherwise be a key-validity oracle. Requests beyond this many per
    # window from a single client IP are rejected uniformly.
    RATE_LIMIT_ENDPOINT_MAX_REQUESTS: int = int(os.getenv("RATE_LIMIT_ENDPOINT_MAX_REQUESTS", "10"))
    RATE_LIMIT_ENDPOINT_WINDOW_SECONDS: int = int(
        os.getenv("RATE_LIMIT_ENDPOINT_WINDOW_SECONDS", "60")
    )

    # Histogram bucket boundaries for duration metrics (seconds)
    HISTOGRAM_BUCKET_BOUNDARIES: list = [
        float(x)
        for x in os.getenv(
            "HISTOGRAM_BUCKET_BOUNDARIES",
            "0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0,30.0,60.0,120.0,300.0",
        ).split(",")
    ]

    # Performance
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "100"))
    FLUSH_INTERVAL_SECONDS: int = int(os.getenv("FLUSH_INTERVAL_SECONDS", "30"))
    MAX_REQUEST_SIZE: str = os.getenv("MAX_REQUEST_SIZE", "10MB")

    @staticmethod
    def get_key_pepper() -> str:
        """Return the validated API-key HMAC pepper, failing closed if unusable.

        Read at hash time (not import time) so the fail-closed behavior applies
        at every signing/verification entrypoint, and so importing this module
        for unrelated purposes does not require the secret to be present.

        Returns:
            The normalized pepper string.

        Raises:
            ValueError: Propagated from :func:`_validate_pepper` when the pepper
                is missing, empty, weak, or too short.
        """
        return _validate_pepper(os.getenv("METRICS_KEY_PEPPER"))

    @staticmethod
    def get_admin_api_key() -> str:
        """Return the validated admin API key, failing closed if unusable.

        Read at verification time (not import time) so the fail-closed behavior
        applies at every ``/admin/*`` entrypoint, and so importing this module
        for unrelated purposes does not require the secret to be present.

        Returns:
            The normalized admin key string.

        Raises:
            ValueError: Propagated from :func:`_validate_admin_api_key` when the
                admin key is missing, empty, or too short.
        """
        return _validate_admin_api_key(os.getenv("METRICS_ADMIN_API_KEY"))


settings = Settings()
