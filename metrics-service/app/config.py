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
    if raw_value is None:
        raise ValueError(
            "METRICS_KEY_PEPPER is required but not set. Set it to a unique, "
            "high-entropy per-deployment secret (e.g. `openssl rand -hex 32`). "
            "The metrics service will not start without it."
        )

    pepper = raw_value.strip()

    if not pepper:
        raise ValueError(
            "METRICS_KEY_PEPPER is set but empty/whitespace. Set it to a "
            "unique, high-entropy per-deployment secret (e.g. "
            "`openssl rand -hex 32`)."
        )

    # Run the weak-value check BEFORE the length check so a known placeholder
    # produces the most actionable error message. Reject exact known-weak
    # literals, known placeholder prefixes, and placeholder markers appearing
    # anywhere in the value (so editing the middle of the example does not slip
    # past a start-only check).
    normalized = pepper.lower()
    if (
        normalized in _WEAK_PEPPER_VALUES
        or normalized.startswith(_WEAK_PEPPER_PREFIXES)
        or any(marker in normalized for marker in _WEAK_PEPPER_MARKERS)
    ):
        raise ValueError(
            "METRICS_KEY_PEPPER is set to a known-weak/placeholder value. Set "
            "it to a unique, high-entropy per-deployment secret (e.g. "
            "`openssl rand -hex 32`)."
        )

    if len(pepper) < _MIN_PEPPER_LENGTH:
        raise ValueError(
            f"METRICS_KEY_PEPPER must be at least {_MIN_PEPPER_LENGTH} "
            f"characters (got {len(pepper)}). Use a high-entropy value such "
            "as `openssl rand -hex 32`."
        )

    return pepper


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
    OTEL_OTLP_ENDPOINT: Optional[str] = os.getenv("OTEL_OTLP_ENDPOINT")
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


settings = Settings()
