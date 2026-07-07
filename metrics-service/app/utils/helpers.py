import hashlib
import hmac
import secrets

from ..config import settings


def generate_api_key() -> str:
    """Generate a new API key."""
    return f"mcp_metrics_{secrets.token_urlsafe(32)}"


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage/lookup using a peppered HMAC-SHA256.

    The HMAC key is a required, per-deployment secret (``METRICS_KEY_PEPPER``)
    rather than a hard-coded constant. Using a deployment-scoped secret keeps
    the hash deterministic — so the UNIQUE ``key_hash`` column and lookup-by-hash
    still work — while ensuring a leaked hash cannot be brute-forced offline
    without also knowing that deployment's secret, and that hashes are not
    portable across deployments.

    Args:
        api_key: The plaintext API key to hash.

    Returns:
        Hex-encoded HMAC-SHA256 digest of the key under the deployment pepper.

    Raises:
        ValueError: If ``METRICS_KEY_PEPPER`` is unset, empty, weak, or too
            short. The service fails closed rather than hashing under a
            predictable key.
    """
    pepper = settings.get_key_pepper()
    return hmac.new(pepper.encode(), api_key.encode(), hashlib.sha256).hexdigest()


def hashes_equal(hash_a: str, hash_b: str) -> bool:
    """Constant-time comparison of two hex-encoded key hashes.

    Args:
        hash_a: First hex hash.
        hash_b: Second hex hash.

    Returns:
        True iff the two hashes are equal, compared without early exit so the
        comparison time does not leak how many leading characters matched.
    """
    return hmac.compare_digest(hash_a, hash_b)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"req_{secrets.token_hex(8)}"
