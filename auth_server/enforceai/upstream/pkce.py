from __future__ import annotations

import base64
import hashlib
import secrets


def generate_code_verifier(
    *,
    bytes_length: int = 32,
) -> str:
    """Generate a PKCE code verifier suitable for S256.

    Args:
        bytes_length: Entropy source length in bytes (default 32).

    Returns:
        URL-safe code verifier string (43+ chars).
    """
    if bytes_length < 16:
        raise ValueError("bytes_length must be >= 16")
    return secrets.token_urlsafe(bytes_length)


def compute_code_challenge(
    *,
    code_verifier: str,
) -> str:
    verifier = code_verifier.strip()
    if not verifier:
        raise ValueError("code_verifier must be non-empty")
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

