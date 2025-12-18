from __future__ import annotations

import binascii
from pathlib import Path


def load_upstream_kek(
    kek_path: Path,
) -> bytes:
    """Load upstream secret KEK bytes from a local file.

    The KEK is expected to be a hex-encoded 32-byte key (64 hex chars).

    Args:
        kek_path: Path to the KEK file on disk.

    Returns:
        Raw key bytes.

    Raises:
        ValueError: If the file is missing, unreadable, empty, or invalid.
    """
    try:
        raw = kek_path.read_bytes()
    except OSError as exc:
        raise ValueError("Upstream KEK file is not readable") from exc

    stripped = raw.strip()
    if not stripped:
        raise ValueError("Upstream KEK file is empty")

    try:
        decoded = binascii.unhexlify(stripped)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Upstream KEK file must contain hex-encoded bytes") from exc

    if len(decoded) != 32:
        raise ValueError("Upstream KEK must be 32 bytes (64 hex characters)")

    return decoded

