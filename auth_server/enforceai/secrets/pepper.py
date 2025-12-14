from __future__ import annotations

from pathlib import Path


def load_api_key_pepper(
    pepper_path: Path,
) -> bytes:
    """Load API key pepper bytes from a file.

    The pepper is treated as an opaque byte sequence and must not be logged.

    Args:
        pepper_path: Path to the pepper file on disk.

    Returns:
        Pepper bytes.

    Raises:
        ValueError: If the file is missing, unreadable, or empty.
    """
    try:
        raw = pepper_path.read_bytes()
    except OSError as exc:
        raise ValueError("API key pepper file is not readable") from exc

    stripped = raw.strip()
    if not stripped:
        raise ValueError("API key pepper file is empty")

    return stripped
