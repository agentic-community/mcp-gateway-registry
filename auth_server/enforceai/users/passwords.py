from __future__ import annotations

import base64
import hashlib
import hmac
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PasswordHash:
    scheme: str
    encoded: str


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def hash_password(
    password: str,
    *,
    salt: Optional[bytes] = None,
    n: int = 2**14,
    r: int = 8,
    p: int = 1,
    dklen: int = 32,
) -> PasswordHash:
    """Hash a password using stdlib scrypt.

    Format: `scrypt$N$r$p$salt_b64$hash_b64`
    """
    if not password:
        raise ValueError("password must not be empty")

    salt_bytes = os.urandom(16) if salt is None else salt
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt_bytes,
        n=n,
        r=r,
        p=p,
        dklen=dklen,
    )
    encoded = "$".join(
        [
            "scrypt",
            str(n),
            str(r),
            str(p),
            _b64encode(salt_bytes),
            _b64encode(derived),
        ]
    )
    return PasswordHash(
        scheme="scrypt",
        encoded=encoded,
    )


def verify_password(
    password: str,
    password_hash: str,
) -> bool:
    if not password_hash:
        return False

    try:
        scheme, n_raw, r_raw, p_raw, salt_b64, hash_b64 = password_hash.split("$", 5)
    except ValueError:
        return False

    if scheme != "scrypt":
        return False

    try:
        n = int(n_raw)
        r = int(r_raw)
        p = int(p_raw)
    except ValueError:
        return False

    try:
        salt = _b64decode(salt_b64)
        expected = _b64decode(hash_b64)
    except Exception:
        return False

    try:
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=len(expected),
        )
    except Exception:
        return False

    return hmac.compare_digest(derived, expected)

