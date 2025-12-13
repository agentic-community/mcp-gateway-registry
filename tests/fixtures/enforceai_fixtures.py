"""
Shared pytest fixtures for EnforceAI stages (RSA keys, temp SQLite, env helpers).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def _generate_rsa_keypair_pem(
    *,
    key_size: int = 2048,
) -> tuple[bytes, bytes]:
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
    )
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def _write_bytes(
    path: Path,
    data: bytes,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_bytes(data)


@dataclass(frozen=True)
class EnforceAIGatewayKeyFiles:
    private_key_path: Path
    public_keys_dir: Path
    active_kid: str


@pytest.fixture
def enforceai_env(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Mapping[str, str]], None]:
    """Set environment variables for tests via a simple callable."""

    def _apply(env: Mapping[str, str]) -> None:
        for key, value in env.items():
            monkeypatch.setenv(key, value)

    return _apply


@pytest.fixture
def enforceai_oidc_issuers_env_json() -> str:
    """Default `OIDC_ISSUERS` JSON value suitable for tests (map-of-one)."""
    return json.dumps(
        {
            "https://issuer.example": {
                "jwks_url": "https://issuer.example/.well-known/jwks.json",
                "audience": "mcp-registry",
            }
        }
    )


@pytest.fixture
def enforceai_sqlite_db_path(
    tmp_path: Path,
) -> Path:
    """Create a temporary SQLite DB file for EnforceAI tests."""
    db_path = tmp_path / "enforceai.db"
    connection = sqlite3.connect(db_path)
    connection.close()
    return db_path


@pytest.fixture
def enforceai_rsa_keypair_pem() -> tuple[bytes, bytes]:
    """Generate an in-memory RSA keypair for RS256 (private_pem, public_pem)."""
    return _generate_rsa_keypair_pem()


@pytest.fixture
def enforceai_gateway_key_files(
    tmp_path: Path,
    enforceai_rsa_keypair_pem: tuple[bytes, bytes],
) -> EnforceAIGatewayKeyFiles:
    """Create a temp key layout matching gateway-token expectations."""
    private_pem, public_pem = enforceai_rsa_keypair_pem
    active_kid = "kid-test-1"

    keys_dir = tmp_path / "keys"
    private_key_path = keys_dir / "private" / "gateway_private.pem"
    public_keys_dir = keys_dir / "public"
    public_key_path = public_keys_dir / f"{active_kid}.pem"

    _write_bytes(private_key_path, private_pem)
    _write_bytes(public_key_path, public_pem)

    return EnforceAIGatewayKeyFiles(
        private_key_path=private_key_path,
        public_keys_dir=public_keys_dir,
        active_kid=active_kid,
    )
