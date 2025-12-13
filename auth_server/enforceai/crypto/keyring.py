from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.rsa import (
    RSAPrivateKey,
    RSAPublicKey,
)


def _load_private_key_pem(
    *,
    private_key_path: Path,
) -> RSAPrivateKey:
    data = private_key_path.read_bytes()
    loaded = serialization.load_pem_private_key(
        data,
        password=None,
    )
    if not isinstance(loaded, RSAPrivateKey):
        raise ValueError("Private key must be an RSA private key")
    return loaded


def _load_public_key_pem(
    *,
    public_key_path: Path,
) -> RSAPublicKey:
    data = public_key_path.read_bytes()
    loaded = serialization.load_pem_public_key(data)
    if not isinstance(loaded, RSAPublicKey):
        raise ValueError("Public key must be an RSA public key")
    return loaded


def _list_kid_public_key_paths(
    *,
    public_keys_dir: Path,
) -> Mapping[str, Path]:
    if not public_keys_dir.exists():
        raise FileNotFoundError(f"Public keys dir does not exist: {public_keys_dir}")
    if not public_keys_dir.is_dir():
        raise NotADirectoryError(f"Public keys path is not a directory: {public_keys_dir}")

    kids: dict[str, Path] = {}
    for path in sorted(public_keys_dir.iterdir(), key=lambda p: p.name):
        if not path.is_file():
            continue
        if path.suffix != ".pem":
            continue
        kid = path.stem
        if not kid:
            continue
        kids[kid] = path

    if not kids:
        raise FileNotFoundError(
            f"No public key PEM files found in {public_keys_dir} (expected <kid>.pem)"
        )

    return kids


@dataclass(frozen=True)
class GatewayKeyring:
    private_key_path: Path
    public_keys_dir: Path
    active_kid: str
    _private_key: RSAPrivateKey
    _public_keys: Mapping[str, RSAPublicKey]

    @classmethod
    def load(
        cls,
        *,
        private_key_path: Path,
        public_keys_dir: Path,
        active_kid: str,
    ) -> "GatewayKeyring":
        if not private_key_path.exists():
            raise FileNotFoundError(f"Private key file does not exist: {private_key_path}")
        if not private_key_path.is_file():
            raise FileNotFoundError(f"Private key path is not a file: {private_key_path}")
        if not active_kid.strip():
            raise ValueError("active_kid must be a non-empty string")

        kid_paths = _list_kid_public_key_paths(public_keys_dir=public_keys_dir)
        if active_kid not in kid_paths:
            raise FileNotFoundError(
                "Active kid public key not found: "
                f"{(public_keys_dir / f'{active_kid}.pem')}"
            )

        private_key = _load_private_key_pem(private_key_path=private_key_path)
        public_keys: dict[str, RSAPublicKey] = {}
        for kid, path in kid_paths.items():
            public_keys[kid] = _load_public_key_pem(public_key_path=path)

        return cls(
            private_key_path=private_key_path,
            public_keys_dir=public_keys_dir,
            active_kid=active_kid,
            _private_key=private_key,
            _public_keys=public_keys,
        )

    def get_public_key(
        self,
        *,
        kid: str,
    ) -> Optional[RSAPublicKey]:
        return self._public_keys.get(kid)

    @property
    def signing_private_key(self) -> RSAPrivateKey:
        return self._private_key


@lru_cache(maxsize=8)
def load_gateway_keyring_cached(
    *,
    private_key_path: Path,
    public_keys_dir: Path,
    active_kid: str,
) -> GatewayKeyring:
    return GatewayKeyring.load(
        private_key_path=private_key_path,
        public_keys_dir=public_keys_dir,
        active_kid=active_kid,
    )

