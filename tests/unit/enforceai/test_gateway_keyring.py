"""
Unit tests for EnforceAI gateway keyring loader (Stage 2.2).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from auth_server.enforceai.crypto.keyring import (
    GatewayKeyring,
    load_gateway_keyring_cached,
)


@pytest.mark.unit
class TestGatewayKeyring:
    def test_loads_keys_from_fixture(
        self,
        enforceai_gateway_key_files,
    ) -> None:
        key_files = enforceai_gateway_key_files
        ring = GatewayKeyring.load(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )
        assert ring.active_kid == key_files.active_kid
        assert ring.signing_private_key is not None
        assert ring.get_public_key(kid=key_files.active_kid) is not None

    def test_missing_active_kid_fails(
        self,
        enforceai_gateway_key_files,
        tmp_path: Path,
    ) -> None:
        key_files = enforceai_gateway_key_files
        other_dir = tmp_path / "public"
        other_dir.mkdir(parents=True)
        (other_dir / "different-kid.pem").write_bytes(
            (key_files.public_keys_dir / f"{key_files.active_kid}.pem").read_bytes()
        )

        with pytest.raises(FileNotFoundError, match="Active kid public key not found"):
            GatewayKeyring.load(
                private_key_path=key_files.private_key_path,
                public_keys_dir=other_dir,
                active_kid=key_files.active_kid,
            )

    def test_invalid_pem_fails(
        self,
        tmp_path: Path,
    ) -> None:
        private_key_path = tmp_path / "private.pem"
        private_key_path.write_text("not a pem", encoding="utf-8")

        public_dir = tmp_path / "public"
        public_dir.mkdir()
        (public_dir / "kid-1.pem").write_text("not a pem", encoding="utf-8")

        with pytest.raises(ValueError):
            GatewayKeyring.load(
                private_key_path=private_key_path,
                public_keys_dir=public_dir,
                active_kid="kid-1",
            )

    def test_cached_loader_does_not_reread_files(
        self,
        enforceai_gateway_key_files,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        key_files = enforceai_gateway_key_files
        load_gateway_keyring_cached.cache_clear()

        read_calls: list[Path] = []
        original_read_bytes = Path.read_bytes

        def _patched_read_bytes(self: Path) -> bytes:
            read_calls.append(self)
            return original_read_bytes(self)

        monkeypatch.setattr(Path, "read_bytes", _patched_read_bytes)

        load_gateway_keyring_cached(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )
        first_reads = len(read_calls)
        assert first_reads > 0

        load_gateway_keyring_cached(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )
        assert len(read_calls) == first_reads

