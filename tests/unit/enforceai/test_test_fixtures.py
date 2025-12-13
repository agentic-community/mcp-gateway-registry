"""
Unit tests for EnforceAI pytest fixtures (Stage 0.3).
"""

import sqlite3
from pathlib import Path

import pytest

from auth_server.enforceai.config import EnforceAISettings


@pytest.mark.unit
class TestEnforceAITestFixtures:
    """Test suite ensuring EnforceAI fixtures behave as expected."""

    def test_gateway_key_files_are_written(
        self,
        enforceai_gateway_key_files,
        caplog: pytest.LogCaptureFixture,
        capsys: pytest.CaptureFixture[str],
    ):
        key_files = enforceai_gateway_key_files

        assert key_files.private_key_path.exists()
        assert key_files.public_keys_dir.exists()
        assert (key_files.public_keys_dir / f"{key_files.active_kid}.pem").exists()

        private_contents = key_files.private_key_path.read_text()
        assert "BEGIN PRIVATE KEY" in private_contents

        public_contents = (key_files.public_keys_dir / f"{key_files.active_kid}.pem").read_text()
        assert "BEGIN PUBLIC KEY" in public_contents

        out = capsys.readouterr()
        assert "BEGIN PRIVATE KEY" not in out.out
        assert "BEGIN PRIVATE KEY" not in out.err
        assert "BEGIN PRIVATE KEY" not in caplog.text

    def test_sqlite_db_fixture_creates_file(
        self,
        enforceai_sqlite_db_path: Path,
    ):
        assert enforceai_sqlite_db_path.exists()

        connection = sqlite3.connect(enforceai_sqlite_db_path)
        connection.execute("SELECT 1")
        connection.close()

    def test_env_helper_supports_settings_construction(
        self,
        enforceai_env,
        enforceai_oidc_issuers_env_json: str,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
    ):
        key_files = enforceai_gateway_key_files
        enforceai_env(
            {
                "OIDC_ISSUERS": enforceai_oidc_issuers_env_json,
                "ENFORCEAI_DB_PATH": str(enforceai_sqlite_db_path),
                "ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH": str(key_files.private_key_path),
                "ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR": str(key_files.public_keys_dir),
                "GATEWAY_ACTIVE_KID": key_files.active_kid,
            }
        )

        settings = EnforceAISettings(_env_file=None)
        assert settings.db_path == enforceai_sqlite_db_path
        assert settings.gateway_active_kid == key_files.active_kid

