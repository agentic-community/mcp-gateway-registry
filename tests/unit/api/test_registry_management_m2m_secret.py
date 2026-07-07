"""Tests for M2M account creation secret handling in the management CLI.

Focus: the newly minted client secret is written to an owner-only (0600) file
and is never emitted to stdout in clear text (not even a prefix/suffix).
"""

import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# registry_management.py lives under api/ and imports sibling modules by name.
_API_DIR = Path(__file__).resolve().parents[3] / "api"
sys.path.insert(0, str(_API_DIR))

import registry_management  # noqa: E402


def _run_create_m2m(tmp_path, client_secret: str):
    """Invoke cmd_user_create_m2m with a mocked client from tmp_path cwd."""
    args = SimpleNamespace(
        groups="group-a,group-b",
        name="svc-account",
        description=None,
    )

    result = SimpleNamespace(
        client_id="client-abc-123",
        client_secret=client_secret,
        groups=["group-a", "group-b"],
        service_principal_id=None,
    )

    mock_client = MagicMock()
    mock_client.create_m2m_account.return_value = result

    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        with patch.object(registry_management, "_create_client", return_value=mock_client):
            code = registry_management.cmd_user_create_m2m(args)
    finally:
        os.chdir(original_cwd)
    return code, result


def test_m2m_secret_written_to_owner_only_file(tmp_path, capsys):
    """The client secret lands in a 0600 file, never in stdout."""
    secret = "super-secret-client-value-abcdef123456"
    code, result = _run_create_m2m(tmp_path, secret)

    assert code == 0

    secret_file = tmp_path / f".m2m_client_secret_{result.client_id}.txt"
    assert secret_file.exists(), "secret file should be written"

    mode = stat.S_IMODE(secret_file.stat().st_mode)
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"

    contents = secret_file.read_text()
    assert secret in contents

    captured = capsys.readouterr()
    # Neither the full secret nor a prefix/suffix fingerprint may appear.
    assert secret not in captured.out
    assert secret[:8] not in captured.out
    assert secret[-4:] not in captured.out
    # The client id (non-sensitive) is still fine to show.
    assert result.client_id in captured.out
