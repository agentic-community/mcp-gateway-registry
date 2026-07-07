"""Tests for the create_api_key CLI helper.

Focus: the generated API key is persisted to an owner-only (0600) file and is
never emitted to stdout in clear text (only a masked fingerprint is shown).
"""

import os
import stat
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# create_api_key.py lives at the metrics-service root, next to app/.
sys.path.insert(0, str(Path(__file__).parent.parent))

import create_api_key  # noqa: E402


@pytest.mark.asyncio
async def test_api_key_file_is_owner_only_and_key_not_printed(tmp_path, capsys):
    """The full API key lands in a 0600 file and never in stdout."""
    service_name = "unit-test-service"
    key_file = tmp_path / f".api_key_{service_name}.txt"

    # Run from tmp_path so the relative key file is created there.
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        with (
            patch.object(create_api_key, "init_database", new=AsyncMock()),
            patch.object(create_api_key, "MetricsStorage") as mock_storage_cls,
        ):
            mock_storage = mock_storage_cls.return_value
            mock_storage.create_api_key = AsyncMock(return_value=True)

            api_key = await create_api_key.create_api_key_for_service(service_name)
    finally:
        os.chdir(original_cwd)

    assert api_key is not None
    assert key_file.exists(), "key file should be written"

    # File must be owner-read/write only (0600) -- no group/other bits.
    mode = stat.S_IMODE(key_file.stat().st_mode)
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"

    # The file holds the real key; stdout must NOT contain the full key.
    file_contents = key_file.read_text()
    assert api_key in file_contents

    captured = capsys.readouterr()
    assert api_key not in captured.out, "full API key must not be printed to stdout"


@pytest.mark.asyncio
async def test_no_file_written_on_storage_failure(tmp_path, capsys):
    """When the DB insert fails, no key file is created."""
    service_name = "unit-test-fail"
    key_file = tmp_path / f".api_key_{service_name}.txt"

    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        with (
            patch.object(create_api_key, "init_database", new=AsyncMock()),
            patch.object(create_api_key, "MetricsStorage") as mock_storage_cls,
        ):
            mock_storage = mock_storage_cls.return_value
            mock_storage.create_api_key = AsyncMock(return_value=False)

            api_key = await create_api_key.create_api_key_for_service(service_name)
    finally:
        os.chdir(original_cwd)

    assert api_key is None
    assert not key_file.exists()
