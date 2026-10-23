"""Unit tests for registry_cli_wrapper's token-file loading.

The registry UI downloads `.token` with the access token nested under
`tokens`, while M2M token files keep it at the top level. The wrapper must
accept both, or `--token-file .token` fails before any command runs.
"""

from __future__ import annotations

import json

import pytest

import cli.registry_cli_wrapper as wrapper


class TestLoadTokenFromFile:
    """Tests for _load_token_from_file."""

    @pytest.mark.parametrize(
        "payload",
        [
            {"access_token": "the-token"},
            {"tokens": {"access_token": "the-token"}},
        ],
        ids=["top-level", "nested"],
    )
    def test_reads_both_token_file_layouts(self, tmp_path, payload):
        """Top-level and nested access tokens are both accepted."""
        path = tmp_path / "token.json"
        path.write_text(json.dumps(payload))

        assert wrapper._load_token_from_file(str(path)) == "the-token"

    def test_rejects_a_file_without_a_token(self, tmp_path):
        """A token file with no access token raises a clear error."""
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"tokens": {}}))

        with pytest.raises(ValueError, match="No access_token found"):
            wrapper._load_token_from_file(str(path))
