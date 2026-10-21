"""Settings tests for the EMBEDDINGS_EXTRA_HEADER_<N> slots (issue #1819).

Covers how the ten scalar slots collapse into the embeddings_extra_headers map,
every rejection rule enforced in Settings.__init__, and the startup banner that
fires when the headers can never be sent.
"""

import logging

import pytest

from registry.core.config import (
    Settings,
    print_embeddings_extra_headers_inert_banner,
)
from registry.utils.header_validation import validate_header_name

SLOT_COUNT: int = 10


@pytest.fixture(autouse=True)
def _clear_header_env(monkeypatch):
    """Keep slots set in the developer's shell out of these tests."""
    for index in range(1, SLOT_COUNT + 2):
        monkeypatch.delenv(f"EMBEDDINGS_EXTRA_HEADER_{index}", raising=False)
    monkeypatch.delenv("EMBEDDINGS_PROVIDER", raising=False)


def _settings(**slots: str) -> Settings:
    """Build Settings with header slots given as slot_<N>="Name: value"."""
    kwargs = {}
    for key, value in slots.items():
        index = key.removeprefix("slot_")
        kwargs[f"embeddings_extra_header_{index}"] = value
    return Settings(**kwargs)


@pytest.mark.unit
@pytest.mark.core
class TestSlotAssembly:
    """The slots collapse into one header map."""

    def test_no_slots_set_is_empty_map(self):
        assert _settings().embeddings_extra_headers == {}

    def test_single_slot(self):
        s = _settings(slot_1="X-Tenant-Id: research")
        assert s.embeddings_extra_headers == {"X-Tenant-Id": "research"}

    def test_two_slots_keep_original_casing(self):
        s = _settings(slot_1="X-Tenant-Id: research", slot_2="x-route: v2")
        assert s.embeddings_extra_headers == {"X-Tenant-Id": "research", "x-route": "v2"}

    def test_whitespace_around_colon_stripped(self):
        s = _settings(slot_1="  X-Tenant-Id   :   research  ")
        assert s.embeddings_extra_headers == {"X-Tenant-Id": "research"}

    def test_no_space_after_colon(self):
        s = _settings(slot_1="X-Tenant-Id:research")
        assert s.embeddings_extra_headers == {"X-Tenant-Id": "research"}

    def test_blank_and_whitespace_slots_skipped(self):
        s = _settings(slot_1="X-A: 1", slot_2="", slot_3="   ")
        assert s.embeddings_extra_headers == {"X-A": "1"}

    def test_sparse_slots_allowed(self):
        s = _settings(slot_4="X-Four: 4", slot_9="X-Nine: 9")
        assert s.embeddings_extra_headers == {"X-Four": "4", "X-Nine": "9"}

    def test_gap_does_not_truncate(self):
        s = _settings(slot_1="X-One: 1", slot_3="X-Three: 3")
        assert s.embeddings_extra_headers == {"X-One": "1", "X-Three": "3"}

    def test_value_containing_colon_preserved(self):
        s = _settings(slot_1="X-Trace: a:b:c")
        assert s.embeddings_extra_headers == {"X-Trace": "a:b:c"}

    def test_value_with_spaces_allowed(self):
        s = _settings(slot_1="X-Ctx: two words")
        assert s.embeddings_extra_headers == {"X-Ctx": "two words"}

    def test_slot_10_is_read(self):
        s = _settings(slot_10="X-Ten: 10")
        assert s.embeddings_extra_headers == {"X-Ten": "10"}

    def test_slot_11_is_ignored(self, monkeypatch):
        monkeypatch.setenv("EMBEDDINGS_EXTRA_HEADER_11", "X-Eleven: 11")
        assert Settings().embeddings_extra_headers == {}

    def test_slot_read_from_environment(self, monkeypatch):
        monkeypatch.setenv("EMBEDDINGS_EXTRA_HEADER_2", "X-Tenant-Id: research")
        assert Settings().embeddings_extra_headers == {"X-Tenant-Id": "research"}


@pytest.mark.unit
@pytest.mark.core
class TestSlotRejections:
    """A bad slot fails at startup with a message naming the slot."""

    def test_rejects_missing_colon(self):
        with pytest.raises(ValueError, match="EMBEDDINGS_EXTRA_HEADER_1 must be in 'Name: value'"):
            _settings(slot_1="X-Tenant-Id research")

    def test_rejects_crlf_in_value(self):
        with pytest.raises(ValueError, match="control characters"):
            _settings(slot_1="X-A: v\r\nX-Evil: y")

    def test_rejects_control_char_in_value(self):
        with pytest.raises(ValueError, match="control characters"):
            _settings(slot_1="X-A: v\u0000w")

    def test_rejects_space_in_name(self):
        with pytest.raises(ValueError, match="RFC token"):
            _settings(slot_1="X Bad: v")

    def test_rejects_empty_name(self):
        with pytest.raises(ValueError, match="non-empty name"):
            _settings(slot_1=": value")

    def test_rejects_empty_value(self):
        with pytest.raises(ValueError, match="non-empty value"):
            _settings(slot_1="X-A:   ")

    @pytest.mark.parametrize(
        "header",
        ["Authorization: Bearer x", "aUtHoRiZaTiOn: x", "Content-Type: text/plain", "Accept: x"],
    )
    def test_rejects_managed_names(self, header):
        with pytest.raises(ValueError, match="the registry manages this header"):
            _settings(slot_1=header)

    def test_managed_name_error_points_at_credential_settings(self):
        with pytest.raises(ValueError, match="EMBEDDINGS_API_KEY"):
            _settings(slot_1="Authorization: Bearer x")

    def test_rejects_duplicate_name_across_slots(self):
        with pytest.raises(ValueError) as exc:
            _settings(slot_1="X-A: 1", slot_4="X-A: 2")
        assert "EMBEDDINGS_EXTRA_HEADER_4" in str(exc.value)
        assert "EMBEDDINGS_EXTRA_HEADER_1" in str(exc.value)

    def test_rejects_duplicate_case_variant(self):
        with pytest.raises(ValueError, match="already set by EMBEDDINGS_EXTRA_HEADER_1"):
            _settings(slot_1="X-A: 1", slot_2="x-a: 2")

    def test_rejects_oversized_value(self):
        with pytest.raises(ValueError, match="exceeds 4096"):
            _settings(slot_1="X-A: " + "v" * 4097)

    def test_rejects_oversized_name(self):
        with pytest.raises(ValueError, match="exceeds 256"):
            _settings(slot_1="X" * 257 + ": v")

    def test_error_message_names_the_slot(self):
        with pytest.raises(ValueError) as exc:
            _settings(slot_7="X Bad: v")
        assert "EMBEDDINGS_EXTRA_HEADER_7" in str(exc.value)
        assert "custom_headers entry" not in str(exc.value)


@pytest.mark.unit
@pytest.mark.core
class TestSharedValidatorDefaults:
    """The validator move kept the custom-headers messages byte-for-byte."""

    def test_default_context_preserves_messages(self):
        with pytest.raises(ValueError) as exc:
            validate_header_name("X Bad")
        assert str(exc.value) == "custom_headers entry name must be a valid RFC token string"


@pytest.mark.unit
@pytest.mark.core
class TestInertBanner:
    """The banner fires only when headers are set but cannot be sent."""

    def test_banner_warns_when_set_and_provider_local(self, capsys, caplog):
        s = _settings(slot_1="X-Tenant-Id: research", slot_2="X-Route: v2")
        s.embeddings_provider = "sentence-transformers"
        with caplog.at_level(logging.WARNING):
            print_embeddings_extra_headers_inert_banner(s)
        out = capsys.readouterr().out
        assert "EMBEDDINGS_EXTRA_HEADER_<N>" in out
        assert "sentence-transformers" in out
        assert "2 configured header(s)" in out
        assert "EMBEDDINGS_EXTRA_HEADER" in caplog.text

    def test_banner_silent_for_litellm(self, capsys):
        s = _settings(slot_1="X-Tenant-Id: research")
        s.embeddings_provider = "litellm"
        print_embeddings_extra_headers_inert_banner(s)
        assert capsys.readouterr().out == ""

    def test_banner_silent_when_no_headers(self, capsys):
        s = _settings()
        s.embeddings_provider = "sentence-transformers"
        print_embeddings_extra_headers_inert_banner(s)
        assert capsys.readouterr().out == ""

    def test_banner_omits_header_names_and_values(self, capsys):
        s = _settings(slot_1="X-Secret-Name: secret-value")
        s.embeddings_provider = "sentence-transformers"
        print_embeddings_extra_headers_inert_banner(s)
        out = capsys.readouterr().out
        assert "X-Secret-Name" not in out
        assert "secret-value" not in out

    def test_banner_called_at_startup(self):
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        main_source = (repo_root / "registry" / "main.py").read_text()
        assert "print_embeddings_extra_headers_inert_banner(settings)" in main_source
