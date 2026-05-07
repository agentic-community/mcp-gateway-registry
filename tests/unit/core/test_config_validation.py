"""
Unit tests for STORAGE_BACKEND field validation in registry.core.config.

Covers the _validate_storage_backend @field_validator added by issue #954:
- Accepts every value in ALLOWED_STORAGE_BACKENDS.
- Rejects typos with a ValidationError whose message lists the allowlist.
- Normalizes case and whitespace.
- Coerces empty/unset values to the historical default ("file").
"""

import pytest
from pydantic import ValidationError

from registry.core.config import (
    ALLOWED_STORAGE_BACKENDS,
    MONGODB_BACKENDS,
    Settings,
)


@pytest.mark.unit
@pytest.mark.core
class TestStorageBackendAllowlist:
    """Cover every value in ALLOWED_STORAGE_BACKENDS."""

    @pytest.mark.parametrize("value", sorted(ALLOWED_STORAGE_BACKENDS))
    def test_every_allowlist_value_accepted(
        self,
        monkeypatch,
        tmp_path,
        value: str,
    ) -> None:
        """Settings() accepts every canonical value and returns it unchanged."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("STORAGE_BACKEND", value)

        settings = Settings()

        assert settings.storage_backend == value

    def test_mongodb_backends_subset_of_allowlist(self) -> None:
        """MONGODB_BACKENDS must be a strict subset of ALLOWED_STORAGE_BACKENDS."""
        assert MONGODB_BACKENDS <= ALLOWED_STORAGE_BACKENDS
        assert "file" not in MONGODB_BACKENDS
        assert "documentdb" in MONGODB_BACKENDS
        assert "mongodb-ce" in MONGODB_BACKENDS
        assert "mongodb" in MONGODB_BACKENDS
        assert "mongodb-atlas" in MONGODB_BACKENDS


@pytest.mark.unit
@pytest.mark.core
class TestStorageBackendRejections:
    """Unknown STORAGE_BACKEND values must fail with a clear message."""

    @pytest.mark.parametrize(
        "bad_value",
        [
            "mongo",
            "mongodb-prod",
            "MongoDb-Atlas-Cluster",
            "mysql",
            "postgres",
            "filez",
            "doc",
        ],
    )
    def test_unknown_value_raises_validation_error(
        self,
        monkeypatch,
        tmp_path,
        bad_value: str,
    ) -> None:
        """Every unknown value must raise ValidationError."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("STORAGE_BACKEND", bad_value)

        with pytest.raises(ValidationError) as excinfo:
            Settings()

        message = str(excinfo.value)
        assert "Invalid STORAGE_BACKEND" in message
        assert "Accepted values:" in message

    def test_error_message_lists_every_accepted_value(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        """The error must name every accepted value so operators can self-serve."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("STORAGE_BACKEND", "totally-bogus")

        with pytest.raises(ValidationError) as excinfo:
            Settings()

        message = str(excinfo.value)
        for accepted in ALLOWED_STORAGE_BACKENDS:
            assert accepted in message, f"error message missing {accepted!r}"


@pytest.mark.unit
@pytest.mark.core
class TestStorageBackendNormalization:
    """Case and whitespace must normalize to the canonical form."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("MongoDB-Atlas", "mongodb-atlas"),
            ("MONGODB", "mongodb"),
            ("MONGODB-CE", "mongodb-ce"),
            ("DocumentDB", "documentdb"),
            ("FILE", "file"),
            ("  mongodb  ", "mongodb"),
            ("\tmongodb-atlas\n", "mongodb-atlas"),
        ],
    )
    def test_case_and_whitespace_normalize(
        self,
        monkeypatch,
        tmp_path,
        raw: str,
        expected: str,
    ) -> None:
        """Leading/trailing whitespace stripped, value lowercased."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("STORAGE_BACKEND", raw)

        settings = Settings()

        assert settings.storage_backend == expected


@pytest.mark.unit
@pytest.mark.core
class TestStorageBackendEmptyValues:
    """Empty/unset STORAGE_BACKEND coerces to the historical default."""

    def test_empty_string_coerces_to_file(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        """STORAGE_BACKEND="" must not error; coerces to 'file'."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("STORAGE_BACKEND", "")

        settings = Settings()

        assert settings.storage_backend == "file"

    def test_unset_defaults_to_file(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        """Unset STORAGE_BACKEND uses the Field default."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("STORAGE_BACKEND", raising=False)

        settings = Settings()

        assert settings.storage_backend == "file"
