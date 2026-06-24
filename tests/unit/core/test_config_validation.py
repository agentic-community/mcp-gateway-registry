"""
Unit tests for STORAGE_BACKEND field validation in registry.core.config.

Covers the _validate_storage_backend @field_validator added by issue #954:
- Accepts every value in ALLOWED_STORAGE_BACKENDS.
- Rejects typos with a ValidationError whose message lists the allowlist.
- Normalizes case and whitespace.
- Coerces empty/unset values to "mongodb-ce".
- Rejects the legacy "file" value with an actionable migration message.
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
    """Empty/unset STORAGE_BACKEND coerces to mongodb-ce."""

    def test_empty_string_coerces_to_mongodb_ce(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        """STORAGE_BACKEND="" must not error; coerces to 'mongodb-ce'."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("STORAGE_BACKEND", "")

        settings = Settings()

        assert settings.storage_backend == "mongodb-ce"

    def test_unset_defaults_to_mongodb_ce(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        """Unset STORAGE_BACKEND uses the Field default ('mongodb-ce')."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("STORAGE_BACKEND", raising=False)

        settings = Settings()

        assert settings.storage_backend == "mongodb-ce"

    def test_file_backend_rejected_with_migration_message(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        """STORAGE_BACKEND='file' must raise ValueError with migration instructions."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("STORAGE_BACKEND", "file")

        with pytest.raises(ValidationError) as excinfo:
            Settings()

        message = str(excinfo.value)
        assert "removed" in message.lower() or "file" in message.lower()
        assert "mongodb-ce" in message or "documentdb" in message


@pytest.mark.unit
@pytest.mark.core
class TestInternalDeploymentType:
    """Cover the _validate_internal_deployment_type @field_validator (issue #1216)."""

    @pytest.mark.parametrize("value", ["none", "dev", "workshop", "other"])
    def test_every_valid_value_accepted(
        self,
        monkeypatch,
        tmp_path,
        value: str,
    ) -> None:
        """Settings() accepts each allowed value (case/space-normalized)."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("INTERNAL_DEPLOYMENT_TYPE", f"  {value.upper()} ")

        settings = Settings()

        assert settings.internal_deployment_type.value == value

    def test_invalid_value_rejected(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        """Unknown values raise ValidationError listing the accepted values."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("INTERNAL_DEPLOYMENT_TYPE", "production")

        with pytest.raises(ValidationError, match="Accepted values"):
            Settings()

    def test_empty_and_unset_default_to_none(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        """Empty string and unset both coerce to 'none'."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("INTERNAL_DEPLOYMENT_TYPE", "")
        assert Settings().internal_deployment_type.value == "none"

        monkeypatch.delenv("INTERNAL_DEPLOYMENT_TYPE", raising=False)
        assert Settings().internal_deployment_type.value == "none"

    def test_internal_only_deployment_defaults_false(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        """internal_only_deployment defaults to False when unset."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("INTERNAL_ONLY_DEPLOYMENT", raising=False)
        assert Settings().internal_only_deployment is False
