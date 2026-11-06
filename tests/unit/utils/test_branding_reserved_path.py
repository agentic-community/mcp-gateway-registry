"""Tests that the 'branding' first path segment cannot be registered.

{ROOT_PATH}/branding/<name> is the public UI branding route. A server at
/branding would get an nginx location block that shadows it, so server and
agent registration both refuse that first segment.
"""

import pytest

from registry.exceptions import UrlValidationError
from registry.schemas.agent_models import _validate_path_format
from registry.utils.url_guard import validate_server_path


class TestServerPathBrandingReserved:
    """validate_server_path (called by server registration) rejects /branding."""

    @pytest.mark.parametrize(
        "path",
        ["/branding", "/branding/", "/Branding", "//branding//", "/branding/logo", "branding"],
    )
    def test_branding_segment_rejected(
        self,
        path: str,
    ) -> None:
        with pytest.raises(UrlValidationError, match="reserved"):
            validate_server_path(path)

    @pytest.mark.parametrize(
        "path",
        ["/branding-tools", "/my-branding", "/team/branding", "/brandingx"],
    )
    def test_other_paths_containing_branding_allowed(
        self,
        path: str,
    ) -> None:
        validate_server_path(path)


class TestServerRegistrationBrandingReserved:
    """The server registration entry point refuses /branding before storing it."""

    async def test_register_server_rejects_branding(
        self,
    ) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from registry.services.server_service import ServerService

        service = ServerService.__new__(ServerService)
        service._repo = MagicMock()
        service._repo.get = AsyncMock(return_value=None)
        service._repo.create = AsyncMock()

        with pytest.raises(UrlValidationError, match="reserved"):
            await service.register_server(
                {"path": "/branding", "server_name": "x", "proxy_pass_url": "http://example.com/"}
            )

        service._repo.create.assert_not_called()


class TestAgentPathBrandingReserved:
    """Agent path validation rejects /branding."""

    @pytest.mark.parametrize("path", ["/branding", "/BRANDING", "/branding/agent"])
    def test_branding_segment_rejected(
        self,
        path: str,
    ) -> None:
        with pytest.raises(ValueError, match="reserved"):
            _validate_path_format(path)

    def test_other_agent_path_allowed(
        self,
    ) -> None:
        assert _validate_path_format("/branding-agent") == "/branding-agent"
