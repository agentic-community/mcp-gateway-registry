"""Tests for the public {ROOT_PATH}/branding/<name> UI branding route.

The route serves operator-supplied branding files from settings.branding_dir.
Tests point UI_BRANDING_DIR at a temporary folder by patching
``settings.ui_branding_dir``; the route reads the setting on every request.
"""

import os
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient

import registry.main as registry_main
from registry.core.config import settings
from registry.main import app
from registry.utils import branding_files
from registry.utils.branding_files import BRANDING_SVG_CSP, MAX_BRANDING_FILE_BYTES

SVG_CONTENT: bytes = b'<svg xmlns="http://www.w3.org/2000/svg"><rect width="1" height="1"/></svg>'


def _configmap_layout(
    branding_dir: Path,
    name: str,
    content: bytes,
) -> Path:
    """Build the Kubernetes layout key -> ..data/key, ..data -> ..<timestamp>/."""
    timestamp_dir = branding_dir / "..2026_11_06_00_00_00.000000000"
    timestamp_dir.mkdir()
    (timestamp_dir / name).write_bytes(content)
    os.symlink(timestamp_dir.name, branding_dir / "..data")
    os.symlink(Path("..data") / name, branding_dir / name)
    return timestamp_dir


def _swap_during_open(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    swap: Callable[[], None],
    restore: Callable[[], None] | None,
) -> None:
    """Run ``swap`` just before the branding read opens ``target``.

    This lands the replacement in the window between the containment check
    and the open. ``restore`` (if given) runs right after the open, as an
    attacker would to make a later re-check see the original file.
    """
    real_open = os.open

    def _open(path, flags, *args, **kwargs):  # noqa: ANN001, ANN202 - os.open signature
        if Path(path) != target:
            return real_open(path, flags, *args, **kwargs)
        swap()
        try:
            return real_open(path, flags, *args, **kwargs)
        finally:
            if restore is not None:
                restore()

    monkeypatch.setattr(branding_files.os, "open", _open)


@pytest.fixture
def branding_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Point the branding folder at an empty temporary directory."""
    folder = tmp_path / "branding"
    folder.mkdir()
    monkeypatch.setattr(settings, "ui_branding_dir", str(folder))
    return folder


@pytest.fixture
def client() -> TestClient:
    """Test client for the registry app."""
    return TestClient(app)


class TestBrandingRouteServesFiles:
    """Allowed files are served with the right type and headers."""

    def test_css_served_with_content_type_and_headers(
        self,
        branding_dir: Path,
        client: TestClient,
    ) -> None:
        (branding_dir / "theme.css").write_text(":root { --primary-600: 1 2 3; }")

        response = client.get("/branding/theme.css")

        assert response.status_code == status.HTTP_200_OK
        assert response.text == ":root { --primary-600: 1 2 3; }"
        assert response.headers["content-type"].startswith("text/css")
        assert response.headers["cache-control"] == "public, max-age=300"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert "content-security-policy" not in response.headers

    def test_svg_gets_locked_down_csp(
        self,
        branding_dir: Path,
        client: TestClient,
    ) -> None:
        (branding_dir / "logo-light.svg").write_bytes(SVG_CONTENT)

        response = client.get("/branding/logo-light.svg")

        assert response.status_code == status.HTTP_200_OK
        assert response.content == SVG_CONTENT
        assert response.headers["content-type"] == "image/svg+xml"
        assert response.headers["content-security-policy"] == BRANDING_SVG_CSP

    def test_png_served_as_image(
        self,
        branding_dir: Path,
        client: TestClient,
    ) -> None:
        (branding_dir / "favicon.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        response = client.get("/branding/favicon.png")

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "image/png"

    def test_head_returns_headers_without_body(
        self,
        branding_dir: Path,
        client: TestClient,
    ) -> None:
        (branding_dir / "logo.svg").write_bytes(SVG_CONTENT)

        response = client.head("/branding/logo.svg")

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "image/svg+xml"
        assert response.headers["content-length"] == str(len(SVG_CONTENT))
        assert response.content == b""

    def test_edited_file_is_served_without_restart(
        self,
        branding_dir: Path,
        client: TestClient,
    ) -> None:
        theme = branding_dir / "theme.css"
        theme.write_text("a{}")
        assert client.get("/branding/theme.css").text == "a{}"

        theme.write_text("b{}")
        assert client.get("/branding/theme.css").text == "b{}"


class TestBrandingRouteDefaults:
    """manifest.json and theme.css have defaults; nothing else does."""

    def test_missing_manifest_returns_empty_object(
        self,
        branding_dir: Path,
        client: TestClient,
    ) -> None:
        response = client.get("/branding/manifest.json")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {}
        assert response.headers["content-type"] == "application/json"
        assert response.headers["cache-control"] == "public, max-age=300"
        assert response.headers["x-content-type-options"] == "nosniff"

    def test_missing_theme_returns_empty_stylesheet(
        self,
        branding_dir: Path,
        client: TestClient,
    ) -> None:
        response = client.get("/branding/theme.css")

        assert response.status_code == status.HTTP_200_OK
        assert response.text == ""
        assert response.headers["content-type"].startswith("text/css")

    def test_missing_branding_dir_still_serves_defaults(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        client: TestClient,
    ) -> None:
        monkeypatch.setattr(settings, "ui_branding_dir", str(tmp_path / "does-not-exist"))

        assert client.get("/branding/manifest.json").json() == {}
        assert client.get("/branding/theme.css").status_code == status.HTTP_200_OK
        assert client.get("/branding/logo.svg").status_code == status.HTTP_404_NOT_FOUND

    def test_other_missing_file_is_404(
        self,
        branding_dir: Path,
        client: TestClient,
    ) -> None:
        response = client.get("/branding/logo.svg")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_present_manifest_overrides_default(
        self,
        branding_dir: Path,
        client: TestClient,
    ) -> None:
        (branding_dir / "manifest.json").write_text('{"version": 1}')

        assert client.get("/branding/manifest.json").json() == {"version": 1}


class TestBrandingRouteRejects:
    """Bad names, disallowed extensions, oversize files and escapes are 404."""

    @pytest.mark.parametrize(
        "name",
        [
            "..%2Fx.css",
            "%2E%2E%2Fx.css",
            "sub/x.css",
            ".hidden.css",
            "..data",
            "x.exe",
            "theme.CSS.exe",
            "noextension",
            "x" * 70 + ".css",
            "-dash.css",
        ],
    )
    def test_bad_names_are_404(
        self,
        branding_dir: Path,
        client: TestClient,
        name: str,
    ) -> None:
        # Plant targets so a 404 proves the name rule, not a missing file.
        (branding_dir / ".hidden.css").write_text("x{}")
        (branding_dir / "x.exe").write_bytes(b"MZ")
        (branding_dir / "noextension").write_text("x")
        sub = branding_dir / "sub"
        sub.mkdir()
        (sub / "x.css").write_text("x{}")
        (branding_dir.parent / "x.css").write_text("outside{}")

        response = client.get(f"/branding/{name}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "outside" not in response.text

    def test_configmap_data_dir_is_not_listed_or_served(
        self,
        branding_dir: Path,
        client: TestClient,
    ) -> None:
        timestamp_dir = branding_dir / "..2026_11_06_00_00_00.000000000"
        timestamp_dir.mkdir()
        (timestamp_dir / "theme.css").write_text("x{}")
        os.symlink(timestamp_dir.name, branding_dir / "..data")

        assert client.get("/branding/..data").status_code == status.HTTP_404_NOT_FOUND
        assert client.get("/branding/..data/theme.css").status_code == status.HTTP_404_NOT_FOUND

    def test_oversize_file_is_404_and_logged(
        self,
        branding_dir: Path,
        client: TestClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        (branding_dir / "big.png").write_bytes(b"\0" * (MAX_BRANDING_FILE_BYTES + 1))

        with caplog.at_level("WARNING", logger="registry.main"):
            response = client.get("/branding/big.png")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert any("big.png" in record.getMessage() for record in caplog.records)

    def test_file_at_size_limit_is_served(
        self,
        branding_dir: Path,
        client: TestClient,
    ) -> None:
        (branding_dir / "edge.png").write_bytes(b"\0" * MAX_BRANDING_FILE_BYTES)

        response = client.get("/branding/edge.png")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.content) == MAX_BRANDING_FILE_BYTES

    def test_directory_with_allowed_name_is_404(
        self,
        branding_dir: Path,
        client: TestClient,
    ) -> None:
        (branding_dir / "logo.svg").mkdir()

        assert client.get("/branding/logo.svg").status_code == status.HTTP_404_NOT_FOUND


class TestBrandingRouteSymlinks:
    """Symlinks are followed only while they stay inside the folder."""

    def test_symlink_escaping_dir_is_refused(
        self,
        branding_dir: Path,
        client: TestClient,
    ) -> None:
        secret = branding_dir.parent / "secret.css"
        secret.write_text("secret{}")
        os.symlink(secret, branding_dir / "theme.css")

        response = client.get("/branding/theme.css")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "secret" not in response.text

    def test_configmap_style_symlink_inside_dir_is_served(
        self,
        branding_dir: Path,
        client: TestClient,
    ) -> None:
        # Kubernetes layout: key -> ..data/key, ..data -> ..<timestamp>/
        timestamp_dir = branding_dir / "..2026_11_06_00_00_00.000000000"
        timestamp_dir.mkdir()
        (timestamp_dir / "logo.svg").write_bytes(SVG_CONTENT)
        os.symlink(timestamp_dir.name, branding_dir / "..data")
        os.symlink(Path("..data") / "logo.svg", branding_dir / "logo.svg")

        response = client.get("/branding/logo.svg")

        assert response.status_code == status.HTTP_200_OK
        assert response.content == SVG_CONTENT
        assert response.headers["content-security-policy"] == BRANDING_SVG_CSP


class TestBrandingRouteSwapDuringOpen:
    """A path swapped after the containment check never serves outside content."""

    def test_file_swapped_for_symlink_is_refused(
        self,
        branding_dir: Path,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        secret = branding_dir.parent / "secret.css"
        secret.write_text("secret{}")
        theme = branding_dir / "theme.css"
        theme.write_text("a{}")

        def _swap() -> None:
            theme.unlink()
            os.symlink(secret, theme)

        _swap_during_open(monkeypatch, theme.resolve(), _swap, None)

        response = client.get("/branding/theme.css")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "secret" not in response.text

    @pytest.mark.parametrize("swap_back", [False, True], ids=["left-swapped", "swapped-back"])
    def test_parent_dir_swapped_for_symlink_is_refused(
        self,
        branding_dir: Path,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        swap_back: bool,
    ) -> None:
        outside = branding_dir.parent / "outside"
        outside.mkdir()
        (outside / "theme.css").write_text("secret{}")
        timestamp_dir = _configmap_layout(branding_dir, "theme.css", b"a{}")
        parked = branding_dir.parent / "parked"

        def _swap() -> None:
            timestamp_dir.rename(parked)
            os.symlink(outside, timestamp_dir)

        def _restore() -> None:
            timestamp_dir.unlink()
            parked.rename(timestamp_dir)

        target = (timestamp_dir / "theme.css").resolve()
        _swap_during_open(monkeypatch, target, _swap, _restore if swap_back else None)

        response = client.get("/branding/theme.css")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "secret" not in response.text


class TestBrandingRootPathRewrite:
    """The cached index.html rewrites /branding/ URLs under ROOT_PATH."""

    def test_branding_urls_get_root_path_prefix(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "index.html").write_text(
            "<html><head>"
            '<link rel="icon" href="/branding/favicon.svg">'
            '<link rel="stylesheet" href="/branding/theme.css">'
            '<script src="/static/js/main.js"></script>'
            "</head><body></body></html>"
        )
        monkeypatch.setattr(registry_main, "FRONTEND_BUILD_PATH", tmp_path)
        monkeypatch.setattr(registry_main, "_ROOT_PATH", "/registry/")

        html = registry_main._build_cached_index_html()

        assert html is not None
        assert 'href="/registry/branding/favicon.svg"' in html
        assert 'href="/registry/branding/theme.css"' in html
        assert 'href="/branding/' not in html
        assert 'src="/registry/static/js/main.js"' in html
