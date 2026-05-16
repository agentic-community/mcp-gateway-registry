"""Tests for the atomic-write helper in registry/core/nginx_service.py.

Covers issue #1044: nginx config writes must be atomic at the filesystem
level so concurrent ``nginx -t`` calls and other readers never see a
truncated mid-write file.
"""

import os
from pathlib import Path

import pytest

from registry.core.nginx_service import _atomic_write_text


class TestAtomicWrite:
    def test_creates_file_when_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "nginx.conf"
        _atomic_write_text(target, "hello")
        assert target.read_text() == "hello"

    def test_replaces_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "nginx.conf"
        target.write_text("old content")
        _atomic_write_text(target, "new content")
        assert target.read_text() == "new content"

    def test_uses_temp_file_in_same_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = tmp_path / "nginx.conf"
        captured: dict[str, str] = {}
        original_replace = os.replace

        def fake_replace(src: str, dst: str) -> None:
            captured["src"] = str(src)
            captured["dst"] = str(dst)
            original_replace(src, dst)

        monkeypatch.setattr("os.replace", fake_replace)
        _atomic_write_text(target, "data")
        assert Path(captured["src"]).parent == tmp_path
        assert captured["dst"] == str(target)

    def test_cleans_up_temp_on_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = tmp_path / "nginx.conf"
        target.write_text("preserved")

        def boom(src: str, dst: str) -> None:
            raise OSError("disk full")

        monkeypatch.setattr("os.replace", boom)
        with pytest.raises(OSError):
            _atomic_write_text(target, "should not appear")

        # Original file unchanged
        assert target.read_text() == "preserved"
        # No leftover temp files
        leftovers = list(tmp_path.glob(".nginx.conf.tmp.*"))
        assert leftovers == []

    def test_preserves_existing_file_permissions(self, tmp_path: Path) -> None:
        target = tmp_path / "nginx.conf"
        target.write_text("old")
        target.chmod(0o644)
        _atomic_write_text(target, "new")
        mode = target.stat().st_mode & 0o777
        assert mode == 0o644

    def test_default_mode_when_destination_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "nginx.conf"
        _atomic_write_text(target, "content")
        mode = target.stat().st_mode & 0o777
        assert mode == 0o644

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "nginx.conf"
        assert not target.parent.exists()
        _atomic_write_text(target, "content")
        assert target.read_text() == "content"


class TestCleanupStaleTempFiles:
    def test_removes_leftover_temp_files(self, tmp_path: Path) -> None:
        from registry.core.nginx_service import _cleanup_stale_temp_files

        config = tmp_path / "nginx.conf"
        # Simulate two crashed writes
        (tmp_path / ".nginx.conf.tmp.aaa").write_text("partial1")
        (tmp_path / ".nginx.conf.tmp.bbb").write_text("partial2")
        # And one unrelated file that should NOT be removed
        (tmp_path / "other.txt").write_text("keep me")

        _cleanup_stale_temp_files(config)

        leftovers = list(tmp_path.glob(".nginx.conf.tmp.*"))
        assert leftovers == []
        assert (tmp_path / "other.txt").exists()

    def test_no_op_when_no_temp_files(self, tmp_path: Path) -> None:
        from registry.core.nginx_service import _cleanup_stale_temp_files

        config = tmp_path / "nginx.conf"
        config.write_text("config content")
        # Should not raise
        _cleanup_stale_temp_files(config)
        assert config.read_text() == "config content"

    def test_no_op_when_directory_missing(self, tmp_path: Path) -> None:
        from registry.core.nginx_service import _cleanup_stale_temp_files

        # Pass a path under a non-existent dir; should warn but not raise
        config = tmp_path / "nope" / "nginx.conf"
        _cleanup_stale_temp_files(config)
