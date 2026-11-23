"""Tests for the UI branding archive writer (python -m registry.utils.branding_writer).

The writer reads UI_BRANDING_ARCHIVE_B64 / UI_BRANDING_DIR from the environment,
validates every tar member, and writes the files with mode 0o644. Any violation
must exit non-zero and leave the target folder untouched.
"""

import base64
import gzip
import io
import logging
import os
import stat
import tarfile
from pathlib import Path

import pytest

from registry.utils import branding_writer
from registry.utils.branding_files import (
    MAX_BRANDING_ARCHIVE_ENTRIES,
    MAX_BRANDING_ARCHIVE_FILES,
    MAX_BRANDING_ARCHIVE_TAR_BYTES,
    MAX_BRANDING_ARCHIVE_TOTAL_BYTES,
)


def _archive_b64(
    members: list[tarfile.TarInfo | tuple[str, bytes]],
) -> str:
    """Build a base64 .tar.gz from (name, data) pairs or raw TarInfo entries."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for member in members:
            if isinstance(member, tarfile.TarInfo):
                tar.addfile(member)
                continue
            name, data = member
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _special_member(
    name: str,
    member_type: bytes,
    linkname: str = "",
) -> tarfile.TarInfo:
    """Build a non-regular tar member header."""
    info = tarfile.TarInfo(name)
    info.type = member_type
    info.linkname = linkname
    return info


@pytest.fixture
def target_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Empty branding folder wired in through UI_BRANDING_DIR."""
    folder = tmp_path / "branding"
    folder.mkdir()
    monkeypatch.setenv("UI_BRANDING_DIR", str(folder))
    return folder


def _run(
    monkeypatch: pytest.MonkeyPatch,
    archive_b64: str,
) -> int:
    """Run the writer CLI with the given archive value."""
    monkeypatch.setenv("UI_BRANDING_ARCHIVE_B64", archive_b64)
    return branding_writer.main()


class TestBrandingWriterSuccess:
    """Valid archives are extracted."""

    def test_valid_archive_is_extracted_with_0644(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        archive = _archive_b64(
            [
                ("manifest.json", b'{"version": 1}'),
                ("theme.css", b":root{}"),
                ("logo-light.svg", b"<svg/>"),
                ("Inter.woff2", b"wOF2"),
            ]
        )

        assert _run(monkeypatch, archive) == 0

        assert sorted(p.name for p in target_dir.iterdir()) == [
            "Inter.woff2",
            "logo-light.svg",
            "manifest.json",
            "theme.css",
        ]
        assert (target_dir / "manifest.json").read_bytes() == b'{"version": 1}'
        for path in target_dir.iterdir():
            assert stat.S_IMODE(path.stat().st_mode) == 0o644

    def test_missing_target_dir_is_created(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        folder = tmp_path / "new" / "branding"
        monkeypatch.setenv("UI_BRANDING_DIR", str(folder))

        assert _run(monkeypatch, _archive_b64([("theme.css", b"a{}")])) == 0
        assert (folder / "theme.css").read_bytes() == b"a{}"

    def test_hidden_entries_in_target_do_not_block(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (target_dir / ".gitkeep").write_text("")

        assert _run(monkeypatch, _archive_b64([("theme.css", b"a{}")])) == 0
        assert (target_dir / "theme.css").exists()

    def test_limits_are_inclusive(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        per_file = MAX_BRANDING_ARCHIVE_TOTAL_BYTES // MAX_BRANDING_ARCHIVE_FILES
        members = [(f"f{i}.png", b"\0" * per_file) for i in range(MAX_BRANDING_ARCHIVE_FILES)]

        assert _run(monkeypatch, _archive_b64(members)) == 0
        assert len(list(target_dir.iterdir())) == MAX_BRANDING_ARCHIVE_FILES

    def test_archive_of_folder_root_is_accepted(
        self,
        tmp_path: Path,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Same layout as `tar -czf - -C ./branding .`: a "." entry plus
        # "./<name>" members.
        source = tmp_path / "source"
        source.mkdir()
        (source / "theme.css").write_bytes(b"a{}")
        (source / "logo.svg").write_bytes(b"<svg/>")
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            tar.add(source, arcname=".")
        names = tarfile.open(fileobj=io.BytesIO(buffer.getvalue()), mode="r:gz").getnames()
        assert "." in names
        assert "./theme.css" in names

        archive = base64.b64encode(buffer.getvalue()).decode("ascii")

        assert _run(monkeypatch, archive) == 0
        assert sorted(p.name for p in target_dir.iterdir()) == ["logo.svg", "theme.css"]
        assert (target_dir / "theme.css").read_bytes() == b"a{}"

    def test_dot_slash_prefixed_members_are_accepted(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        archive = _archive_b64(
            [
                _special_member("./", tarfile.DIRTYPE),
                ("./manifest.json", b"{}"),
                ("theme.css", b"a{}"),
            ]
        )

        assert _run(monkeypatch, archive) == 0
        assert sorted(p.name for p in target_dir.iterdir()) == ["manifest.json", "theme.css"]

    @pytest.mark.parametrize(
        "metadata_name",
        ["._logo.svg", "./._logo.svg", ".DS_Store", "./.DS_Store"],
    )
    def test_macos_metadata_members_are_skipped(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        metadata_name: str,
    ) -> None:
        archive = _archive_b64([("./logo.svg", b"<svg/>"), (metadata_name, b"\0\x05\x16\x07")])

        with caplog.at_level(logging.INFO):
            assert _run(monkeypatch, archive) == 0

        assert sorted(p.name for p in target_dir.iterdir()) == ["logo.svg"]
        skip_logs = [r.getMessage() for r in caplog.records if "skipped" in r.getMessage()]
        assert skip_logs == ["Branding: skipped 1 archive root and macOS metadata entries"]

    def test_dot_slash_duplicate_of_bare_name_is_rejected(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        archive = _archive_b64([("theme.css", b"a{}"), ("./theme.css", b"b{}")])

        assert _run(monkeypatch, archive) != 0
        assert list(target_dir.iterdir()) == []

    def test_archive_with_trailing_nul_padding_is_accepted(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # macOS bsdtar pads `tar -czf -` output with NUL bytes up to a
        # 10240-byte block after the gzip stream ends.
        raw = base64.b64decode(_archive_b64([("./theme.css", b"a{}"), ("logo.svg", b"<svg/>")]))
        assert len(raw) < 10240
        padded = raw + b"\0" * (10240 - len(raw))

        assert _run(monkeypatch, base64.b64encode(padded).decode("ascii")) == 0
        assert sorted(p.name for p in target_dir.iterdir()) == ["logo.svg", "theme.css"]
        assert (target_dir / "theme.css").read_bytes() == b"a{}"


class TestBrandingWriterNoop:
    """An unset or empty archive variable does nothing."""

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_empty_var_is_noop(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        value: str | None,
    ) -> None:
        if value is None:
            monkeypatch.delenv("UI_BRANDING_ARCHIVE_B64", raising=False)
            assert branding_writer.main() == 0
        else:
            assert _run(monkeypatch, value) == 0
        assert list(target_dir.iterdir()) == []

    def test_empty_var_ignores_non_empty_target(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (target_dir / "theme.css").write_text("mounted{}")

        assert _run(monkeypatch, "") == 0
        assert (target_dir / "theme.css").read_text() == "mounted{}"


class TestBrandingWriterRejects:
    """Every rule violation exits non-zero and writes nothing."""

    @pytest.mark.parametrize(
        "name",
        [
            "../evil.css",
            "/etc/evil.css",
            "noextension",
            "sub/theme.css",
            ".hidden.css",
            "..data",
            "evil.exe",
            "./sub/theme.css",
            "././theme.css",
            "x" * 70 + ".css",
        ],
    )
    def test_rejects_bad_member_names(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        name: str,
    ) -> None:
        archive = _archive_b64([("theme.css", b"a{}"), (name, b"x")])

        with caplog.at_level(logging.ERROR):
            assert _run(monkeypatch, archive) != 0

        assert list(target_dir.iterdir()) == []
        assert not (target_dir.parent / "evil.css").exists()
        assert "not an allowed branding file name" in caplog.text

    @pytest.mark.parametrize(
        "member",
        [
            _special_member("logo.svg", tarfile.SYMTYPE, "/etc/passwd"),
            _special_member("logo.svg", tarfile.LNKTYPE, "theme.css"),
            _special_member("assets", tarfile.DIRTYPE),
            _special_member("dev.css", tarfile.CHRTYPE),
            _special_member("fifo.css", tarfile.FIFOTYPE),
        ],
        ids=["symlink", "hardlink", "directory", "device", "fifo"],
    )
    def test_rejects_non_regular_members(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        member: tarfile.TarInfo,
    ) -> None:
        archive = _archive_b64([("theme.css", b"a{}"), member])

        with caplog.at_level(logging.ERROR):
            assert _run(monkeypatch, archive) != 0

        assert list(target_dir.iterdir()) == []
        assert "is not a regular file" in caplog.text

    def test_rejects_nested_directory_files(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        archive = _archive_b64(
            [_special_member("img", tarfile.DIRTYPE), ("img/logo.svg", b"<svg/>")]
        )

        assert _run(monkeypatch, archive) != 0
        assert list(target_dir.iterdir()) == []

    def test_rejects_too_many_files(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        members = [(f"f{i}.css", b"") for i in range(MAX_BRANDING_ARCHIVE_FILES + 1)]

        with caplog.at_level(logging.ERROR):
            assert _run(monkeypatch, _archive_b64(members)) != 0

        assert list(target_dir.iterdir()) == []
        assert f"more than {MAX_BRANDING_ARCHIVE_FILES} files" in caplog.text

    def test_rejects_oversize_total(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        half = MAX_BRANDING_ARCHIVE_TOTAL_BYTES // 2
        members = [("a.png", b"\0" * half), ("b.png", b"\0" * (half + 1))]

        with caplog.at_level(logging.ERROR):
            assert _run(monkeypatch, _archive_b64(members)) != 0

        assert list(target_dir.iterdir()) == []
        assert "exceed" in caplog.text

    def test_rejects_too_many_skipped_entries(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Zero-sized entries the writer skips still cost a header parse each,
        # so they count toward the entry cap.
        members: list[tarfile.TarInfo | tuple[str, bytes]] = [("logo.svg", b"<svg/>")]
        members += [(f"._f{i}.css", b"") for i in range(MAX_BRANDING_ARCHIVE_ENTRIES)]

        with caplog.at_level(logging.INFO):
            assert _run(monkeypatch, _archive_b64(members)) != 0

        assert list(target_dir.iterdir()) == []
        assert f"more than {MAX_BRANDING_ARCHIVE_ENTRIES} entries" in caplog.text
        assert "skipped" not in caplog.text

    def test_entry_limit_is_inclusive(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        members: list[tarfile.TarInfo | tuple[str, bytes]] = [("logo.svg", b"<svg/>")]
        members += [(f"._f{i}.css", b"") for i in range(MAX_BRANDING_ARCHIVE_ENTRIES - 1)]

        assert _run(monkeypatch, _archive_b64(members)) == 0
        assert [p.name for p in target_dir.iterdir()] == ["logo.svg"]

    def test_rejects_gzip_bomb_before_parsing(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Zeros compress about 1000:1, so the compressed value is small while
        # the stream would pass the tar-stream cap.
        bomb = gzip.compress(b"\0" * (MAX_BRANDING_ARCHIVE_TAR_BYTES + 1))
        assert len(bomb) < MAX_BRANDING_ARCHIVE_TAR_BYTES

        with caplog.at_level(logging.ERROR):
            assert _run(monkeypatch, base64.b64encode(bomb).decode("ascii")) != 0

        assert list(target_dir.iterdir()) == []
        assert f"decompresses to more than {MAX_BRANDING_ARCHIVE_TAR_BYTES} bytes" in caplog.text

    def test_rejects_oversize_encoded_archive(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        oversize = b"\0" * (MAX_BRANDING_ARCHIVE_TAR_BYTES + 1)

        with caplog.at_level(logging.ERROR):
            assert _run(monkeypatch, base64.b64encode(oversize).decode("ascii")) != 0

        assert list(target_dir.iterdir()) == []
        assert f"larger than the {MAX_BRANDING_ARCHIVE_TAR_BYTES} byte limit" in caplog.text

    def test_rejects_data_after_gzip_stream(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        raw = base64.b64decode(_archive_b64([("theme.css", b"a{}")])) + b"junk"

        with caplog.at_level(logging.ERROR):
            assert _run(monkeypatch, base64.b64encode(raw).decode("ascii")) != 0

        assert list(target_dir.iterdir()) == []
        assert "unexpected data after the gzip stream" in caplog.text

    def test_rejects_truncated_gzip_stream(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        raw = base64.b64decode(_archive_b64([("theme.css", b"a{}" * 1000)]))

        assert _run(monkeypatch, base64.b64encode(raw[:-12]).decode("ascii")) != 0
        assert list(target_dir.iterdir()) == []

    def test_rejects_duplicate_names(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        archive = _archive_b64([("theme.css", b"a{}"), ("theme.css", b"b{}")])

        assert _run(monkeypatch, archive) != 0
        assert list(target_dir.iterdir()) == []

    def test_rejects_existing_files_in_target(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        (target_dir / "logo.svg").write_text("<svg/>")

        with caplog.at_level(logging.ERROR):
            assert _run(monkeypatch, _archive_b64([("theme.css", b"a{}")])) != 0

        assert sorted(p.name for p in target_dir.iterdir()) == ["logo.svg"]
        assert "UI_BRANDING_ARCHIVE_B64" in caplog.text
        assert "UI_BRANDING_DIR" in caplog.text
        assert "logo.svg" in caplog.text

    @pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root bypasses directory permission bits",
    )
    def test_rejects_read_only_target(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Docker Compose mounts the (usually empty) folder read-only.
        target_dir.chmod(0o555)
        try:
            with caplog.at_level(logging.ERROR):
                assert _run(monkeypatch, _archive_b64([("theme.css", b"a{}")])) != 0
        finally:
            target_dir.chmod(0o755)

        assert list(target_dir.iterdir()) == []
        assert f"UI_BRANDING_DIR ({target_dir}) is read-only" in caplog.text
        assert "unset UI_BRANDING_ARCHIVE_B64" in caplog.text
        assert "Traceback" not in caplog.text

    @pytest.mark.parametrize(
        "value",
        ["not base64!!", base64.b64encode(b"not a tarball").decode("ascii")],
        ids=["bad-base64", "not-a-tar-gz"],
    )
    def test_rejects_undecodable_archive(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        value: str,
    ) -> None:
        assert _run(monkeypatch, value) != 0
        assert list(target_dir.iterdir()) == []

    def test_rejects_empty_archive(
        self,
        target_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        assert _run(monkeypatch, _archive_b64([])) != 0
