"""CLI entrypoint: extract the UI branding archive at container start.

Invoked by ``docker/registry-entrypoint.sh``:

    python -m registry.utils.branding_writer

Reads ``UI_BRANDING_ARCHIVE_B64`` (base64 of a ``.tar.gz`` of the flat branding
folder) and ``UI_BRANDING_DIR`` (target folder, default ``/app/branding``) from
the environment. Every member is validated before any byte is written, and
members are read one by one with ``TarFile.extractfile``; the archive is never
handed to ``extractall``. Unlike the RUM writer this step fails startup: a
branding archive the operator asked for but that cannot be applied is a
configuration error, so the process exits non-zero with a message saying what
was wrong and how to fix it. An empty ``UI_BRANDING_ARCHIVE_B64`` is a no-op.
"""

import base64
import binascii
import io
import logging
import os
import sys
import tarfile
import zlib
from pathlib import Path

from registry.utils.branding_files import (
    MAX_BRANDING_ARCHIVE_ENTRIES,
    MAX_BRANDING_ARCHIVE_FILES,
    MAX_BRANDING_ARCHIVE_TAR_BYTES,
    MAX_BRANDING_ARCHIVE_TOTAL_BYTES,
    branding_content_type,
)

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

ARCHIVE_ENV_VAR: str = "UI_BRANDING_ARCHIVE_B64"
DIR_ENV_VAR: str = "UI_BRANDING_DIR"
DEFAULT_BRANDING_DIR: str = "/app/branding"

_FILE_MODE: int = 0o644
_REGULAR_TYPES: frozenset[bytes] = frozenset({tarfile.REGTYPE, tarfile.AREGTYPE})
# The archive root entry written by `tar -C <folder> .` ("./", which tarfile
# reports as ".").
_ROOT_ENTRY_NAMES: frozenset[str] = frozenset({".", "./"})
_HOW_TO_BUILD: str = (
    "Build the value from the branding folder with "
    "`tar -czf - -C ./branding . | base64 | tr -d '\\n'`."
)
# zlib window bits that accept exactly one gzip member (header and trailer).
_GZIP_WBITS: int = 16 + zlib.MAX_WBITS


class BrandingArchiveError(Exception):
    """Raised when the branding archive or its target folder is unusable."""


def _decode_archive(
    archive_b64: str,
) -> bytes:
    """Decode the base64 archive value.

    Args:
        archive_b64: Value of UI_BRANDING_ARCHIVE_B64.

    Returns:
        The raw .tar.gz bytes.

    Raises:
        BrandingArchiveError: If the value is not valid base64 or decodes to
            more than ``MAX_BRANDING_ARCHIVE_TAR_BYTES``.
    """
    compact = "".join(archive_b64.split())
    try:
        archive_bytes = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as e:
        raise BrandingArchiveError(
            f"{ARCHIVE_ENV_VAR} is not valid base64 ({e}). Set it to the base64 "
            f"encoding of a .tar.gz of the branding folder. {_HOW_TO_BUILD}"
        ) from e
    if len(archive_bytes) > MAX_BRANDING_ARCHIVE_TAR_BYTES:
        raise BrandingArchiveError(
            f"{ARCHIVE_ENV_VAR}: archive is {len(archive_bytes)} bytes, larger than the "
            f"{MAX_BRANDING_ARCHIVE_TAR_BYTES} byte limit. Shrink or remove large images "
            "and fonts."
        )
    return archive_bytes


def _decompress_archive(
    archive_bytes: bytes,
) -> bytes:
    """Gunzip the archive, stopping at ``MAX_BRANDING_ARCHIVE_TAR_BYTES``.

    The whole tar stream is bounded here, before tarfile parses any header, so
    a gzip bomb or an oversized pax or long-name record costs at most the
    limit. Trailing NUL bytes after the gzip stream are accepted: bsdtar (the
    macOS ``tar``) pads its output to a 10240-byte record.

    Args:
        archive_bytes: The raw .tar.gz bytes.

    Returns:
        The uncompressed tar stream.

    Raises:
        BrandingArchiveError: If the data is not one complete gzip stream or
            decompresses past the limit.
    """
    decompressor = zlib.decompressobj(wbits=_GZIP_WBITS)
    try:
        tar_bytes = decompressor.decompress(archive_bytes, MAX_BRANDING_ARCHIVE_TAR_BYTES + 1)
    except zlib.error as e:
        raise BrandingArchiveError(
            f"{ARCHIVE_ENV_VAR} does not decode to a readable .tar.gz archive ({e}). "
            f"{_HOW_TO_BUILD}"
        ) from e
    if len(tar_bytes) > MAX_BRANDING_ARCHIVE_TAR_BYTES:
        raise BrandingArchiveError(
            f"{ARCHIVE_ENV_VAR}: archive decompresses to more than "
            f"{MAX_BRANDING_ARCHIVE_TAR_BYTES} bytes. Shrink or remove large images and fonts."
        )
    if not decompressor.eof:
        raise BrandingArchiveError(
            f"{ARCHIVE_ENV_VAR}: the .tar.gz archive is truncated. {_HOW_TO_BUILD}"
        )
    if decompressor.unused_data.strip(b"\0"):
        raise BrandingArchiveError(
            f"{ARCHIVE_ENV_VAR}: unexpected data after the gzip stream. {_HOW_TO_BUILD}"
        )
    return tar_bytes


def _normalize_member_name(
    member: tarfile.TarInfo,
) -> str | None:
    """Return the name a member is validated under, or None to skip it.

    Skips the archive root directory entry and macOS metadata members
    (AppleDouble ``._*`` files and ``.DS_Store``), and strips one leading
    ``./`` so archives made with ``tar -C <folder> .`` are accepted.

    Args:
        member: The tar member header.

    Returns:
        The normalized name, or None when the member is skipped.
    """
    raw_name = member.name
    if member.isdir() and raw_name in _ROOT_ENTRY_NAMES:
        return None
    name = raw_name.removeprefix("./")
    if name.startswith("._") or name == ".DS_Store":
        return None
    return name


def _validate_member(
    member: tarfile.TarInfo,
    name: str,
    seen_names: set[str],
) -> None:
    """Check one archive member against the branding archive rules.

    Args:
        member: The tar member header.
        name: The member name after :func:`_normalize_member_name`.
        seen_names: Names already accepted, to reject duplicates.

    Raises:
        BrandingArchiveError: If the member is not an allowed regular file.
    """
    if member.type not in _REGULAR_TYPES:
        raise BrandingArchiveError(
            f"{ARCHIVE_ENV_VAR}: archive member {member.name!r} is not a regular file "
            "(directories, links, devices and other special entries are not allowed). "
            f"The archive must contain only the files of one flat folder. {_HOW_TO_BUILD}"
        )
    if branding_content_type(name) is None:
        raise BrandingArchiveError(
            f"{ARCHIVE_ENV_VAR}: archive member {member.name!r} is not an allowed branding "
            "file name. Names must be bare file names (no subfolders, no leading '.', at "
            "most 64 characters of letters, digits, '.', '_', '-') ending in .json, .css, "
            f".svg, .png, .webp, .jpg, .jpeg, .ico or .woff2. {_HOW_TO_BUILD}"
        )
    if name in seen_names:
        raise BrandingArchiveError(
            f"{ARCHIVE_ENV_VAR}: archive member {name!r} appears more than once. "
            "Rebuild the archive so each file is included once."
        )


def _collect_members(
    tar: tarfile.TarFile,
) -> list[tuple[str, tarfile.TarInfo]]:
    """Walk every tar header and return the members to extract.

    Every header counts toward ``MAX_BRANDING_ARCHIVE_ENTRIES`` and every
    declared size toward ``MAX_BRANDING_ARCHIVE_TOTAL_BYTES``, skipped ones
    included, so entries the writer ignores cannot make it do unbounded work.

    Args:
        tar: The open archive.

    Returns:
        Normalized names paired with their members, in archive order.

    Raises:
        BrandingArchiveError: On any rule violation.
    """
    members: list[tuple[str, tarfile.TarInfo]] = []
    seen_names: set[str] = set()
    entries = 0
    total_bytes = 0
    while (member := tar.next()) is not None:
        entries += 1
        if entries > MAX_BRANDING_ARCHIVE_ENTRIES:
            raise BrandingArchiveError(
                f"{ARCHIVE_ENV_VAR}: archive has more than {MAX_BRANDING_ARCHIVE_ENTRIES} "
                f"entries. Include only the branding files. {_HOW_TO_BUILD}"
            )
        total_bytes += member.size
        if total_bytes > MAX_BRANDING_ARCHIVE_TOTAL_BYTES:
            raise BrandingArchiveError(
                f"{ARCHIVE_ENV_VAR}: archive contents exceed "
                f"{MAX_BRANDING_ARCHIVE_TOTAL_BYTES} bytes uncompressed. "
                "Shrink or remove large images and fonts."
            )
        name = _normalize_member_name(member)
        if name is None:
            continue
        _validate_member(member, name, seen_names)
        seen_names.add(name)
        members.append((name, member))
        if len(members) > MAX_BRANDING_ARCHIVE_FILES:
            raise BrandingArchiveError(
                f"{ARCHIVE_ENV_VAR}: archive has more than "
                f"{MAX_BRANDING_ARCHIVE_FILES} files. Remove unused files "
                "from the branding folder."
            )
    skipped = entries - len(members)
    if skipped:
        logger.info("Branding: skipped %d archive root and macOS metadata entries", skipped)
    return members


def _read_validated_members(
    archive_bytes: bytes,
) -> dict[str, bytes]:
    """Validate every archive member, then read the contents.

    The archive is decompressed under a fixed size cap first, then every
    header is checked before any member is read.

    Args:
        archive_bytes: The raw .tar.gz bytes.

    Returns:
        Mapping of file name to file contents.

    Raises:
        BrandingArchiveError: On any rule violation or unreadable archive.
    """
    tar_bytes = _decompress_archive(archive_bytes)
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tar:
            members = _collect_members(tar)
            contents: dict[str, bytes] = {}
            for name, member in members:
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise BrandingArchiveError(
                        f"{ARCHIVE_ENV_VAR}: archive member {member.name!r} has no "
                        f"readable content. {_HOW_TO_BUILD}"
                    )
                data = extracted.read(MAX_BRANDING_ARCHIVE_TOTAL_BYTES + 1)
                if len(data) != member.size:
                    raise BrandingArchiveError(
                        f"{ARCHIVE_ENV_VAR}: archive member {member.name!r} is truncated "
                        f"or corrupt. {_HOW_TO_BUILD}"
                    )
                contents[name] = data
    except (tarfile.TarError, EOFError, OSError) as e:
        raise BrandingArchiveError(
            f"{ARCHIVE_ENV_VAR} does not decode to a readable .tar.gz archive ({e}). "
            f"{_HOW_TO_BUILD}"
        ) from e
    return contents


def _check_target_dir(
    target_dir: Path,
) -> None:
    """Make sure the target folder exists, is writable and holds no files.

    Hidden entries are ignored: a Kubernetes ConfigMap mount with no keys holds
    only ``..data`` style entries, and the repo's compose folder holds
    ``.gitkeep``. Any visible entry means branding files already come from a
    mounted folder.

    Args:
        target_dir: The branding folder (UI_BRANDING_DIR).

    Raises:
        BrandingArchiveError: If the folder cannot be used.
    """
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise BrandingArchiveError(
            f"{DIR_ENV_VAR} ({target_dir}) could not be created ({e.strerror}). Point "
            f"{DIR_ENV_VAR} at a writable folder or unset {ARCHIVE_ENV_VAR}."
        ) from e
    if not target_dir.is_dir():
        raise BrandingArchiveError(
            f"{DIR_ENV_VAR} ({target_dir}) is not a directory. Point {DIR_ENV_VAR} at a folder."
        )

    existing = sorted(
        entry.name for entry in target_dir.iterdir() if not entry.name.startswith(".")
    )
    if existing:
        shown = ", ".join(existing[:5]) + (", ..." if len(existing) > 5 else "")
        raise BrandingArchiveError(
            f"Branding is configured twice: {ARCHIVE_ENV_VAR} is set and {DIR_ENV_VAR} "
            f"({target_dir}) already contains files ({shown}), for example from a "
            f"mounted ConfigMap or folder. Use one source: unset {ARCHIVE_ENV_VAR}, or "
            "remove the mounted branding files."
        )

    if not os.access(target_dir, os.W_OK):
        raise BrandingArchiveError(
            f"{DIR_ENV_VAR} ({target_dir}) is read-only, which means it is a mounted "
            f"folder. Put the branding files in the mounted folder or unset "
            f"{ARCHIVE_ENV_VAR}."
        )


def _write_files(
    target_dir: Path,
    contents: dict[str, bytes],
) -> None:
    """Write the validated files, removing any partial output on failure.

    Files are created exclusively (never overwritten, never through a symlink)
    with mode 0o644.

    Args:
        target_dir: The branding folder.
        contents: Mapping of validated file name to contents.

    Raises:
        BrandingArchiveError: If a file cannot be written.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    written: list[Path] = []
    try:
        for name, data in contents.items():
            path = target_dir / name
            fd = os.open(path, flags, _FILE_MODE)
            written.append(path)
            with os.fdopen(fd, "wb") as f:
                os.fchmod(f.fileno(), _FILE_MODE)
                f.write(data)
    except OSError as e:
        for path in written:
            path.unlink(missing_ok=True)
        raise BrandingArchiveError(
            f"Could not write branding files to {DIR_ENV_VAR} ({target_dir}): "
            f"{e.strerror}. Make sure the folder is writable by the registry user, or "
            f"put the files in a mounted folder and unset {ARCHIVE_ENV_VAR}."
        ) from e


def extract_branding_archive(
    archive_b64: str,
    target_dir: Path,
) -> list[str]:
    """Validate and extract a base64 branding archive into the target folder.

    Args:
        archive_b64: Value of UI_BRANDING_ARCHIVE_B64 (non-empty).
        target_dir: The branding folder (UI_BRANDING_DIR).

    Returns:
        Sorted names of the files written.

    Raises:
        BrandingArchiveError: On any rule violation; nothing is left behind.
    """
    contents = _read_validated_members(_decode_archive(archive_b64))
    if not contents:
        raise BrandingArchiveError(f"{ARCHIVE_ENV_VAR}: archive contains no files. {_HOW_TO_BUILD}")
    _check_target_dir(target_dir)
    _write_files(target_dir, contents)
    return sorted(contents)


def main() -> int:
    """Extract UI_BRANDING_ARCHIVE_B64 into UI_BRANDING_DIR.

    Returns:
        Process exit code: 0 on success or when the archive is unset, 1 on
        any violation.
    """
    archive_b64 = os.environ.get(ARCHIVE_ENV_VAR, "").strip()
    if not archive_b64:
        return 0

    target_dir = Path(os.environ.get(DIR_ENV_VAR, "").strip() or DEFAULT_BRANDING_DIR)
    try:
        names = extract_branding_archive(archive_b64, target_dir)
    except BrandingArchiveError as e:
        logger.error("Branding: %s", e)
        return 1

    logger.info("Branding: wrote %d files to %s: %s", len(names), target_dir, ", ".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
