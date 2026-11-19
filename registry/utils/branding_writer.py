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
import fcntl
import hashlib
import io
import json
import logging
import os
import secrets
import stat
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
_OWNERSHIP_FILE: str = ".branding-archive.json"
_OWNERSHIP_MAX_BYTES: int = 16 * 1024
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


def _open_directory_chain(
    target_dir: Path,
) -> int:
    """Open each component before creating children; never follow untrusted links."""
    parts = list(target_dir.absolute().parts[1:])
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open("/", flags)
    links = 0
    trusted_root = True
    try:
        while parts:
            name = parts.pop(0)
            if name in {".", ""}:
                continue
            if name == "..":
                raise BrandingArchiveError(f"{DIR_ENV_VAR} must not contain parent traversal.")
            parent = os.fstat(fd)
            shared = parent.st_mode & 0o022
            sticky_root = parent.st_uid == 0 and parent.st_mode & stat.S_ISVTX
            if parent.st_uid not in {0, os.geteuid()} or (shared and not sticky_root):
                raise BrandingArchiveError(
                    f"{DIR_ENV_VAR} has an untrusted writable ancestor; use a private folder."
                )
            trusted_root = trusted_root and parent.st_uid == 0 and not shared
            try:
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                try:
                    os.mkdir(name, mode=0o755, dir_fd=fd)
                except FileExistsError:
                    # Another startup may create this component before either
                    # writer acquires the target lock. The no-follow open below
                    # still rejects a file or symlink in its place.
                    pass
            else:
                if stat.S_ISLNK(info.st_mode):
                    # macOS /var and /tmp are root-owned aliases. Only the
                    # root-controlled prefix may contain links; after entering
                    # a user-owned or shared tree every component is no-follow.
                    if not trusted_root or info.st_uid != 0 or not parts or links >= 40:
                        raise BrandingArchiveError(
                            f"{DIR_ENV_VAR} contains a symlink; use its canonical directory path."
                        )
                    linked = Path(os.readlink(name, dir_fd=fd))
                    if linked.is_absolute():
                        os.close(fd)
                        fd = os.open("/", flags)
                        parts = list(linked.parts[1:]) + parts
                    else:
                        parts = list(linked.parts) + parts
                    links += 1
                    continue
            child = os.open(name, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _open_target_dir(
    target_dir: Path,
) -> int:
    """Pin a private, writable target directory and serialize startup writers."""
    fd = _open_directory_chain(target_dir)
    try:
        info = os.fstat(fd)
        if info.st_uid != os.geteuid() or info.st_mode & 0o022:
            raise BrandingArchiveError(
                f"{DIR_ENV_VAR} ({target_dir}) must be owned by the registry user and "
                "not writable by other users. Use a dedicated writable branding folder."
            )
        if not info.st_mode & 0o200 or not os.access(target_dir, os.W_OK):
            raise BrandingArchiveError(
                f"{DIR_ENV_VAR} ({target_dir}) is read-only. Use a writable folder or "
                f"unset {ARCHIVE_ENV_VAR}."
            )
        fcntl.flock(fd, fcntl.LOCK_EX)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read_owned_file(
    directory_fd: int,
    name: str,
    limit: int,
    expected_hash: str,
) -> bytes:
    """Read a bounded, exclusively owned regular file without following links."""
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o022
            or info.st_size > limit
        ):
            raise BrandingArchiveError(f"Branding ownership cannot be verified for {name}.")
        data = stream.read(limit + 1)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if len(data) > limit or (info.st_dev, info.st_ino) != (current.st_dev, current.st_ino):
            raise BrandingArchiveError(f"Branding file changed while being read: {name}.")
        if info.st_nlink != 1:
            # A stop between link() and unlink() leaves the complete file and
            # its private staging name. Only discard that exact verified alias.
            aliases = [
                entry
                for entry in os.listdir(directory_fd)
                if entry.startswith(".branding-write-")
                and (alias := os.stat(entry, dir_fd=directory_fd, follow_symlinks=False)).st_ino
                == info.st_ino
                and alias.st_dev == info.st_dev
            ]
            if (
                info.st_nlink != 2
                or len(aliases) != 1
                or hashlib.sha256(data).hexdigest() != expected_hash
            ):
                raise BrandingArchiveError(f"Branding ownership cannot be verified for {name}.")
            os.unlink(aliases[0], dir_fd=directory_fd)
            os.fsync(directory_fd)
        return data


def _publish_file(
    directory_fd: int,
    name: str,
    data: bytes,
    mode: int,
) -> None:
    """Publish a complete file atomically, never replacing an existing entry."""
    temporary = f".branding-write-{secrets.token_hex(16)}"
    fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        os.link(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
    finally:
        os.unlink(temporary, dir_fd=directory_fd)
    os.fsync(directory_fd)


def _write_files(
    directory_fd: int,
    target_dir: Path,
    contents: dict[str, bytes],
) -> None:
    """Verify ownership before resuming an interrupted or completed extraction."""
    expected = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    marker = {"version": 1, "files": expected}
    marker_bytes = json.dumps(marker, sort_keys=True).encode()
    entries = os.listdir(directory_fd)
    visible = {name for name in entries if not name.startswith(".")}
    if _OWNERSHIP_FILE in entries:
        try:
            owned = json.loads(
                _read_owned_file(
                    directory_fd,
                    _OWNERSHIP_FILE,
                    _OWNERSHIP_MAX_BYTES,
                    hashlib.sha256(marker_bytes).hexdigest(),
                )
            )
        except (ValueError, UnicodeError) as e:
            raise BrandingArchiveError(
                "The branding ownership marker is invalid; use a new empty folder."
            ) from e
        if owned != marker:
            raise BrandingArchiveError(
                f"{ARCHIVE_ENV_VAR} differs from the archive recorded in {DIR_ENV_VAR} "
                f"({target_dir}). To change the archive, point {DIR_ENV_VAR} at a new "
                "empty writable folder; existing branding files are not replaced."
            )
        unexpected = visible - expected.keys()
    else:
        unexpected = visible
    if unexpected:
        shown = ", ".join(sorted(unexpected)[:5])
        raise BrandingArchiveError(
            f"Branding is configured twice: {ARCHIVE_ENV_VAR} is set and {DIR_ENV_VAR} "
            f"({target_dir}) contains files not owned by this archive ({shown}). "
            f"Use one source: unset {ARCHIVE_ENV_VAR}, or select a new empty folder."
        )
    # Verify every existing asset before publishing any missing ones. A marker
    # alone never authorizes replacing modified, mounted, or symlinked files.
    for name in visible:
        data = _read_owned_file(
            directory_fd, name, MAX_BRANDING_ARCHIVE_TOTAL_BYTES, expected[name]
        )
        if hashlib.sha256(data).hexdigest() != expected[name]:
            raise BrandingArchiveError(
                f"Branding file {name} differs from its ownership record. "
                f"Use a new empty {DIR_ENV_VAR} or unset {ARCHIVE_ENV_VAR}."
            )
    if _OWNERSHIP_FILE not in entries:
        _publish_file(directory_fd, _OWNERSHIP_FILE, marker_bytes, 0o600)
    # The durable marker precedes asset publication. A stopped startup can
    # safely resume missing files; each visible asset is already complete.
    for name, data in contents.items():
        if name not in visible:
            _publish_file(directory_fd, name, data, _FILE_MODE)


def extract_branding_archive(
    archive_b64: str,
    target_dir: Path,
) -> list[str]:
    """Validate and extract a base64 branding archive into the target folder.

    Args:
        archive_b64: Value of UI_BRANDING_ARCHIVE_B64 (non-empty).
        target_dir: The branding folder (UI_BRANDING_DIR).

    Returns:
        Sorted names of the files verified or written.

    Raises:
        BrandingArchiveError: On invalid input or unverifiable target ownership.
    """
    contents = _read_validated_members(_decode_archive(archive_b64))
    if not contents:
        raise BrandingArchiveError(f"{ARCHIVE_ENV_VAR}: archive contains no files. {_HOW_TO_BUILD}")
    try:
        directory_fd = _open_target_dir(target_dir)
        try:
            _write_files(directory_fd, target_dir, contents)
            current = target_dir.lstat()
            pinned = os.fstat(directory_fd)
            if (current.st_dev, current.st_ino) != (pinned.st_dev, pinned.st_ino):
                raise BrandingArchiveError(
                    f"{DIR_ENV_VAR} changed during extraction; use a stable writable folder."
                )
        finally:
            os.close(directory_fd)
    except OSError as e:
        raise BrandingArchiveError(
            f"Could not apply branding to {DIR_ENV_VAR} ({target_dir}): {e.strerror}. "
            "Use a dedicated writable folder without symlinks; existing files are not replaced."
        ) from e
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

    logger.info(
        "Branding: verified or wrote %d files to %s: %s", len(names), target_dir, ", ".join(names)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
