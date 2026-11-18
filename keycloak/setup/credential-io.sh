#!/bin/bash
# Source before reading credentials. Compatible with Bash 3.2 and newer.
set +x

credential_io_init() {
    # Atomic publication uses Python's renameat support on both macOS and Linux.
    command -v python3 >/dev/null 2>&1 || {
        printf '%s\n' 'python3 is required for secure credential files' >&2
        return 1
    }
    CREDENTIAL_IO_DIR=$(umask 077; mktemp -d "${TMPDIR:-/tmp}/mcp-credentials.XXXXXXXX") || return 1
    trap 'credential_io_cleanup' EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM
}

credential_io_cleanup() {
    if [ -n "${CREDENTIAL_IO_DIR:-}" ]; then
        rm -rf -- "$CREDENTIAL_IO_DIR"
        CREDENTIAL_IO_DIR=
    fi
}

credential_temp_file() {
    [ -n "${CREDENTIAL_IO_DIR:-}" ] && [ -d "$CREDENTIAL_IO_DIR" ] || return 1
    CREDENTIAL_TEMP_FILE=$(umask 077; mktemp "$CREDENTIAL_IO_DIR/data.XXXXXXXX") || return 1
}

credential_bearer_header() {
    case "$1" in
        ''|*$'\r'*|*$'\n'*)
            printf '%s\n' 'Invalid bearer credential' >&2
            return 1
            ;;
    esac
    credential_temp_file || return 1
    CREDENTIAL_HEADER=$CREDENTIAL_TEMP_FILE
    printf 'Authorization: Bearer %s\n' "$1" > "$CREDENTIAL_HEADER"
}

# Shell function arguments never become process arguments. Move sensitive curl
# options to private files before invoking the executable, preserving request bytes.
credential_curl() {
    local option value field
    local args=()
    while [ "$#" -gt 0 ]; do
        option=$1
        shift
        case "$option" in
            -H|--header)
                [ "$#" -gt 0 ] || return 2
                value=$1
                shift
                case "$value" in
                    [Aa][Uu][Tt][Hh][Oo][Rr][Ii][Zz][Aa][Tt][Ii][Oo][Nn]:*)
                        case "$value" in *$'\r'*|*$'\n'*) return 2 ;; esac
                        credential_temp_file || return 1
                        printf '%s\n' "$value" > "$CREDENTIAL_TEMP_FILE" || return 1
                        args+=(--header "@$CREDENTIAL_TEMP_FILE")
                        ;;
                    *) args+=("$option" "$value") ;;
                esac
                ;;
            -d|--data|--data-raw|--data-binary|--data-urlencode)
                [ "$#" -gt 0 ] || return 2
                value=$1
                shift
                case "$option:$value" in
                    --data-urlencode:*=*)
                        field=${value%%=*}
                        credential_temp_file || return 1
                        printf '%s' "${value#*=}" > "$CREDENTIAL_TEMP_FILE" || return 1
                        args+=(--data-urlencode "$field@$CREDENTIAL_TEMP_FILE")
                        ;;
                    --data-urlencode:*@*|--data:@*|-d:@*|--data-binary:@*)
                        args+=("$option" "$value")
                        ;;
                    --data-urlencode:*)
                        credential_temp_file || return 1
                        printf '%s' "$value" > "$CREDENTIAL_TEMP_FILE" || return 1
                        args+=(--data-urlencode "@$CREDENTIAL_TEMP_FILE")
                        ;;
                    *)
                        credential_temp_file || return 1
                        printf '%s' "$value" > "$CREDENTIAL_TEMP_FILE" || return 1
                        args+=(--data-binary "@$CREDENTIAL_TEMP_FILE")
                        ;;
                esac
                ;;
            -v|--verbose|--trace|--trace-ascii|--trace=*|--trace-ascii=*)
                printf '%s\n' 'Credential requests cannot enable curl tracing' >&2
                return 2
                ;;
            *) args+=("$option") ;;
        esac
    done
    command curl -q "${args[@]}"
}
# Prepare only the leaf; never create or chmod checkout/shared parent directories.
# --dedicated permits repairing the known .oauth-tokens directory, not custom paths.
credential_prepare_dir() {
    _credential_path_io prepare "$1" "${2:-}"
}

# Check output safety with disposable empty probes; never change the destination.
credential_preflight() {
    _credential_path_io preflight "$1"
}

# Read bytes from stdin; replace (never truncate/follow) an existing regular file.
credential_write() {
    _credential_path_io write "$1"
}

_credential_path_io() {
    python3 -c '
import os
import secrets
import signal
import stat
import sys

operation = sys.argv[1]
parent, name = os.path.split(os.path.abspath(sys.argv[2]))
dedicated = len(sys.argv) == 4 and sys.argv[3] == "--dedicated"
fd = None
leaf_fd = None
stage = None
probe = None
stage_created = False
probe_created = False

def interrupted(signum, frame):
    raise SystemExit(128 + signum)

for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, interrupted)
try:
    if operation not in ("prepare", "preflight", "write") or not name:
        raise ValueError("invalid credential path operation")
    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    directory = os.fstat(fd)
    if directory.st_uid != os.geteuid() or directory.st_mode & 0o022:
        raise ValueError(
            "parent directory must be owned by this user and not writable by others; "
            "no permissions were changed"
        )
    if operation == "prepare":
        try:
            os.mkdir(name, 0o700, dir_fd=fd)
        except FileExistsError:
            pass
        leaf_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
        leaf = os.fstat(leaf_fd)
        if leaf.st_uid != os.geteuid():
            raise ValueError("credential directory must be owned by this user")
        if dedicated and name == ".oauth-tokens":
            if leaf.st_mode & 0o077:
                os.fchmod(leaf_fd, 0o700)
        elif leaf.st_mode & 0o022:
            raise ValueError(
                "existing credential directory is writable by others; "
                "choose a private output directory or explicitly repair its permissions"
            )
        sys.exit(0)
    try:
        existing = os.stat(name, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        existing = None
    if existing is not None and not stat.S_ISREG(existing.st_mode):
        raise ValueError("credential destination must be a regular file, not a symlink")
    if existing is not None and name not in os.listdir(fd):
        raise ValueError("credential filename aliases an existing name on this filesystem")
    if directory.st_mode & 0o300 != 0o300:
        raise ValueError("credential directory must allow owner write and search access")
    stage = ".credential-" + secrets.token_hex(16)
    out = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
    stage_created = True
    with os.fdopen(out, "wb") as stream:
        if operation == "write":
            while True:
                chunk = sys.stdin.buffer.read(65536)
                if not chunk:
                    break
                stream.write(chunk)
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o600)
    if operation == "preflight":
        # Exercise actual create/sync/replacement permissions without touching the target.
        probe = ".credential-" + secrets.token_hex(16)
        probe_fd = os.open(
            probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd
        )
        probe_created = True
        os.close(probe_fd)
        os.replace(stage, probe, src_dir_fd=fd, dst_dir_fd=fd)
        stage = None
        os.unlink(probe, dir_fd=fd)
        probe = None
    else:
        os.replace(stage, name, src_dir_fd=fd, dst_dir_fd=fd)
        stage = None
except ValueError as exc:
    print("Could not prepare credential output: " + str(exc), file=sys.stderr)
    sys.exit(1)
except OSError as exc:
    print(
        "Could not prepare credential output: "
        + (exc.strerror or "filesystem operation failed")
        + "; require an existing owned, non-shared parent and no symlink at the destination",
        file=sys.stderr,
    )
    sys.exit(1)
finally:
    if leaf_fd is not None:
        os.close(leaf_fd)
    if fd is not None:
        if probe is not None and probe_created:
            try:
                os.unlink(probe, dir_fd=fd)
            except FileNotFoundError:
                pass
        if stage is not None and stage_created:
            try:
                os.unlink(stage, dir_fd=fd)
            except FileNotFoundError:
                pass
        os.close(fd)
' "$@"
}
