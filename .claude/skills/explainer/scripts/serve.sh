#!/bin/bash
#
# Serve a generated explainer over loopback HTTP and verify it is reachable.
#
# Usage:
#   ./serve.sh .scratchpad/issue-1832/explainer.html
#
# Override the port with EXPLAINER_PORT=9000 ./serve.sh <path>
#
# SECURITY: the server is rooted at a private temporary directory containing
# only a copy of the explainer as index.html. python3's http.server serves every
# file below its root and follows symlinks, so serving the explainer's source
# directory could expose sibling files or files reached through a sibling link.
# Explainers are self-contained, so a single copied file is sufficient.
#
# The last thing this prints is a COPY/PASTE block. Relay those two lines to the
# user verbatim, each in its own fenced code block, so the fence copy button
# yields something runnable. Never add a comment inside those fences.

set -eu

BIND_ADDR="127.0.0.1"
REL_PATH="${1:-}"

if [ -z "$REL_PATH" ]; then
    echo "ERROR: pass the repo-relative path to the HTML file."
    echo "Usage: $0 .scratchpad/issue-1832/explainer.html"
    exit 1
fi

# Validate the port before it reaches a command string. This value is
# interpolated into the copy/paste line the user is told to run, so an
# unvalidated port is a paste-jacking vector, not just a bad argument.
PORT="${EXPLAINER_PORT:-8111}"
case "$PORT" in
    ''|*[!0-9]*)
        echo "ERROR: EXPLAINER_PORT must be digits only, got: $PORT"
        exit 1
        ;;
esac
if [ "$PORT" -lt 1024 ] || [ "$PORT" -gt 65535 ]; then
    echo "ERROR: EXPLAINER_PORT must be between 1024 and 65535, got: $PORT"
    exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
TARGET="$REPO_ROOT/$REL_PATH"

if [ ! -f "$TARGET" ]; then
    printf "ERROR: file not found: %q\n" "$TARGET"
    exit 1
fi

# Resolve the path fully, following symlinks, and confirm it is inside the repo.
# Resolving the directory alone is not enough: python3's http.server follows
# symlinks, so a symlinked .html inside the repo would otherwise let this script
# publish any file on the filesystem.
REPO_ROOT="$(cd "$REPO_ROOT" && pwd -P)"
TARGET="$(realpath "$TARGET" 2>/dev/null || readlink -f "$TARGET")"
case "$TARGET" in
    "$REPO_ROOT"/*) ;;
    *)
        echo "ERROR: refusing to serve a path that resolves outside the repo:"
        printf "  %q\n" "$TARGET"
        exit 1
        ;;
esac

# Do not make the source directory (or an environment-selected directory)
# available to http.server. The random, owner-only directory contains one
# regular file and therefore cannot disclose repo files via sibling symlinks.
SERVE_ROOT="$(mktemp -d -t "explainer-server-$PORT.XXXXXX")"
chmod 700 "$SERVE_ROOT"
SERVE_FILE="$SERVE_ROOT/index.html"
cp -- "$TARGET" "$SERVE_FILE"
chmod 600 "$SERVE_FILE"

URL="http://$BIND_ADDR:$PORT/"
# This string is intentionally shown for users to paste into a shell. %q is
# Bash's shell-escaped representation, so unusual checkout paths cannot turn
# into shell syntax in that displayed command.
printf -v SERVE_CMD 'python3 -m http.server %q --bind %q --directory %q' \
    "$PORT" "$BIND_ADDR" "$SERVE_ROOT"


# Echo the HTTP status for a URL, or 000 if nothing answers.
_probe() {
    curl -s -o /dev/null -w "%{http_code}" --max-time 3 "$1" 2>/dev/null || echo "000"
}


_print_copy_block() {
    local url="$1"

    echo ""
    echo "----- COPY/PASTE: open this URL -----"
    echo "$url"
    echo "----- COPY/PASTE: run this in a VS Code integrated terminal if the browser cannot reach it -----"
    echo "$SERVE_CMD"
    echo "-------------------------------------"
    echo ""
    echo "Why the second one: VS Code forwards only ports it sees opened in an"
    echo "integrated terminal, so a server started anywhere else listens on this"
    echo "host but is unreachable from the browser (ERR_CONNECTION_REFUSED)."
}


# Log to a private temp file. A predictable /tmp path lets another local user
# pre-create it as a symlink and have this redirect clobber the target.
LOG_FILE="$(mktemp -t "explainer-server-$PORT.XXXXXX")"
chmod 600 "$LOG_FILE"

echo "Starting http.server on $BIND_ADDR:$PORT rooted at $SERVE_ROOT"
nohup python3 -m http.server "$PORT" --bind "$BIND_ADDR" --directory "$SERVE_ROOT" \
    > "$LOG_FILE" 2>&1 &
SERVER_PID="$!"

sleep 1.5

# Confirm the process we started is alive before accepting a readiness response.
# Otherwise another process that won a bind race could make a 200 response look
# like this server started successfully.
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: http.server exited before becoming ready (the port may be in use)."
    echo "Log tail:"
    tail -5 "$LOG_FILE" 2>/dev/null || true
    exit 1
fi

STATUS="$(_probe "$URL")"
if [ "$STATUS" != "200" ]; then
    echo "ERROR: server did not come up cleanly (HTTP $STATUS for $URL)."
    echo "Log tail:"
    tail -5 "$LOG_FILE" 2>/dev/null || true
    exit 1
fi

echo "Verified HTTP 200 (listener pid $SERVER_PID). Log: $LOG_FILE"
_print_copy_block "$URL"
