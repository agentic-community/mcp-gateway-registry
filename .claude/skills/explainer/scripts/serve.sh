#!/bin/bash
#
# Serve a generated explainer over loopback HTTP and verify it is reachable.
#
# Usage:
#   ./serve.sh .scratchpad/issue-1832/explainer.html
#
# Override the port with EXPLAINER_PORT=9000 ./serve.sh <path>
#
# SECURITY: the server is rooted at the explainer's OWN directory, never at the
# repo root. python3's http.server serves every file under its root, including
# dotfiles, so rooting it at the repo would publish .env and .git/config to any
# local process or local user on the box. Explainers are self-contained (no
# external references), so they need nothing outside their own directory.
# EXPLAINER_SERVE_ROOT can widen the root, and the script warns when it does.
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
    echo "ERROR: file not found: $TARGET"
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
        echo "  $TARGET"
        exit 1
        ;;
esac

BASE_NAME="$(basename "$TARGET")"
SERVE_ROOT="${EXPLAINER_SERVE_ROOT:-$(dirname "$TARGET")}"

if [ -n "${EXPLAINER_SERVE_ROOT:-}" ]; then
    echo "WARNING: EXPLAINER_SERVE_ROOT is set, so every file under"
    echo "  $SERVE_ROOT"
    echo "is readable by any local process while this server runs. Setting it to a"
    echo "repo root publishes .env and .git/ on loopback. Only do this knowingly."
fi

URL="http://$BIND_ADDR:$PORT/$BASE_NAME"
SERVE_CMD="python3 -m http.server $PORT --bind $BIND_ADDR --directory $SERVE_ROOT"


# Echo the HTTP status for a URL, or 000 if nothing answers.
_probe() {
    curl -s -o /dev/null -w "%{http_code}" --max-time 3 "$1" 2>/dev/null || echo "000"
}


# Echo the pid holding the port, or empty if it cannot be read.
_holder_pid() {
    ss -ltnp 2>/dev/null | grep ":$PORT " | grep -o "pid=[0-9]*" | head -1 | cut -d= -f2
}


# Echo the pid only when it is one of our own http.server processes. Never hand
# the user a kill command for a pid we have not positively identified: the port
# could be held by an unrelated service.
_our_server_pid() {
    local pid
    pid="$(_holder_pid)"
    [ -n "$pid" ] || return 0
    if ps -p "$pid" -o args= 2>/dev/null | grep -q "http\.server"; then
        echo "$pid"
    fi
}


_print_copy_block() {
    local url="$1"
    local pid="$2"
    local restart="$SERVE_CMD"

    if [ -n "$pid" ]; then
        restart="kill $pid 2>/dev/null; $SERVE_CMD"
    fi

    echo ""
    echo "----- COPY/PASTE: open this URL -----"
    echo "$url"
    echo "----- COPY/PASTE: run this in a VS Code integrated terminal if the browser cannot reach it -----"
    echo "$restart"
    echo "-------------------------------------"
    echo ""
    echo "Why the second one: VS Code forwards only ports it sees opened in an"
    echo "integrated terminal, so a server started anywhere else listens on this"
    echo "host but is unreachable from the browser (ERR_CONNECTION_REFUSED)."
}


# Reuse an existing server when one already serves this file.
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    if [ "$(_probe "$URL")" = "200" ]; then
        echo "Port $PORT already serves this file. Reusing it."
        _print_copy_block "$URL" "$(_our_server_pid)"
        exit 0
    fi

    echo "ERROR: port $PORT is in use by something that does not serve this file."
    echo "Tried: $URL"
    echo ""
    echo "Holder:"
    ss -ltnp 2>/dev/null | grep ":$PORT " || echo "  (unable to identify)"
    echo ""
    echo "Pick another port with EXPLAINER_PORT=<port> $0 $REL_PATH"
    exit 1
fi

# Log to a private temp file. A predictable /tmp path lets another local user
# pre-create it as a symlink and have this redirect clobber the target.
LOG_FILE="$(mktemp -t "explainer-server-$PORT.XXXXXX")"
chmod 600 "$LOG_FILE"

echo "Starting http.server on $BIND_ADDR:$PORT rooted at $SERVE_ROOT"
nohup python3 -m http.server "$PORT" --bind "$BIND_ADDR" --directory "$SERVE_ROOT" \
    > "$LOG_FILE" 2>&1 &

sleep 1.5

STATUS="$(_probe "$URL")"
if [ "$STATUS" != "200" ]; then
    echo "ERROR: server did not come up cleanly (HTTP $STATUS for $URL)."
    echo "Log tail:"
    tail -5 "$LOG_FILE" 2>/dev/null || true
    exit 1
fi

# Read the pid from ss rather than trusting $!. Depending on how this script is
# invoked, $! can be an intermediate shell that exits immediately, leaving the
# real python3 listener under a different pid, and a kill against the wrong pid
# silently does nothing.
SERVER_PID="$(_our_server_pid)"

echo "Verified HTTP 200 (listener pid ${SERVER_PID:-unknown}). Log: $LOG_FILE"
_print_copy_block "$URL" "$SERVER_PID"
