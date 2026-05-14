#!/bin/bash
# Install Phase 3 stress-test dependencies: Python Playwright package +
# Chromium browser. ~150 MB Chromium download is first-time only.
#
# Usage:
#   bash tests/stress/playwright_install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

REQUIREMENTS="$SCRIPT_DIR/requirements-stress.txt"
if [ ! -f "$REQUIREMENTS" ]; then
  echo "Missing requirements file: $REQUIREMENTS" >&2
  exit 1
fi

echo "[1/2] Installing Python stress-test dependencies from $REQUIREMENTS ..."
uv pip install -r "$REQUIREMENTS"

echo "[2/2] Installing Playwright Chromium browser ..."
uv run playwright install chromium

echo
echo "Done. Run a UI measurement with:"
echo "  uv run python -m tests.stress.measure_ui_performance \\"
echo "      --backend mongodb-ce --size 100 --base-url http://localhost"
