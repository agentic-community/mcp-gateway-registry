#!/bin/bash

# MCP Gateway Registry Stress Test Runner
# This script runs the stress test with reasonable defaults using uv

echo "🚀 Starting MCP Gateway Stress Test with uv..."
echo "=================================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Default options - modify as needed
CONCURRENCY="5,10,15,25"
SKIP_MOCK="--skip-mock"  # Change to "" to create new mock servers
SHOW_UI=""  # Change to "--show-ui" to see browser during UI tests

echo "📋 Configuration:"
echo "   Concurrency levels: $CONCURRENCY"
echo "   Skip mock creation: $SKIP_MOCK"
echo "   Show UI: ${SHOW_UI:-"(headless mode)"}"
echo ""

# Run the stress test
echo "🏃‍♂️ Running stress test..."
uv run complete_mcp_stress_test.py \
    --concurrency "$CONCURRENCY" \
    $SKIP_MOCK \
    $SHOW_UI

# Check if the test completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Stress test completed successfully!"
    echo "📁 Check the results-* directory for your HTML report and JSON data"
    echo ""
    echo "💡 To view the report:"
    echo "   - Open the stress_test_report.html file in your browser"
    echo "   - Or use: python -m http.server 8000 (then visit http://localhost:8000)"
else
    echo ""
    echo "❌ Stress test failed. Check the output above for errors."
    exit 1
fi 