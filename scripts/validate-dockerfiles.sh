#!/bin/bash
# Validate Dockerfiles for security best practices
# Checks for non-root USER directive in all project Dockerfiles

set -e

echo "Validating Dockerfiles for security best practices..."
echo "=================================================="

# List of Dockerfiles to check
DOCKERFILES=(
    "Dockerfile"
    "docker/Dockerfile.auth"
    "docker/Dockerfile.registry"
    "docker/Dockerfile.registry-cpu"
    "docker/Dockerfile.mcp-server"
    "docker/Dockerfile.mcp-server-cpu"
    "docker/Dockerfile.mcp-server-light"
    "docker/Dockerfile.scopes-init"
    "docker/Dockerfile.metrics-db"
    "docker/Dockerfile.openapi-mcp"
    "docker/keycloak/Dockerfile"
    "metrics-service/Dockerfile"
    "terraform/aws-ecs/grafana/Dockerfile"
)

ERRORS=0
WARNINGS=0

for dockerfile in "${DOCKERFILES[@]}"; do
    if [ ! -f "$dockerfile" ]; then
        echo "❌ ERROR: $dockerfile not found"
        ERRORS=$((ERRORS + 1))
        continue
    fi

    echo ""
    echo "Checking: $dockerfile"
    echo "---"

    # Check for USER directive
    if grep -q "^USER " "$dockerfile"; then
        USER_LINE=$(grep "^USER " "$dockerfile" | tail -1)
        echo "✓ Has USER directive: $USER_LINE"
    else
        echo "❌ ERROR: Missing USER directive"
        ERRORS=$((ERRORS + 1))
    fi

    # Check for HEALTHCHECK directive
    if grep -q "^HEALTHCHECK " "$dockerfile"; then
        echo "✓ Has HEALTHCHECK directive"
    else
        echo "⚠ WARNING: Missing HEALTHCHECK directive"
        WARNINGS=$((WARNINGS + 1))
    fi

    # Check for PIP_NO_CACHE_DIR (Python images only)
    if grep -q "FROM.*python" "$dockerfile" 2>/dev/null; then
        if grep -q "PIP_NO_CACHE_DIR" "$dockerfile"; then
            echo "✓ Has PIP_NO_CACHE_DIR set"
        else
            echo "⚠ WARNING: Python image but missing PIP_NO_CACHE_DIR"
            WARNINGS=$((WARNINGS + 1))
        fi
    fi

    # Check for sudo package (should be removed)
    if grep -q "sudo" "$dockerfile"; then
        echo "❌ ERROR: Contains 'sudo' package (security risk)"
        ERRORS=$((ERRORS + 1))
    else
        echo "✓ No sudo package found"
    fi

    # Check for low-numbered ports in EXPOSE (< 1024 requires root)
    if grep -E "^EXPOSE.*(^| )(80|443|22|21)( |$)" "$dockerfile"; then
        echo "⚠ WARNING: Exposes privileged port (< 1024), requires root or port mapping"
        WARNINGS=$((WARNINGS + 1))
    fi
done

echo ""
echo "=================================================="
echo "Validation Summary:"
echo "  Total Dockerfiles: ${#DOCKERFILES[@]}"
echo "  Errors: $ERRORS"
echo "  Warnings: $WARNINGS"

if [ $ERRORS -gt 0 ]; then
    echo ""
    echo "❌ VALIDATION FAILED"
    exit 1
else
    echo ""
    echo "✅ VALIDATION PASSED"
    if [ $WARNINGS -gt 0 ]; then
        echo "   (with $WARNINGS warnings)"
    fi
    exit 0
fi
