FROM public.ecr.aws/docker/library/python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies.
# apt-get upgrade pulls in security patches against the base image's
# already-installed packages (e.g. libc6 / glibc CVE updates) that would
# otherwise stay at whatever version was baked into python:3.14-slim.
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv and create venv
RUN pip install uv==0.11.14 && uv venv .venv --python 3.14

# Copy dependency manifests first so the install layer is cached on the
# (pyproject, lock) pair, not on source changes.
COPY pyproject.toml uv.lock ./

# Install dependencies from uv.lock. --locked fails the build if the
# lock is out of sync with pyproject.toml. If the lock pins a
# supply-chain quarantine cutoff ([options] exclude-newer), re-export it
# as UV_EXCLUDE_NEWER so --locked re-resolves under the SAME constraint
# that produced the lock (otherwise uv treats the missing global cutoff
# as drift and fails). Mirrors .github/workflows/uv-lock-check.yml.
RUN CUTOFF=$(awk '/^\[options\]/{o=1;next} /^\[/{o=0} o&&/^exclude-newer/{gsub(/.*= *"|" *$/,"");print;exit}' uv.lock) && \
    if [ -n "$CUTOFF" ]; then export UV_EXCLUDE_NEWER="$CUTOFF"; echo "Using UV_EXCLUDE_NEWER=$CUTOFF from uv.lock"; fi && \
    uv sync --locked --no-dev

# Copy application
COPY app/ app/
COPY create_api_key.py ./

# Install the app package itself in editable mode (runtime deps already
# installed above by uv sync; --no-deps skips re-resolving).
RUN . .venv/bin/activate && uv pip install --no-deps -e .

# Create data directory
RUN mkdir -p /var/lib/sqlite

# Create non-root user for security (UID 1000 to match metrics-db)
RUN groupadd -g 1000 appuser && useradd -u 1000 -g appuser appuser

# Set ownership of application files and data directory
RUN chown -R appuser:appuser /app /var/lib/sqlite

# Expose port
EXPOSE 8890

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8890/health || exit 1

# Bake the build version into the image (issue #919). Requires
# `docker build --build-arg BUILD_VERSION=$VERSION`; defaults to "1.0.0".
ARG BUILD_VERSION="1.0.0"
ENV BUILD_VERSION=$BUILD_VERSION

# Switch to non-root user
USER appuser

CMD ["/app/.venv/bin/python", "-m", "app.main"]
