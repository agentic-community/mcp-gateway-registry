ARG TARGETPLATFORM
FROM --platform=${TARGETPLATFORM} public.ecr.aws/docker/library/python:3.14-slim

WORKDIR /app

# Install system dependencies and uv
# build-essential is required to compile asyncpg from source (no py3.14 wheel yet)
# apt-get upgrade ensures latest security patches (e.g. openssl ~deb13u2)
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    sqlite3 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# Copy dependency files (build-images.sh copies these to .tmp/ directory before build)
COPY .tmp/pyproject.toml .tmp/uv.lock ./

# Install Python dependencies using uv (as root, before switching users).
# --locked fails the build if uv.lock is out of sync with pyproject.toml.
# If the lockfile pins an exclude-newer in its [options] block, export it as
# UV_EXCLUDE_NEWER so `uv sync --locked` re-resolves against the same
# supply-chain quarantine that produced the lock. Without this, newer uv
# versions report the lockfile as out of sync because the global exclude-newer
# is "missing" at sync time. This mirrors .github/workflows/uv-lock-check.yml.
RUN lock_cutoff=$(awk '/^\[options\]/{o=1;next} /^\[/{o=0} o&&/^exclude-newer[[:space:]]*=/{gsub(/.*=[[:space:]]*"|"[[:space:]]*$/,"");print;exit}' uv.lock) && \
    if [ -n "$lock_cutoff" ]; then export UV_EXCLUDE_NEWER="$lock_cutoff"; fi && \
    uv sync --locked --no-dev

# Copy agent code (all files from the context directory)
COPY . ./

# Create non-root user
RUN useradd -m -u 1000 bedrock_agentcore

# Create data directory for SQLite database with proper ownership
# Also fix ownership of installed packages
RUN mkdir -p /app/data && \
    chown -R bedrock_agentcore:bedrock_agentcore /app

USER bedrock_agentcore

# Set environment variables
ENV PYTHONPATH=/app
ENV AWS_REGION=us-east-1
ENV AWS_DEFAULT_REGION=us-east-1

# Expose port for A2A communication (port 9000 for A2A protocol)
EXPOSE 9000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9000/ping || exit 1

# Run the agent with uv (uses the virtual environment created by uv sync)
CMD ["uv", "run", "--no-sync", "agent.py"]
