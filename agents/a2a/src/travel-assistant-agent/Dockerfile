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

# Install Python dependencies using uv (as root, before switching users)
RUN uv sync --frozen --no-dev

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
CMD ["uv", "run", "--no-sync", "server.py"]
