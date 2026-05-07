FROM public.ecr.aws/docker/library/python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY pyproject.toml .
RUN pip install uv && uv pip install --system -e .

# Copy application
COPY app/ app/
COPY create_api_key.py ./

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

CMD ["python", "-m", "app.main"]