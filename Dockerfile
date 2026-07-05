# Use an official Python runtime as a parent image
FROM python:3.14-slim

# Set environment variables to prevent interactive prompts during installation
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies including nginx with lua module.
# apt-get upgrade pulls in security patches against the base image's
# already-installed packages (e.g. libc6 / glibc CVE updates) that would
# otherwise stay at whatever version was baked into python:3.14-slim.
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    nginx \
    nginx-extras \
    lua-cjson \
    curl \
    procps \
    openssl \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the application code
COPY . /app/

# Copy nginx configurations (both HTTP-only and HTTP+HTTPS versions)
COPY docker/nginx_rev_proxy_http_only.conf /app/docker/nginx_rev_proxy_http_only.conf
COPY docker/nginx_rev_proxy_http_and_https.conf /app/docker/nginx_rev_proxy_http_and_https.conf

# Copy custom error pages for nginx
COPY docker/502.html /usr/share/nginx/html/502.html

# Make the entrypoint script executable
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
RUN chmod +x /app/docker/entrypoint.sh

# Create nginx lua directories and remove default sites (needed by entrypoint script)
RUN mkdir -p /etc/nginx/lua/virtual_mappings && \
    rm -f /etc/nginx/sites-enabled/default /etc/nginx/sites-available/default && \
    mkdir -p /var/lib/nginx/body /var/lib/nginx/proxy /var/lib/nginx/fastcgi /var/lib/nginx/uwsgi /var/lib/nginx/scgi && \
    mkdir -p /var/log/nginx && \
    mkdir -p /run/nginx

# Expose ports for Nginx (HTTP/HTTPS on high ports for non-root) and the Registry
EXPOSE 8080 8443 7860

# Define environment variables for registry/server configuration.
# BUILD_VERSION is a non-sensitive build arg. SECRET_KEY and POLYGON_API_KEY
# are secrets and are NOT accepted as build args: build args are persisted in
# the image history/metadata and would leak if passed at build time. They are
# injected at runtime only (docker-compose `environment:` / `env_file`), so we
# declare empty ENV placeholders that the runtime value overrides.
ARG BUILD_VERSION="1.0.0"

ENV BUILD_VERSION=$BUILD_VERSION
ENV SECRET_KEY=""
ENV POLYGON_API_KEY=""

# Add health check using the new HTTP endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Create non-root user for security (CIS Docker Benchmark 4.1)
RUN groupadd -g 1000 appuser && useradd -u 1000 -g appuser appuser

# Create security scan directories and certs directory with proper permissions
RUN mkdir -p /app/security_scans /app/skill_security_scans /app/agent_security_scans /app/certs && \
    chown -R appuser:appuser /app/security_scans /app/skill_security_scans /app/agent_security_scans /app/certs

# Set ownership of application files, nginx configs, and entrypoint
RUN chown -R appuser:appuser /app /etc/nginx /var/log/nginx /var/lib/nginx /run/nginx /app/docker/entrypoint.sh

# Switch to non-root user
USER appuser

# Run the entrypoint script when the container launches
ENTRYPOINT ["/app/docker/entrypoint.sh"]
