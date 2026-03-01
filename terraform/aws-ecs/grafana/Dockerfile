# Grafana OSS for MCP Gateway Observability Pipeline
# Pinned to stable version to avoid breaking changes
FROM grafana/grafana:12.3.1

# Switch to root to set up directories
USER root

# Install wget for health checks
RUN apk add --no-cache wget

# Copy provisioning configurations
COPY provisioning/datasources /etc/grafana/provisioning/datasources
COPY provisioning/dashboards /etc/grafana/provisioning/dashboards

# Copy dashboard JSON files
COPY dashboards /var/lib/grafana/dashboards

# Set ownership
RUN chown -R grafana:root /etc/grafana/provisioning && \
    chown -R grafana:root /var/lib/grafana/dashboards && \
    chmod -R 755 /var/lib/grafana/dashboards

# Switch back to grafana user
USER grafana

# Authentication defaults (ECS task definition overrides GF_SECURITY_ADMIN_PASSWORD at runtime)
ENV GF_AUTH_ANONYMOUS_ENABLED=false
ENV GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer
ENV GF_AUTH_DISABLE_LOGIN_FORM=false
ENV GF_USERS_ALLOW_SIGN_UP=false

# Server settings for ALB path-based routing
ENV GF_SERVER_ROOT_URL=%(protocol)s://%(domain)s/grafana/
ENV GF_SERVER_SERVE_FROM_SUB_PATH=true

# Logging
ENV GF_LOG_MODE=console
ENV GF_LOG_LEVEL=info

# Performance settings
ENV GF_DASHBOARDS_MIN_REFRESH_INTERVAL=10s

EXPOSE 3000

# Health check
HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD wget -q --spider http://localhost:3000/api/health || exit 1
