ARG BASE_IMAGE=quay.io/keycloak/keycloak:26.8.2
FROM ${BASE_IMAGE}

ENV KC_HEALTH_ENABLED=true
ENV KC_METRICS_ENABLED=true
ENV KC_FEATURES=token-exchange
ENV KC_DB=mysql

WORKDIR /opt/keycloak

# Health check for the keycloak ready endpoint.
# The Keycloak base image ships without curl (upstream removes HTTP clients to
# cut attack surface), so a curl-based check can never succeed and leaves the
# container permanently unhealthy. bash's /dev/tcp redirection is what the
# Keycloak health guide documents for HEALTHCHECK. Since Keycloak 25 the health
# endpoints are served on the management port (9000), not on the main HTTP port.
# start-period covers the DB schema migration on first boot.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD ["/bin/bash", "-c", "{ printf 'HEAD /health/ready HTTP/1.0\\r\\n\\r\\n' >&0; grep 'HTTP/1.[01] 200'; } 0<>/dev/tcp/localhost/9000"]

# Switch to keycloak user (provided by base image, UID 1000)
USER keycloak

# Match the official image entrypoint so ECS/CDK can supply the start command.
# Non-optimized startup applies the deployment's build-time KC_* settings.
ENTRYPOINT ["/opt/keycloak/bin/kc.sh"]
CMD ["start"]
