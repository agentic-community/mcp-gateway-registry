FROM quay.io/keycloak/keycloak:25.0 as builder

ENV KC_HEALTH_ENABLED=true
ENV KC_METRICS_ENABLED=true
ENV KC_FEATURES=token-exchange
ENV KC_DB=mysql

WORKDIR /opt/keycloak

RUN keytool -genkeypair -storepass password -storetype PKCS12 -keyalg RSA -keysize 2048 -dname "CN=server" -alias server -ext "SAN:c=DNS:localhost,IP:127.0.0.1" -keystore conf/server.keystore
RUN /opt/keycloak/bin/kc.sh build

FROM quay.io/keycloak/keycloak:25.0

COPY --from=builder /opt/keycloak/ /opt/keycloak/

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

# configuring email listener to send emails for specific events
ENTRYPOINT ["/opt/keycloak/bin/kc.sh", "start", "--optimized"]
