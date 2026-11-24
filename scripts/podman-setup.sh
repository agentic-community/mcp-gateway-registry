#!/bin/bash
#
# podman-setup.sh
#
# Podman-specific deployment steps, sourced by build_and_run.sh. Keep this file
# compatible with the bash 3.2 that ships with macOS.

PODMAN_MONGO_IMAGE="mongo:8.2"
PODMAN_MONGO_MIGRATION_CONTAINER="mcp-mongodb-auth-migration"
PODMAN_MONGO_WAIT_SECONDS=60
PODMAN_LOG_DIR="/var/log/containers/ai-registry"
PODMAN_CONTAINER_UID="1000"
PODMAN_FIXED_HOST_PORTS="3000 8000 8002 8003 8200 8890 9090 18080 27017"

# Image builds can exhaust an 8 GB Podman VM when they all run at once.
export COMPOSE_PARALLEL_LIMIT="${COMPOSE_PARALLEL_LIMIT:-2}"


_podman_project_name() {
    if [ -n "${COMPOSE_PROJECT_NAME:-}" ]; then
        printf '%s\n' "$COMPOSE_PROJECT_NAME"
        return 0
    fi
    basename "$PWD" | LC_ALL=C tr 'A-Z' 'a-z'
}


# Print the first process or foreign container holding a port. Containers from
# this project are ignored because the deployment replaces them.
_podman_port_owner() {
    local port="$1"
    local project="$2"
    local row name published own_containers

    own_containers="$(podman ps --filter "label=com.docker.compose.project=${project}" \
        --format '{{.Names}}' 2>/dev/null)"

    while IFS='|' read -r name published; do
        [ -n "$name" ] || continue
        case "$published" in
            *":${port}->"*)
                if ! printf '%s\n' "$own_containers" | grep -qxF "$name"; then
                    printf 'container %s\n' "$name"
                    return 0
                fi
                ;;
        esac
    done <<EOF
$(podman ps --format '{{.Names}}|{{.Ports}}' 2>/dev/null)
EOF

    command -v lsof >/dev/null 2>&1 || return 0
    while read -r row; do
        case "$row" in
            c*)
                case "${row#c}" in
                    gvproxy|krunkit|vfkit|podman*|rootlessport*|pasta*) ;;
                    *) printf 'process %s\n' "${row#c}"; return 0 ;;
                esac
                ;;
        esac
    done <<EOF
$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -Fc 2>/dev/null)
EOF
}


# Replace the Docker URLs shipped in .env.example with their Podman equivalents.
# Only the exact shipped values change, so a customized URL is left alone.
podman_apply_url_defaults() {
    GATEWAY_PORT="${REGISTRY_HTTP_HOST_PORT:-8080}" python3 - <<'PYEOF'
import os
import re

gateway = f"http://localhost:{os.environ['GATEWAY_PORT']}"
keycloak = "http://localhost:18080"
replacements = {
    "REGISTRY_URL": ("http://localhost", gateway),
    "AUTH_SERVER_EXTERNAL_URL": ("http://localhost:8888", gateway),
    "KEYCLOAK_EXTERNAL_URL": ("http://localhost:8080", keycloak),
    "KEYCLOAK_ADMIN_URL": ("http://localhost:8080", keycloak),
}
content = open(".env").read()
for key, (shipped, value) in replacements.items():
    pattern = rf"^{key}={re.escape(shipped)}$"
    content, count = re.subn(pattern, f"{key}={value}", content, flags=re.M)
    if count:
        print(f"Set {key} to {value} for Podman")
fd = os.open(".env.tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as handle:
    handle.write(content)
os.replace(".env.tmp", ".env")
PYEOF
}


# Fail before anything is stopped when a published host port is already taken.
podman_check_ports() {
    local project port owner failed=0
    project="$(_podman_project_name)"

    for port in "${REGISTRY_HTTP_HOST_PORT:-8080}" "${REGISTRY_HTTPS_HOST_PORT:-8443}" \
        "${AUTH_SERVER_HOST_PORT:-8888}" $PODMAN_FIXED_HOST_PORTS; do
        owner="$(_podman_port_owner "$port" "$project")"
        if [ -n "$owner" ]; then
            log "ERROR: host port ${port} is already used by ${owner}."
            failed=1
        fi
    done

    if [ "$failed" -ne 0 ]; then
        log "Stop that service, or move the gateway with REGISTRY_HTTP_HOST_PORT,"
        log "REGISTRY_HTTPS_HOST_PORT or AUTH_SERVER_HOST_PORT in .env."
        return 1
    fi
    log "Required host ports are available"
}


# On macOS, bind mounts for this path resolve inside the Podman VM.
podman_prepare_log_dir() {
    local commands="sudo mkdir -p ${PODMAN_LOG_DIR} \
        && sudo chown -R \$(id -u):\$(id -g) ${PODMAN_LOG_DIR} \
        && podman unshare chown -R ${PODMAN_CONTAINER_UID}:${PODMAN_CONTAINER_UID} ${PODMAN_LOG_DIR} \
        && sudo chmod 0750 ${PODMAN_LOG_DIR} \
        && if command -v selinuxenabled >/dev/null 2>&1 && selinuxenabled; then \
               sudo chcon -R -t container_file_t -l s0 ${PODMAN_LOG_DIR}; \
           fi"

    if [ "$(uname -s)" = "Darwin" ]; then
        log "Preparing ${PODMAN_LOG_DIR} inside the Podman VM..."
        podman machine ssh "$commands"
        return
    fi

    # Rootful Podman uses host UIDs, which the standard script already handles.
    if [ "$(id -u)" -eq 0 ]; then
        "${SCRIPT_DIR}/scripts/prepare-log-dirs.sh"
        return
    fi

    log "Preparing ${PODMAN_LOG_DIR} for rootless Podman..."
    bash -c "$commands"
}


_podman_stop_migration_container() {
    podman rm -f "$PODMAN_MONGO_MIGRATION_CONTAINER" >/dev/null 2>&1 || true
}


_podman_wait_for_mongo_primary() {
    local waited=0 state

    while [ "$waited" -lt "$PODMAN_MONGO_WAIT_SECONDS" ]; do
        state="$(podman exec "$PODMAN_MONGO_MIGRATION_CONTAINER" mongosh --quiet --norc \
            --eval 'print(db.hello().isWritablePrimary)' 2>/dev/null || true)"
        if [ "$state" = "true" ]; then
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done
    return 1
}


# The script travels over stdin so the password never appears on process argv.
# Its output is discarded apart from the success marker, because mongosh can
# echo piped input.
_podman_create_mongo_root_user() {
    local output
    output="$(MIGRATION_USER="$DOCUMENTDB_USERNAME" MIGRATION_PASS="$DOCUMENTDB_PASSWORD" python3 -c '
import json, os
user = json.dumps(os.environ["MIGRATION_USER"])
password = json.dumps(os.environ["MIGRATION_PASS"])
print(f"""db = db.getSiblingDB("admin");
if (!db.getUser({user})) {{
  db.createUser({{user: {user}, pwd: {password}, roles: [{{role: "root", db: "admin"}}]}});
}}
print("MIGRATION_" + "OK");""")
' | podman exec -i "$PODMAN_MONGO_MIGRATION_CONTAINER" mongosh --quiet --norc 2>/dev/null || true)"

    case "$output" in
        *MIGRATION_OK*) return 0 ;;
        *) return 1 ;;
    esac
}


# Volumes from before MongoDB auth have data but no keyfile volume. Add the root
# user while the database is isolated with no network. Run after compose down.
podman_migrate_mongodb_auth() {
    local project data_volume config_volume keyfile_volume
    project="$(_podman_project_name)"
    data_volume="${project}_mongodb-data"
    config_volume="${project}_mongodb-config"
    keyfile_volume="${project}_mongodb-keyfile"

    podman volume exists "$data_volume" || return 0
    if podman volume exists "$keyfile_volume"; then
        return 0
    fi

    log "Adding the MongoDB root user to the existing data volume..."
    _podman_stop_migration_container
    podman run -d --rm --name "$PODMAN_MONGO_MIGRATION_CONTAINER" \
        --hostname mongodb --network none \
        -v "${data_volume}:/data/db" -v "${config_volume}:/data/configdb" \
        "$PODMAN_MONGO_IMAGE" mongod --replSet rs0 --bind_ip 127.0.0.1,mongodb >/dev/null || return 1

    if ! _podman_wait_for_mongo_primary; then
        log "ERROR: MongoDB did not become primary during the auth upgrade."
        log "Inspect it with: podman logs ${PODMAN_MONGO_MIGRATION_CONTAINER}"
        return 1
    fi
    if ! _podman_create_mongo_root_user; then
        _podman_stop_migration_container
        log "ERROR: could not create the MongoDB root user. Your data was not changed."
        return 1
    fi

    _podman_stop_migration_container
    log "MongoDB root user is ready"
}
