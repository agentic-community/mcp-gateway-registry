#!/bin/bash
set -Eeuo pipefail
umask 077

fail() { echo "Keycloak PostgreSQL bootstrap refused: $*" >&2; exit 1; }
: "${PGDATA:?}" "${BOOTSTRAP_MODE:?}" "${KEYCLOAK_DB_USER:?}" "${KEYCLOAK_DB_NAME:?}"
root="$(dirname "$PGDATA")"
target="$PGDATA"
staging="$target.bootstrap"
pending="$root/.keycloak-bootstrap-pending"
receipt_name=.keycloak-bootstrap-complete
intent="$BOOTSTRAP_MODE:${BOOTSTRAP_SHA256:-}"
mkdir -p "$root"

case "$BOOTSTRAP_MODE" in
  fresh-install) ;;
  restore)
    [ "${BOOTSTRAP_SOURCE_QUIESCED:-}" = true ] || fail "all source writers must remain stopped through cutover"
    [[ "${BOOTSTRAP_SHA256:-}" =~ ^[a-f0-9]{64}$ ]] || fail "a snapshot SHA256 is required"
    ;;
  existing)
    [ "${BOOTSTRAP_EXISTING_VERIFIED:-}" = true ] || fail "existing data must be externally verified"
    ;;
  *) fail "choose fresh-install, restore, or existing explicitly" ;;
esac

# An interrupted attempt cannot be bypassed by choosing another mode or snapshot.
if [ -e "$pending" ]; then
  [ "$(cat "$pending")" = "$intent" ] || fail "unfinished bootstrap has a different intent; retain the volume and resume its original mode and SHA256"
fi
if [ -s "$target/PG_VERSION" ] && [ ! -f "$target/$receipt_name" ]; then
  [ ! -e "$pending" ] && [ "$BOOTSTRAP_MODE" = existing ] || fail "existing PGDATA has no completion receipt; never start it as a fresh database; verify/recover it offline"
fi
if [ ! -s "$target/PG_VERSION" ] && [ "$BOOTSTRAP_MODE" = existing ]; then
  fail "existing mode requires an initialized, externally verified volume; it will not create one"
fi
if [ -e "$target" ] && [ ! -s "$target/PG_VERSION" ]; then
  fail "PGDATA exists without PG_VERSION; preserve it and investigate before moving it aside for recovery"
fi

# docker-entrypoint.sh defines the official image's initialization helpers when sourced.
source /usr/local/bin/docker-entrypoint.sh
server_started=false
stop_server() {
  if [ "$server_started" = true ]; then docker_temp_server_stop; server_started=false; fi
}
trap stop_server EXIT
psql_db() { psql -X -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$KEYCLOAK_DB_NAME" "$@"; }
verify_existing() {
  psql_db -v kc_user="$KEYCLOAK_DB_USER" <<'SQL'
SELECT 1 / (EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'kc_user' AND NOT rolsuper AND NOT rolcreatedb AND NOT rolcreaterole AND NOT rolreplication))::int;
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.realm) OR NOT EXISTS (SELECT 1 FROM public.databasechangelog)
     OR to_regclass('public.user_entity') IS NULL OR to_regclass('public.client') IS NULL THEN
    RAISE EXCEPTION 'Missing initialized Keycloak schema/data; existing/restore verification failed';
  END IF;
END $$;
SQL
}

if [ -f "$target/$receipt_name" ] && [ "$BOOTSTRAP_MODE" != existing ]; then
  [ "$(cat "$target/$receipt_name")" = "$intent" ] || fail "completed volume belongs to another bootstrap intent; use existing only after external verification"
  # Receipt is written only after a successful transaction and a clean shutdown.
  rm -f "$pending"
  exec postgres
fi

if [ "$BOOTSTRAP_MODE" = existing ]; then
  docker_setup_env
  docker_temp_server_start postgres
  server_started=true
  verify_existing
  stop_server
  printf '%s\n' "$intent" > "$target/$receipt_name.tmp"
  sync
  mv "$target/$receipt_name.tmp" "$target/$receipt_name"
  sync
  exec postgres
fi

if [ "$BOOTSTRAP_MODE" = restore ]; then
  [ -r /migration/restore.dump ] || fail "snapshot /migration/restore.dump is missing or unreadable"
  printf '%s  %s\n' "$BOOTSTRAP_SHA256" /migration/restore.dump | sha256sum --check --status || fail "snapshot checksum mismatch; source and archive must be retained"
  pg_restore --list /migration/restore.dump > /dev/null
fi
if [ ! -e "$pending" ]; then
  printf '%s\n' "$intent" > "$pending.tmp"
  sync
  mv "$pending.tmp" "$pending"
  sync
fi

if [ ! -d "$staging" ]; then
  # A crash during initdb never turns an incomplete cluster into PGDATA.
  # Preserve the interrupted staging directory rather than deleting any data.
  if [ -e "$target.initializing" ]; then
    mv "$target.initializing" "$target.incomplete.$(date +%s).$$"
  fi
  export PGDATA="$target.initializing"
  docker_setup_env
  docker_create_db_directories
  docker_verify_minimum_env
  docker_init_database_dir
  pg_setup_hba_conf postgres
  docker_temp_server_start postgres
  server_started=true
  docker_setup_db
  stop_server
  mv "$PGDATA" "$staging"
  sync
fi

export PGDATA="$staging"
docker_setup_env
docker_temp_server_start postgres
server_started=true
# Only bootstrap staging is modified. The source and input archive are read-only.
psql -X -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
  -v kc_user="$KEYCLOAK_DB_USER" -v kc_db="$KEYCLOAK_DB_NAME" <<'SQL'
\getenv kc_password KEYCLOAK_DB_PASSWORD
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION', :'kc_user', :'kc_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'kc_user') \gexec
SELECT format('CREATE DATABASE %I OWNER %I', :'kc_db', :'kc_user')
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = :'kc_db') \gexec
SQL
has_receipt="$(psql_db -Atc "SELECT to_regclass('_mcp_bootstrap.receipt') IS NOT NULL")"
if [ "$has_receipt" = t ]; then
  [ "$(psql_db -Atc 'SELECT intent FROM _mcp_bootstrap.receipt')" = "$intent" ] || fail "staging database receipt does not match snapshot"
else
  sql="$root/.keycloak-bootstrap.sql"
  if [ "$BOOTSTRAP_MODE" = restore ]; then
    # Materialize first: a broken pg_restore pipe must never commit partial SQL.
    pg_restore --no-owner --no-acl --file="$sql" /migration/restore.dump
  else
    : > "$sql"
  fi
  # The restore and receipt commit together. A crash before commit can retry;
  # after commit the receipt prevents replaying the dump into populated tables.
  psql -X -v ON_ERROR_STOP=1 --single-transaction --username "$KEYCLOAK_DB_USER" \
    --dbname "$KEYCLOAK_DB_NAME" -v intent="$intent" -v mode="$BOOTSTRAP_MODE" \
    -f "$sql" -f /opt/keycloak-bootstrap/complete.sql
fi
if [ "$BOOTSTRAP_MODE" = restore ]; then verify_existing; fi
stop_server
printf '%s\n' "$intent" > "$staging/$receipt_name.tmp"
sync
mv "$staging/$receipt_name.tmp" "$staging/$receipt_name"
sync
mv "$staging" "$target"
sync
rm -f "$pending" "$root/.keycloak-bootstrap.sql"
export PGDATA="$target"
exec postgres
