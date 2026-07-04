# How do I migrate an existing local MongoDB to authenticated mode?

> **This applies to LOCAL DEVELOPMENT (Docker Compose) ONLY.** It is relevant only if you previously ran the default `docker-compose.yml` with **MongoDB CE and no authentication** and are now pulling a version where the default MongoDB requires authentication.
>
> **You do NOT need this if any of the following are true:**
> - You deploy on **AWS ECS / Terraform** or **EKS / Helm** — those use Amazon DocumentDB (or an external MongoDB), which is already authenticated and unaffected by this change.
> - You used the **`docker-compose.prebuilt.yml`** variant — its MongoDB was already authenticated, so your data volume already has the admin user.
> - This is a **fresh install** — MongoDB creates the admin user on first boot from `DOCUMENTDB_USERNAME` / `DOCUMENTDB_PASSWORD`, so there is nothing to migrate.

## Why this is needed

The default local `docker-compose.yml` now starts MongoDB with `--auth` and creates a root user on first boot from `DOCUMENTDB_USERNAME` / `DOCUMENTDB_PASSWORD`. MongoDB only creates that user when the **data directory is empty** (first boot). If you already have a `mongodb-data` volume that was created **without** authentication, turning on `--auth` leaves you in a state where authentication is required but **no user exists**, so every connection is rejected.

You have two options. Pick based on whether your local data is worth keeping.

---

## Option A: Recreate the volume (simplest — discards local data)

For most local-dev setups the MongoDB data is disposable (servers/agents are re-registered and scopes are re-seeded by `mongodb-init`). Drop and recreate the volume, then let the fresh-boot path create the user.

```bash
cd <your-clone>/mcp-gateway-registry

# 1. Set a strong password in .env (build_and_run.sh also auto-generates one if
#    DOCUMENTDB_PASSWORD is left blank).
#    DOCUMENTDB_USERNAME=admin
#    DOCUMENTDB_PASSWORD=<a strong value>

# 2. Stop MongoDB and remove ONLY its volumes (leaves Keycloak/other data alone).
#    The volume names are prefixed with the compose project name (the directory
#    name by default, e.g. mcp-gateway-registry_mongodb-data).
docker compose stop mongodb mongodb-init
docker compose rm -f mongodb mongodb-init mongodb-keyfile-init
docker volume rm \
  "$(basename "$PWD")_mongodb-data" \
  "$(basename "$PWD")_mongodb-config" \
  "$(basename "$PWD")_mongodb-keyfile" 2>/dev/null || true

# 3. Bring the stack back up. MongoDB creates the root user on first boot,
#    mongodb-init re-seeds collections/scopes.
./build_and_run.sh
```

Verify:

```bash
# Unauthenticated access is now rejected:
docker exec mcp-mongodb mongosh --quiet --eval \
  "db.getSiblingDB('mcp_registry').mcp_servers_default.countDocuments({})"
# -> MongoServerError: ... requires authentication

# Authenticated access succeeds:
docker exec mcp-mongodb mongosh --quiet \
  -u "$DOCUMENTDB_USERNAME" -p "$DOCUMENTDB_PASSWORD" \
  --authenticationDatabase admin --eval "db.adminCommand('ping')"
# -> { ok: 1 }
```

---

## Option B: Keep your data (create the admin user in place)

If you want to preserve the existing local database, create the admin user on the old no-auth volume **before** enabling `--auth`, then switch auth on.

```bash
cd <your-clone>/mcp-gateway-registry

# 1. Start MongoDB WITHOUT auth on the existing volume (temporary), matching the
#    replica set it was created with.
docker run --rm -d --name mongo-migrate \
  -v "$(basename "$PWD")_mongodb-data:/data/db" \
  mongo:8.2 mongod --replSet rs0 --bind_ip 127.0.0.1

# Wait a few seconds for it to accept connections.
sleep 8

# 2. Create the root user (use the SAME username/password you will put in .env).
docker exec mongo-migrate mongosh --quiet admin --eval '
  db.createUser({
    user: "admin",
    pwd:  "REPLACE_WITH_YOUR_STRONG_PASSWORD",
    roles: [ { role: "root", db: "admin" } ]
  })
'

# 3. Stop the temporary instance.
docker rm -f mongo-migrate

# 4. Put the SAME credentials in .env, then start the stack normally (now with --auth).
#    DOCUMENTDB_USERNAME=admin
#    DOCUMENTDB_PASSWORD=REPLACE_WITH_YOUR_STRONG_PASSWORD
./build_and_run.sh
```

> **Note on the keyfile:** the authenticated stack also uses a replica-set keyfile, generated automatically by the `mongodb-keyfile-init` service into the `mongodb-keyfile` volume on startup. You do not need to create it by hand.

Verify the same way as Option A.

---

## Troubleshooting

- **`Authentication failed` from `mongodb-init` or the registry.** The password in `.env` does not match the user stored in the volume. Either fix `.env` to match, or use Option A to start clean.
- **`docker compose up` errors with `required variable DOCUMENTDB_PASSWORD is missing a value`.** That is the intended fail-closed behavior — set `DOCUMENTDB_PASSWORD` in `.env` (or run `build_and_run.sh`, which generates one).
- **`node is not in primary or recovering state`.** The replica set needs to reinitialize after the migration; restart the `mongodb` container and wait for the healthcheck to pass.
