# How do I rotate my MongoDB password and OpenBao token for a local Docker Compose deployment?

> **This applies to LOCAL DEVELOPMENT (Docker Compose) on EC2 or macOS.** It is for existing users who are pulling a newer version and find that `build_and_run.sh` now refuses to start with an error like:
>
> ```
> ERROR: DOCUMENTDB_PASSWORD is set to the known-weak default 'admin'.
> ERROR: OPENBAO_TOKEN is set to the known-weak default 'dev-root-token'.
> ```
>
> The preflight validator now rejects these known-weak default secrets. You must set strong values in `.env`. The two secrets are handled differently:
>
> - **OpenBao token** — the dev OpenBao runs in-memory and is recreated on every restart, so there is no stored data to preserve. Just set a new value in `.env`.
> - **MongoDB password** — your `mongodb-data` volume already holds a user created with the OLD password. MongoDB only creates the user on first boot (empty data dir), so changing `.env` alone does NOT change the stored user's password. If they diverge, the app is locked out of your existing data. You rotate the password **in place** on the volume first, then set the same value in `.env`.
>
> If you do NOT care about your existing local data (registered servers/agents), you can skip the in-place rotation and instead recreate the volume — see [How do I migrate an existing local MongoDB to authenticated mode?](migrate-local-mongodb-to-authenticated.md).

## OpenBao token (no data to preserve)

The dev OpenBao container (`command: ["server", "-dev"]`) stores everything in memory and is auto-unsealed with the token you provide. Data does not survive a restart, so there is nothing to migrate — just set a strong token.

```bash
cd <your-clone>/mcp-gateway-registry

# Generate a strong token and set it in .env
NEW_OPENBAO_TOKEN=$(openssl rand -hex 32)
sed -i "s|^OPENBAO_TOKEN=.*|OPENBAO_TOKEN=${NEW_OPENBAO_TOKEN}|" .env

# macOS (BSD sed) uses a slightly different in-place flag:
#   sed -i '' "s|^OPENBAO_TOKEN=.*|OPENBAO_TOKEN=${NEW_OPENBAO_TOKEN}|" .env
```

That is all OpenBao needs. Any previously vaulted egress credentials are gone on the next restart regardless (dev mode), so users simply reconnect their accounts via the "Connected Accounts" page after the stack is back up.

## MongoDB password (rotate in place to keep your data)

Your existing `mongodb-data` volume has an `admin` user whose password matches whatever `DOCUMENTDB_PASSWORD` was set to when the volume was first created (typically `admin`). Change that stored user's password to a new strong value **before** updating `.env`, so the two match and you keep every registered server/agent.

### Step 1: Stop the stack

```bash
cd <your-clone>/mcp-gateway-registry
docker compose down
```

### Step 2: Change the stored user's password in place

Boot MongoDB standalone against the existing volume, authenticate with the OLD password, and rotate it. Replace `OLD_PASSWORD` with your current value (usually `admin`) and generate the new one.

```bash
NEW_MONGO_PASSWORD=$(openssl rand -hex 24)
echo "New MongoDB password: ${NEW_MONGO_PASSWORD}"   # note it; you set it in .env in Step 3

# The volume name is the compose project name + _mongodb-data. With the default
# project name it is mcp-gateway-registry_mongodb-data. Confirm with:
#   docker volume ls | grep mongodb-data
VOLUME="mcp-gateway-registry_mongodb-data"

# Start a throwaway mongod on the existing volume, with auth, on a spare port
docker run --rm -d --name mongo-rotate \
  -v "${VOLUME}:/data/db" \
  mongo:8.2 mongod --bind_ip 127.0.0.1 --port 27099 --auth

# Wait for it to accept connections
sleep 8

# Rotate the admin password (uses the OLD password to authenticate)
docker exec mongo-rotate mongosh --port 27099 --quiet \
  -u admin -p 'OLD_PASSWORD' --authenticationDatabase admin --eval "
    db.getSiblingDB('admin').changeUserPassword('admin', '${NEW_MONGO_PASSWORD}');
    print('password changed');
  "

# Stop the throwaway instance (your data stays in the volume)
docker rm -f mongo-rotate
```

Verify the new password works and the old one no longer does:

```bash
docker run --rm -d --name mongo-verify -v "${VOLUME}:/data/db" \
  mongo:8.2 mongod --bind_ip 127.0.0.1 --port 27099 --auth
sleep 8

# NEW password succeeds and your data is intact:
docker exec mongo-verify mongosh --port 27099 --quiet \
  -u admin -p "${NEW_MONGO_PASSWORD}" --authenticationDatabase admin --eval "
    print('ping:', db.adminCommand('ping').ok);
    print('servers:', db.getSiblingDB('mcp_registry').mcp_servers_default.countDocuments({}));
  "

# OLD password now fails (expected):
docker exec mongo-verify mongosh --port 27099 --quiet \
  -u admin -p 'OLD_PASSWORD' --authenticationDatabase admin --eval "db.adminCommand('ping')" \
  || echo "old password correctly rejected"

docker rm -f mongo-verify
```

### Step 3: Set the new password in `.env`

```bash
sed -i "s|^DOCUMENTDB_PASSWORD=.*|DOCUMENTDB_PASSWORD=${NEW_MONGO_PASSWORD}|" .env
# Keep DOCUMENTDB_USERNAME unchanged (the stored user is 'admin'); only the password rotated.

# macOS (BSD sed):
#   sed -i '' "s|^DOCUMENTDB_PASSWORD=.*|DOCUMENTDB_PASSWORD=${NEW_MONGO_PASSWORD}|" .env
```

If you keep alternate IdP config files (`.env.keycloak`, `.env.entra`, `.env.okta`, etc.) that point at the same MongoDB, set the same `DOCUMENTDB_PASSWORD` in each of them too. Do **not** put the real password in `.env.example` (it is a committed template and must stay a placeholder).

### Step 4: Start the stack

```bash
./build_and_run.sh
```

The preflight now passes, MongoDB authenticates with the rotated password, and all your registered servers/agents are still there.

## Verify the whole stack is healthy

```bash
curl -s http://localhost/health        # -> {"status":"healthy",...}
docker compose ps                      # registry / auth-server / mongodb healthy
```

## Troubleshooting

- **`Authentication failed` from `mongodb-init` or the registry after Step 4.** The `DOCUMENTDB_PASSWORD` in `.env` does not match the stored user. Re-check Step 3, or re-run the Step 2 rotation using the correct OLD password.
- **You do not know / cannot recall the OLD password.** If the data is disposable, recreate the volume instead (see [migrate-local-mongodb-to-authenticated.md](migrate-local-mongodb-to-authenticated.md), Option A). That drops the stored user and lets first boot recreate it from `.env`.
- **`build_and_run.sh` still errors on OPENBAO_TOKEN.** Make sure the value is not `dev-root-token` and not empty; the preflight requires a non-default value.
