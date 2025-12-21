DROP INDEX IF EXISTS idx_upstream_oauth_states_expires_at;

CREATE TABLE IF NOT EXISTS upstream_oauth_states_new (
    state_id TEXT PRIMARY KEY,
    server_path TEXT NOT NULL,
    credential_type TEXT NOT NULL,
    credential_binding TEXT NOT NULL,
    user_id TEXT NOT NULL,
    agent_id TEXT NULL,
    provider TEXT NOT NULL,
    redirect_uri TEXT NOT NULL,
    secret_version INTEGER NOT NULL,
    secret_nonce BLOB NOT NULL,
    secret_ciphertext BLOB NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    CHECK (credential_type IN ('oauth2','oidc','provider-oauth')),
    CHECK (credential_binding IN ('user','user+agent')),
    CHECK (
        (credential_binding = 'user' AND agent_id IS NULL) OR
        (credential_binding = 'user+agent' AND agent_id IS NOT NULL)
    )
);

INSERT INTO upstream_oauth_states_new (
    state_id,
    server_path,
    credential_type,
    credential_binding,
    user_id,
    agent_id,
    provider,
    redirect_uri,
    secret_version,
    secret_nonce,
    secret_ciphertext,
    created_at,
    expires_at
)
SELECT
    state_id,
    server_path,
    credential_type,
    credential_binding,
    user_id,
    agent_id,
    provider,
    redirect_uri,
    secret_version,
    secret_nonce,
    secret_ciphertext,
    created_at,
    expires_at
FROM upstream_oauth_states;

DROP TABLE IF EXISTS upstream_oauth_states;

ALTER TABLE upstream_oauth_states_new RENAME TO upstream_oauth_states;

CREATE INDEX IF NOT EXISTS idx_upstream_oauth_states_expires_at
    ON upstream_oauth_states(expires_at);

