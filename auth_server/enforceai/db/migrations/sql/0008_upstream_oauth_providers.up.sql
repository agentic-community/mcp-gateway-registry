CREATE TABLE IF NOT EXISTS upstream_oauth_providers (
    provider_id TEXT PRIMARY KEY,
    authorization_endpoint TEXT NOT NULL,
    token_endpoint TEXT NOT NULL,
    client_id TEXT NOT NULL,
    default_scopes_json TEXT NULL,
    extra_authorize_params_json TEXT NULL,
    secret_version INTEGER NULL,
    secret_nonce BLOB NULL,
    secret_ciphertext BLOB NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_upstream_oauth_providers_updated_at
    ON upstream_oauth_providers(updated_at);
