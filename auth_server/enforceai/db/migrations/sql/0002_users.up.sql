CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    auth_method TEXT NOT NULL CHECK (auth_method IN ('oidc', 'password')),
    username TEXT NULL,
    email TEXT NOT NULL,
    password_hash TEXT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'user')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_login_at TEXT NULL,
    disabled_at TEXT NULL,
    CHECK (
        (auth_method = 'password' AND username IS NOT NULL AND password_hash IS NOT NULL)
        OR (auth_method = 'oidc' AND password_hash IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username_unique ON users(username) WHERE username IS NOT NULL;

