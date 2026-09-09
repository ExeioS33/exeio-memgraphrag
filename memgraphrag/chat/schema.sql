-- Application-level chat storage. Applied idempotently at startup.
--
-- This schema lives in its own database, separate from the RAG's Postgres: product
-- data and knowledge data have different lifecycles, different backup needs, and
-- different blast radii when one of them has to be wiped.
--
-- Note the column is `refs`, not `references`: REFERENCES is a reserved SQL keyword
-- and would need quoting at every single call site.

CREATE TABLE IF NOT EXISTS chat_thread (
    id         TEXT PRIMARY KEY,
    owner      TEXT   NOT NULL DEFAULT 'guest',
    title      TEXT   NOT NULL,
    model      TEXT,
    params     JSONB  NOT NULL DEFAULT '{}'::jsonb,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
);

-- The sidebar lists one owner's threads, most recently touched first.
CREATE INDEX IF NOT EXISTS chat_thread_owner_updated_idx
    ON chat_thread (owner, updated_at DESC);

CREATE TABLE IF NOT EXISTS chat_message (
    id         TEXT PRIMARY KEY,
    thread_id  TEXT   NOT NULL REFERENCES chat_thread (id) ON DELETE CASCADE,
    role       TEXT   NOT NULL,
    content    TEXT   NOT NULL,
    refs       JSONB  NOT NULL DEFAULT '[]'::jsonb,
    created_at BIGINT NOT NULL
);

-- Messages are always read as a whole thread in chronological order.
CREATE INDEX IF NOT EXISTS chat_message_thread_idx
    ON chat_message (thread_id, created_at);

-- Accounts. Two tables in a 1:1, deliberately: app_user carries everything a
-- profile read, an admin listing or a join on chat_thread.owner could want and no
-- credential at all, so none of those code paths can return a password hash by
-- accident. app_auth holds the hash and nothing a normal query needs. The role
-- CHECK is enforced here rather than only in code: an unexpected role is exactly
-- the kind of value that should fail at write time, not at an authorization check.
CREATE TABLE IF NOT EXISTS app_user (
    id         TEXT   PRIMARY KEY,
    email      TEXT   NOT NULL UNIQUE,
    name       TEXT   NOT NULL,
    role       TEXT   NOT NULL DEFAULT 'pending'
               CHECK (role IN ('admin', 'user', 'pending')),
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS app_auth (
    user_id       TEXT    PRIMARY KEY REFERENCES app_user (id) ON DELETE CASCADE,
    password_hash TEXT    NOT NULL,
    active        BOOLEAN NOT NULL DEFAULT TRUE
);
