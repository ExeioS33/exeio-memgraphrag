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
