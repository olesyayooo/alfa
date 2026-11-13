-- Requires the pgvector server files. No source table changes.
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;
CREATE SCHEMA IF NOT EXISTS search;

CREATE TABLE IF NOT EXISTS search.material_embeddings (
    material_type text NOT NULL CHECK (material_type IN ('course','article','document','webinar','podcast')),
    material_id uuid NOT NULL,
    embedding public.vector(1024) NOT NULL,
    content_hash text NOT NULL CHECK (length(content_hash)=64),
    model_name text NOT NULL,
    model_revision text NOT NULL,
    encoder_signature text NOT NULL,
    original_tokens integer NOT NULL CHECK (original_tokens >= 0),
    used_tokens integer NOT NULL CHECK (used_tokens >= 0 AND used_tokens <= original_tokens),
    truncated boolean NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (material_type, material_id),
    CHECK (truncated = (used_tokens < original_tokens)),
    CHECK (public.vector_norm(embedding) > 0)
);

CREATE TABLE IF NOT EXISTS search.semantic_index_state (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    model_name text NOT NULL,
    model_revision text NOT NULL,
    encoder_signature text NOT NULL,
    source_policy text NOT NULL,
    material_count integer NOT NULL CHECK (material_count >= 0),
    updated_at timestamptz NOT NULL DEFAULT now()
);
-- Deliberately no HNSW or IVFFlat index: exact search over the whole small catalog.
