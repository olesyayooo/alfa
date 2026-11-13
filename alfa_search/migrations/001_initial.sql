CREATE SCHEMA IF NOT EXISTS search;

CREATE TABLE IF NOT EXISTS search.materials (
    material_type text NOT NULL CHECK (material_type IN ('course','article','document','webinar','podcast')),
    material_id uuid NOT NULL,
    -- Positions: title, description, topics, tags, industries.
    field_lengths integer[] NOT NULL CHECK (
        array_length(field_lengths, 1) = 5 AND array_ndims(field_lengths) = 1
        AND array_lower(field_lengths, 1) = 1 AND 0 <= ALL(field_lengths)
        AND array_position(field_lengths, NULL) IS NULL),
    indexed_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (material_type, material_id)
);

CREATE TABLE IF NOT EXISTS search.terms (
    term_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    term text NOT NULL UNIQUE CHECK (term <> ''),
    document_frequency integer NOT NULL CHECK (document_frequency > 0)
);

CREATE TABLE IF NOT EXISTS search.postings (
    term_id bigint NOT NULL REFERENCES search.terms(term_id) ON DELETE CASCADE,
    material_type text NOT NULL,
    material_id uuid NOT NULL,
    field smallint NOT NULL CHECK (field BETWEEN 1 AND 5),
    tf integer NOT NULL CHECK (tf > 0),
    PRIMARY KEY (term_id, material_type, material_id, field),
    FOREIGN KEY (material_type, material_id) REFERENCES search.materials(material_type, material_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS postings_material_idx ON search.postings(material_type, material_id);

CREATE TABLE IF NOT EXISTS search.index_stats (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    document_count integer NOT NULL CHECK (document_count >= 0),
    avg_lengths double precision[] NOT NULL CHECK (
        array_length(avg_lengths, 1) = 5 AND array_ndims(avg_lengths) = 1
        AND array_lower(avg_lengths, 1) = 1 AND 0 <= ALL(avg_lengths)
        AND array_position(avg_lengths, NULL) IS NULL),
    normalization_version text NOT NULL,
    source_policy text NOT NULL,
    indexed_at timestamptz NOT NULL DEFAULT now()
);
